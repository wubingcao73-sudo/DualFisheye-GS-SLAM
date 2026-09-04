"""Frozen DROID correspondence frontend with Double Sphere rig pose LM.

This module deliberately does not use DROID's pinhole projective operators or
bundle adjustment.  The learned network supplies target pixels and confidence;
``OracleMotionOnlyTracker`` supplies the already validated fisheye pose update.
The target frame ground-truth pose is not part of any public tracker input.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as functional

from hislam2.camera import DoubleSphereCamera
from hislam2.data.frame_types import StereoFisheyeFrame
from hislam2.geom.fisheye_reprojection import FisheyeRigReprojector
from hislam2.range import RangeProvider
from hislam2.tracking.oracle_motion_only import (
    OracleCameraCorrespondences,
    OracleMotionOnlyConfig,
    OracleMotionOnlyResult,
    OracleMotionOnlyTracker,
    OracleMotionProblem,
    bilinear_sample_range,
)


CameraMode = Literal["front", "back", "both"]


def _load_droid_classes():
    """Load the legacy top-level ``modules`` package without changing it."""
    package_root = Path(__file__).resolve().parents[1]
    package_root_string = str(package_root)
    if package_root_string not in sys.path:
        sys.path.insert(0, package_root_string)
    from modules.corr import AltCorrBlock  # type: ignore
    from modules.droid_net import DroidNet  # type: ignore

    return DroidNet, AltCorrBlock


def load_pretrained_droid(
    checkpoint: str | Path,
    device: torch.device | str = "cuda",
) -> torch.nn.Module:
    """Load the frozen DROID feature/context/update subnetworks."""
    DroidNet, _ = _load_droid_classes()
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DROID checkpoint does not exist: {checkpoint}")
    network = DroidNet()
    raw_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = OrderedDict((key.replace("module.", ""), value) for key, value in raw_state.items())
    for key in (
        "update.weight.2.weight",
        "update.weight.2.bias",
        "update.delta.2.weight",
        "update.delta.2.bias",
    ):
        state[key] = state[key][:2]
    network.load_state_dict(state)
    network.requires_grad_(False)
    return network.to(device).eval()


@dataclass(frozen=True)
class DroidFisheyeMotionConfig:
    processed_height: int = 720
    processed_width: int = 720
    encoder_stride: int = 8
    correlation_levels: int = 4
    correlation_radius: int = 3
    outer_iterations: int = 4
    camera_mode: CameraMode = "both"
    minimum_network_weight: float = 1e-4
    target_safety_margin_feature_px: float = 4.0
    use_amp: bool = True
    solver: OracleMotionOnlyConfig = OracleMotionOnlyConfig(
        camera_weighting="balanced",
        huber_threshold_px=1.0,
        invalid_residual_penalty_px=20.0,
        minimum_total_observations=2000,
        minimum_observations_per_camera=500,
        maximum_candidate_invalid_fraction=0.01,
        maximum_iterations=20,
    )

    def __post_init__(self) -> None:
        if self.processed_height < 1 or self.processed_width < 1:
            raise ValueError("processed image dimensions must be positive")
        if self.encoder_stride < 1:
            raise ValueError("encoder_stride must be positive")
        if self.processed_height % self.encoder_stride or self.processed_width % self.encoder_stride:
            raise ValueError("processed image dimensions must be divisible by encoder_stride")
        if self.correlation_levels < 1 or self.correlation_radius < 0:
            raise ValueError("correlation pyramid settings are invalid")
        minimum_pyramid_size = 2 ** (self.correlation_levels - 1)
        if min(self.feature_height, self.feature_width) < minimum_pyramid_size:
            raise ValueError("feature image is too small for the correlation pyramid")
        if self.outer_iterations < 1:
            raise ValueError("outer_iterations must be positive")
        if self.camera_mode not in ("front", "back", "both"):
            raise ValueError("camera_mode must be front, back, or both")
        if not math.isfinite(self.minimum_network_weight) or self.minimum_network_weight <= 0.0:
            raise ValueError("minimum_network_weight must be finite and positive")
        if (
            not math.isfinite(self.target_safety_margin_feature_px)
            or self.target_safety_margin_feature_px < 0.0
        ):
            raise ValueError("target safety margin must be finite and non-negative")

    @property
    def feature_height(self) -> int:
        return self.processed_height // self.encoder_stride

    @property
    def feature_width(self) -> int:
        return self.processed_width // self.encoder_stride


@dataclass(frozen=True)
class DroidFrameFeatures:
    """Pose-free network state for one stereo frame."""

    frame_index: int
    feature_maps: torch.Tensor
    context_hidden: torch.Tensor
    context_input: torch.Tensor

    def __post_init__(self) -> None:
        shapes = (
            self.feature_maps.shape,
            self.context_hidden.shape,
            self.context_input.shape,
        )
        if any(len(shape) != 4 or shape[0] != 2 for shape in shapes):
            raise ValueError("all frame feature tensors must have shape [2, C, H, W]")
        if len({shape[-2:] for shape in shapes}) != 1:
            raise ValueError("frame feature spatial shapes must agree")
        if self.feature_maps.shape[1] != 128:
            raise ValueError("DROID feature maps must have 128 channels")
        if self.context_hidden.shape[1] != 128 or self.context_input.shape[1] != 128:
            raise ValueError("DROID context tensors must have 128 channels")
        device = self.feature_maps.device
        if self.context_hidden.device != device or self.context_input.device != device:
            raise ValueError("frame feature tensors must share a device")
        if not all(tensor.is_floating_point() for tensor in self.tensors):
            raise TypeError("frame feature tensors must be floating point")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in self.tensors):
            raise ValueError("frame feature tensors contain NaN or Inf")

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.feature_maps, self.context_hidden, self.context_input

    def to(self, device: torch.device | str) -> "DroidFrameFeatures":
        return DroidFrameFeatures(
            self.frame_index,
            self.feature_maps.to(device),
            self.context_hidden.to(device),
            self.context_input.to(device),
        )


@dataclass(frozen=True)
class DroidSourceGeometry:
    """Fixed source pixels/range; no target information is stored here."""

    frame_index: int
    source_pixels: torch.Tensor
    source_inverse_range: torch.Tensor
    source_validity: torch.Tensor

    def __post_init__(self) -> None:
        if self.source_pixels.ndim != 3 or self.source_pixels.shape[0] != 2:
            raise ValueError("source_pixels must have shape [2, N, 2]")
        if self.source_pixels.shape[-1] != 2:
            raise ValueError("source_pixels must end in two coordinates")
        expected = self.source_pixels.shape[:2]
        if self.source_inverse_range.shape != expected:
            raise ValueError("source_inverse_range must have shape [2, N]")
        if self.source_validity.shape != expected or self.source_validity.dtype != torch.bool:
            raise ValueError("source_validity must be bool with shape [2, N]")
        if self.source_inverse_range.dtype != self.source_pixels.dtype:
            raise TypeError("source geometry floating dtypes must match")
        if any(tensor.device != self.source_pixels.device for tensor in (
            self.source_inverse_range, self.source_validity
        )):
            raise ValueError("source geometry tensors must share a device")

    def to(self, device: torch.device | str) -> "DroidSourceGeometry":
        return DroidSourceGeometry(
            self.frame_index,
            self.source_pixels.to(device),
            self.source_inverse_range.to(device),
            self.source_validity.to(device),
        )


@dataclass(frozen=True)
class DroidFisheyeOuterIteration:
    iteration: int
    status: str
    pose_before: torch.Tensor
    pose_after: torch.Tensor
    correspondence_count: int
    front_count: int
    back_count: int
    weight_mean: float
    weight_minimum: float
    weight_maximum: float
    lm_result: OracleMotionOnlyResult


@dataclass(frozen=True)
class DroidFisheyeMotionResult:
    T_rig_from_world_target: torch.Tensor
    status: str
    iterations: int
    history: tuple[DroidFisheyeOuterIteration, ...]
    final_problem: Optional[OracleMotionProblem]


def build_feature_camera(
    camera: DoubleSphereCamera,
    processed_width: int,
    processed_height: int,
    encoder_stride: int = 8,
) -> DoubleSphereCamera:
    """Transform native DS calibration through resize and encoder sampling."""
    scale_x = processed_width / camera.width
    scale_y = processed_height / camera.height
    return DoubleSphereCamera(
        camera.xi,
        camera.alpha,
        scale_x * camera.fx / encoder_stride,
        scale_y * camera.fy / encoder_stride,
        (scale_x * (camera.cx + 0.5) - 0.5) / encoder_stride,
        (scale_y * (camera.cy + 0.5) - 0.5) / encoder_stride,
        processed_width // encoder_stride,
        processed_height // encoder_stride,
    )


def feature_to_native_pixels(
    pixels: torch.Tensor,
    camera: DoubleSphereCamera,
    processed_width: int,
    processed_height: int,
    encoder_stride: int = 8,
) -> torch.Tensor:
    """Map feature centers to native pixels for ``align_corners=False`` resize."""
    if not isinstance(pixels, torch.Tensor) or pixels.shape[-1] != 2:
        raise ValueError("pixels must be a tensor ending in shape [2]")
    scale_x = processed_width / camera.width
    scale_y = processed_height / camera.height
    u = (encoder_stride * pixels[..., 0] + 0.5) / scale_x - 0.5
    v = (encoder_stride * pixels[..., 1] + 0.5) / scale_y - 0.5
    return torch.stack((u, v), dim=-1)


def feature_to_processed_pixels(pixels: torch.Tensor, encoder_stride: int = 8) -> torch.Tensor:
    """Map feature coordinates to the corresponding resized-image centers."""
    return encoder_stride * pixels + 0.5


def _feature_grid(config: DroidFisheyeMotionConfig, dtype, device) -> torch.Tensor:
    rows = torch.arange(config.feature_height, dtype=dtype, device=device)
    columns = torch.arange(config.feature_width, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _safety_valid(camera: DoubleSphereCamera, pixels: torch.Tensor, margin: float) -> torch.Tensor:
    offsets = pixels.new_tensor(
        ((0.0, 0.0), (margin, 0.0), (-margin, 0.0), (0.0, margin),
         (0.0, -margin), (margin, margin), (margin, -margin),
         (-margin, margin), (-margin, -margin))
    )
    candidates = pixels[:, None] + offsets[None]
    return (camera.valid_mask(candidates) & camera.image_valid(candidates)).all(dim=1)


class DroidFisheyeMotionTracker:
    """Alternate frozen DROID updates with the independent fisheye pose LM."""

    def __init__(
        self,
        native_reprojector: FisheyeRigReprojector,
        range_provider: RangeProvider,
        network: torch.nn.Module,
        config: DroidFisheyeMotionConfig = DroidFisheyeMotionConfig(),
        *,
        corr_block_factory: Optional[Callable] = None,
        device: torch.device | str = "cuda",
    ) -> None:
        if len(native_reprojector.cameras) != 2:
            raise ValueError("DROID fisheye tracking requires exactly two cameras")
        if not all(isinstance(camera, DoubleSphereCamera) for camera in native_reprojector.cameras):
            raise TypeError("both cameras must be DoubleSphereCamera instances")
        self.config = config
        self.device = torch.device(device)
        self.range_provider = range_provider
        self.network = network.to(self.device).eval().requires_grad_(False)
        self.native_cameras = tuple(native_reprojector.cameras)
        feature_cameras = tuple(
            build_feature_camera(
                camera,
                config.processed_width,
                config.processed_height,
                config.encoder_stride,
            )
            for camera in native_reprojector.cameras
        )
        extrinsics = native_reprojector.T_rig_from_camera.to(
            device=self.device, dtype=torch.float32
        )
        self.feature_reprojector = FisheyeRigReprojector(feature_cameras, extrinsics)
        self.pose_tracker = OracleMotionOnlyTracker(self.feature_reprojector, config.solver)
        if corr_block_factory is None:
            _, corr_block_factory = _load_droid_classes()
        self.corr_block_factory = corr_block_factory

    def extract_features(self, rgb: torch.Tensor, frame_index: int) -> DroidFrameFeatures:
        """Extract pose-free stereo DROID features from native uint8 RGB."""
        if not isinstance(rgb, torch.Tensor) or rgb.ndim != 4 or rgb.shape[:2] != (2, 3):
            raise ValueError("rgb must have shape [2, 3, H, W]")
        images = functional.interpolate(
            rgb.to(device=self.device, dtype=torch.float32),
            size=(self.config.processed_height, self.config.processed_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[None]
        amp_enabled = self.config.use_amp and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=amp_enabled
        ):
            feature_maps, hidden, context = self.network.extract_features(images)
        return DroidFrameFeatures(
            int(frame_index),
            feature_maps[0].detach(),
            hidden[0].detach(),
            context[0].detach(),
        )

    def prepare_source_geometry(self, frame: StereoFisheyeFrame) -> DroidSourceGeometry:
        """Sample only the source frame GT Euclidean range at feature rays."""
        observation = self.range_provider.provide(frame)
        pixels = _feature_grid(self.config, torch.float32, self.device)
        all_pixels, all_inverse_range, all_valid = [], [], []
        for camera_index, (source_native_camera, feature_camera) in enumerate(
            zip(self.native_cameras, self.feature_reprojector.cameras)
        ):
            native_pixels = feature_to_native_pixels(
                pixels,
                source_native_camera,
                self.config.processed_width,
                self.config.processed_height,
                self.config.encoder_stride,
            )
            range_m = observation.range_m[camera_index].to(
                device=self.device, dtype=torch.float32
            )
            range_valid = observation.observation_valid[camera_index].to(self.device)
            sampled_range, sampled_valid = bilinear_sample_range(
                range_m, range_valid, native_pixels
            )
            rays_valid = feature_camera.valid_mask(pixels) & feature_camera.image_valid(pixels)
            valid = sampled_valid & rays_valid
            inverse_range = torch.where(valid, sampled_range.reciprocal(), torch.zeros_like(sampled_range))
            all_pixels.append(pixels)
            all_inverse_range.append(inverse_range)
            all_valid.append(valid)
        return DroidSourceGeometry(
            int(frame.index),
            torch.stack(all_pixels),
            torch.stack(all_inverse_range),
            torch.stack(all_valid),
        )

    def _edges(self, mode: CameraMode) -> tuple[list[int], list[int], list[int]]:
        camera_indices = {"front": [0], "back": [1], "both": [0, 1]}[mode]
        return camera_indices, [2 * index for index in camera_indices], [2 * index + 1 for index in camera_indices]

    def _project(
        self,
        geometry: DroidSourceGeometry,
        source_pose: torch.Tensor,
        target_pose: torch.Tensor,
        camera_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates, validity = [], []
        for index in camera_indices:
            projection = self.feature_reprojector.reproject(
                geometry.source_pixels[index],
                geometry.source_inverse_range[index],
                source_pose,
                target_pose,
                index,
                index,
            )
            coordinates.append(
                projection.pixels.reshape(self.config.feature_height, self.config.feature_width, 2)
            )
            validity.append(projection.validity.geometric_valid.reshape(-1))
        return torch.stack(coordinates)[None], torch.stack(validity)

    def _build_problem(
        self,
        geometry: DroidSourceGeometry,
        source_pose: torch.Tensor,
        predicted: torch.Tensor,
        predicted_validity: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        camera_indices: list[int],
    ) -> OracleMotionProblem:
        groups: list[Optional[OracleCameraCorrespondences]] = [None, None]
        for edge_index, camera_index in enumerate(camera_indices):
            observed = target[0, edge_index].reshape(-1, 2).float()
            predicted_pixels = predicted[0, edge_index].reshape(-1, 2).float()
            component_weights = weight[0, edge_index].reshape(-1, 2).float()
            base_weight = torch.sqrt(torch.clamp(component_weights.prod(dim=-1), min=0.0))
            finite = (
                torch.isfinite(observed).all(dim=-1)
                & torch.isfinite(predicted_pixels).all(dim=-1)
                & torch.isfinite(base_weight)
            )
            target_safe = _safety_valid(
                self.feature_reprojector.cameras[camera_index],
                observed,
                self.config.target_safety_margin_feature_px,
            )
            selected = (
                geometry.source_validity[camera_index]
                & predicted_validity[edge_index]
                & finite
                & target_safe
                & (base_weight > self.config.minimum_network_weight)
            )
            indices = torch.nonzero(selected, as_tuple=False).flatten()
            groups[camera_index] = OracleCameraCorrespondences(
                geometry.source_pixels[camera_index, indices].float().detach().clone(),
                geometry.source_inverse_range[camera_index, indices].float().detach().clone(),
                observed[indices].detach().clone(),
                camera_index,
                camera_index,
                torch.ones(len(indices), dtype=torch.bool, device=self.device),
                base_weight[indices].detach().clone(),
            )
        return OracleMotionProblem(source_pose.detach().clone(), groups[0], groups[1])

    def track_pair(
        self,
        source_features: DroidFrameFeatures,
        target_features: DroidFrameFeatures,
        source_geometry: DroidSourceGeometry,
        fixed_source_pose: torch.Tensor,
        initial_target_pose: torch.Tensor,
        *,
        camera_mode: Optional[CameraMode] = None,
    ) -> DroidFisheyeMotionResult:
        """Track one pair without accepting any target GT pose or target frame."""
        mode = self.config.camera_mode if camera_mode is None else camera_mode
        if mode not in ("front", "back", "both"):
            raise ValueError("camera_mode must be front, back, or both")
        source_features = source_features.to(self.device)
        target_features = target_features.to(self.device)
        source_geometry = source_geometry.to(self.device)
        if source_features.frame_index != source_geometry.frame_index:
            raise ValueError("source features and geometry refer to different frames")
        if source_features.feature_maps.shape[-2:] != (
            self.config.feature_height, self.config.feature_width
        ):
            raise ValueError("source feature resolution does not match tracker config")
        source_pose = fixed_source_pose.to(device=self.device, dtype=torch.float32)
        pose = initial_target_pose.to(device=self.device, dtype=torch.float32)
        camera_indices, ii_values, jj_values = self._edges(mode)
        sequence_maps = []
        for camera_index in range(2):
            sequence_maps.extend(
                (source_features.feature_maps[camera_index], target_features.feature_maps[camera_index])
            )
        feature_maps = torch.stack(sequence_maps)[None]
        ii = torch.tensor(ii_values, dtype=torch.long, device=self.device)
        jj = torch.tensor(jj_values, dtype=torch.long, device=self.device)
        corr_block = self.corr_block_factory(
            feature_maps,
            num_levels=self.config.correlation_levels,
            radius=self.config.correlation_radius,
        )
        net = source_features.context_hidden[camera_indices][None]
        context = source_features.context_input[camera_indices][None]
        coords0 = geometry_grid = _feature_grid(self.config, torch.float32, self.device)
        coords0 = geometry_grid.reshape(
            self.config.feature_height, self.config.feature_width, 2
        )[None, None].expand(1, len(camera_indices), -1, -1, -1)
        previous_target: Optional[torch.Tensor] = None
        outer_history = []
        final_problem = None
        status = "numerical_failure"
        amp_enabled = self.config.use_amp and self.device.type == "cuda"

        with torch.inference_mode():
            for outer_iteration in range(self.config.outer_iterations):
                predicted, predicted_validity = self._project(
                    source_geometry, source_pose, pose, camera_indices
                )
                if previous_target is None:
                    previous_target = predicted.detach().clone()
                corr = corr_block(predicted, ii, jj)
                motion = torch.cat((predicted - coords0, previous_target - predicted), dim=-1)
                motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)
                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    net, delta, weight = self.network.update(net, context, corr, motion)
                target = predicted + delta.float()
                problem = self._build_problem(
                    source_geometry,
                    source_pose,
                    predicted,
                    predicted_validity,
                    target,
                    weight,
                    camera_indices,
                )
                pose_before = pose.detach().clone()
                lm_result = self.pose_tracker.optimize(problem, pose)
                pose = lm_result.T_rig_from_world_target
                previous_target = target.detach().clone()
                counts = [0, 0]
                for group in problem.groups:
                    counts[group.source_camera_index] = group.count
                selected_weights = torch.cat([group.base_weights for group in problem.groups])
                outer_history.append(
                    DroidFisheyeOuterIteration(
                        outer_iteration,
                        lm_result.status,
                        pose_before,
                        pose.detach().clone(),
                        problem.fixed_count,
                        counts[0],
                        counts[1],
                        float(selected_weights.mean()),
                        float(selected_weights.min()),
                        float(selected_weights.max()),
                        lm_result,
                    )
                )
                final_problem = problem
                status = lm_result.status
                if status in ("insufficient_observations", "numerical_failure"):
                    break

        return DroidFisheyeMotionResult(
            pose.detach().clone(), status, len(outer_history), tuple(outer_history), final_problem
        )


__all__ = [
    "CameraMode",
    "DroidFisheyeMotionConfig",
    "DroidFisheyeMotionResult",
    "DroidFisheyeMotionTracker",
    "DroidFisheyeOuterIteration",
    "DroidFrameFeatures",
    "DroidSourceGeometry",
    "build_feature_camera",
    "feature_to_native_pixels",
    "feature_to_processed_pixels",
    "load_pretrained_droid",
]
