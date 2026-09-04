"""Oracle motion-only pose optimization for a calibrated dual-fisheye rig.

Ground-truth target poses are deliberately confined to
``build_oracle_motion_problem``.  The optimizer consumes only fixed pixel
correspondences, a fixed source pose, and an independently supplied target
pose initial estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as functional
from lietorch import SE3

from hislam2.data.frame_types import StereoFisheyeFrame
from hislam2.geom.fisheye_reprojection import FisheyeRigReprojector
from hislam2.range import RangeProvider


TrackerStatus = Literal[
    "converged",
    "max_iterations",
    "insufficient_observations",
    "numerical_failure",
]


def _require_floating_tensor(value: torch.Tensor, shape_suffix: tuple[int, ...], name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if value.shape[-len(shape_suffix) :] != shape_suffix:
        raise ValueError(f"{name} must end in shape {shape_suffix}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")


def _require_transform(value: torch.Tensor, name: str) -> None:
    _require_floating_tensor(value, (4, 4), name)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4], got {tuple(value.shape)}")
    tolerance = 1e-5 if value.dtype == torch.float32 else 1e-7
    expected_bottom = value.new_tensor((0.0, 0.0, 0.0, 1.0))
    if not bool(torch.allclose(value[3], expected_bottom, atol=tolerance, rtol=0.0)):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = value[:3, :3]
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    if not bool(
        torch.allclose(rotation.transpose(0, 1) @ rotation, identity, atol=tolerance, rtol=tolerance)
    ) or not bool(
        torch.allclose(torch.det(rotation), value.new_tensor(1.0), atol=tolerance, rtol=tolerance)
    ):
        raise ValueError(f"{name} rotation is not a proper orthonormal matrix")


@dataclass(frozen=True)
class OracleCameraCorrespondences:
    """One fixed same-camera temporal correspondence set."""

    source_pixels: torch.Tensor
    source_inverse_range: torch.Tensor
    observed_target_pixels: torch.Tensor
    source_camera_index: int
    target_camera_index: int
    fixed_validity: torch.Tensor
    base_weights: torch.Tensor

    def __post_init__(self) -> None:
        _require_floating_tensor(self.source_pixels, (2,), "source_pixels")
        _require_floating_tensor(
            self.observed_target_pixels, (2,), "observed_target_pixels"
        )
        count = self.source_pixels.shape[0]
        if self.source_pixels.ndim != 2:
            raise ValueError("source_pixels must have shape [N, 2]")
        if self.observed_target_pixels.shape != (count, 2):
            raise ValueError("observed_target_pixels must have shape [N, 2]")
        for name in ("source_inverse_range", "base_weights"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != (count,):
                raise ValueError(f"{name} must have shape [N]")
            if value.dtype != self.source_pixels.dtype:
                raise TypeError(f"{name} must match source_pixels dtype")
            if value.device != self.source_pixels.device:
                raise ValueError(f"{name} must match source_pixels device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN or Inf")
        if not bool((self.source_inverse_range > 0.0).all()):
            raise ValueError("source_inverse_range must be positive")
        if not bool((self.base_weights > 0.0).all()):
            raise ValueError("base_weights must be positive")
        if not isinstance(self.fixed_validity, torch.Tensor) or self.fixed_validity.shape != (count,):
            raise ValueError("fixed_validity must have shape [N]")
        if self.fixed_validity.dtype != torch.bool:
            raise TypeError("fixed_validity must use bool dtype")
        if self.fixed_validity.device != self.source_pixels.device:
            raise ValueError("fixed_validity must match source_pixels device")
        if not bool(self.fixed_validity.all()):
            raise ValueError("builder output must contain only fixed-valid correspondences")
        if self.observed_target_pixels.dtype != self.source_pixels.dtype:
            raise TypeError("pixel tensors must use the same dtype")
        if self.observed_target_pixels.device != self.source_pixels.device:
            raise ValueError("pixel tensors must use the same device")
        if not isinstance(self.source_camera_index, int) or self.source_camera_index < 0:
            raise ValueError("source_camera_index must be a non-negative integer")
        if not isinstance(self.target_camera_index, int) or self.target_camera_index < 0:
            raise ValueError("target_camera_index must be a non-negative integer")

    @property
    def count(self) -> int:
        return int(self.source_pixels.shape[0])


@dataclass(frozen=True)
class OracleMotionProblem:
    """Fixed optimizer input.  It intentionally has no target ground truth."""

    T_rig_from_world_source: torch.Tensor
    front: Optional[OracleCameraCorrespondences]
    back: Optional[OracleCameraCorrespondences]

    def __post_init__(self) -> None:
        _require_transform(self.T_rig_from_world_source, "T_rig_from_world_source")
        if self.front is None and self.back is None:
            raise ValueError("at least one camera correspondence group is required")
        for index, group in enumerate((self.front, self.back)):
            if group is None:
                continue
            if group.source_camera_index != index or group.target_camera_index != index:
                raise ValueError("front/back groups must be same-camera temporal pairs")
            if group.source_pixels.dtype != self.T_rig_from_world_source.dtype:
                raise TypeError("problem tensors and source pose must use the same dtype")
            if group.source_pixels.device != self.T_rig_from_world_source.device:
                raise ValueError("problem tensors and source pose must use the same device")

    @property
    def fixed_count(self) -> int:
        return sum(group.count for group in self.groups)

    @property
    def groups(self) -> tuple[OracleCameraCorrespondences, ...]:
        return tuple(group for group in (self.front, self.back) if group is not None)

    def camera_mode(self, mode: Literal["front", "back", "both"]) -> "OracleMotionProblem":
        if mode == "front":
            if self.front is None:
                raise ValueError("problem has no front correspondences")
            return OracleMotionProblem(self.T_rig_from_world_source, self.front, None)
        if mode == "back":
            if self.back is None:
                raise ValueError("problem has no back correspondences")
            return OracleMotionProblem(self.T_rig_from_world_source, None, self.back)
        if mode == "both":
            return self
        raise ValueError(f"unknown camera mode: {mode}")

    def with_source_pose(self, pose: torch.Tensor) -> "OracleMotionProblem":
        """Return the same fixed correspondences anchored at a different source estimate."""
        return OracleMotionProblem(pose.detach().clone(), self.front, self.back)


@dataclass(frozen=True)
class OracleMotionOnlyConfig:
    camera_weighting: Literal["all", "balanced"] = "all"
    huber_threshold_px: float = 5.0
    initial_damping: float = 1e-3
    accepted_damping_scale: float = 0.3
    rejected_damping_scale: float = 10.0
    maximum_retries: int = 6
    maximum_iterations: int = 20
    maximum_translation_step_m: float = 0.1
    maximum_rotation_step_rad: float = math.radians(5.0)
    minimum_total_observations: int = 2000
    minimum_observations_per_camera: int = 500
    maximum_candidate_invalid_fraction: float = 0.01
    invalid_residual_penalty_px: float = 100.0
    diagonal_epsilon: float = 1e-12
    translation_tolerance_m: float = 1e-6
    rotation_tolerance_rad: float = 1e-6
    relative_cost_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.camera_weighting not in ("all", "balanced"):
            raise ValueError("camera_weighting must be 'all' or 'balanced'")
        positive = (
            "huber_threshold_px",
            "initial_damping",
            "accepted_damping_scale",
            "rejected_damping_scale",
            "maximum_translation_step_m",
            "maximum_rotation_step_rad",
            "invalid_residual_penalty_px",
            "diagonal_epsilon",
            "translation_tolerance_m",
            "rotation_tolerance_rad",
            "relative_cost_tolerance",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_retries < 1 or self.maximum_iterations < 1:
            raise ValueError("retry and iteration limits must be positive")
        if self.minimum_total_observations < 1 or self.minimum_observations_per_camera < 1:
            raise ValueError("observation limits must be positive")
        if not 0.0 <= self.maximum_candidate_invalid_fraction < 1.0:
            raise ValueError("maximum_candidate_invalid_fraction must lie in [0, 1)")


@dataclass(frozen=True)
class OracleMotionIteration:
    iteration: int
    retry: int
    damping: float
    current_cost: float
    candidate_cost: float
    accepted: bool
    valid_count: int
    candidate_invalid_count: int
    translation_step_m: float
    rotation_step_rad: float
    hessian_minimum_eigenvalue: float
    hessian_maximum_eigenvalue: float
    hessian_condition_number: float


@dataclass(frozen=True)
class OracleMotionOnlyResult:
    T_rig_from_world_target: torch.Tensor
    status: TrackerStatus
    iterations: int
    initial_cost: float
    final_cost: float
    valid_count: int
    camera_scales: tuple[float, float]
    hessian_eigenvalues: tuple[float, ...]
    hessian_condition_number: float
    final_damping: float
    history: tuple[OracleMotionIteration, ...]


@dataclass(frozen=True)
class _Evaluation:
    cost: torch.Tensor
    valid_count: int
    invalid_count: int
    per_camera_valid: tuple[int, int]
    hessian: Optional[torch.Tensor]
    gradient: Optional[torch.Tensor]


def bilinear_sample_range(
    range_m: torch.Tensor,
    observation_valid: torch.Tensor,
    pixels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a range image at pixels using the exact align_corners=True map."""
    if not isinstance(range_m, torch.Tensor) or range_m.ndim != 2:
        raise ValueError("range_m must have shape [H, W]")
    if range_m.dtype not in (torch.float32, torch.float64):
        raise TypeError("range_m must use float32 or float64")
    if not isinstance(observation_valid, torch.Tensor) or observation_valid.shape != range_m.shape:
        raise ValueError("observation_valid must match range_m shape")
    if observation_valid.dtype != torch.bool:
        raise TypeError("observation_valid must use bool dtype")
    _require_floating_tensor(pixels, (2,), "pixels")
    if pixels.ndim != 2:
        raise ValueError("pixels must have shape [N, 2]")
    if pixels.dtype != range_m.dtype or pixels.device != range_m.device:
        raise ValueError("pixels and range_m must use the same dtype and device")
    height, width = range_m.shape
    if width < 2 or height < 2:
        raise ValueError("bilinear sampling requires width and height of at least two")

    u, v = pixels.unbind(dim=-1)
    finite = torch.isfinite(pixels).all(dim=-1)
    in_bounds = finite & (u >= 0.0) & (u < width - 1) & (v >= 0.0) & (v < height - 1)
    safe_u = torch.where(in_bounds, u, torch.zeros_like(u))
    safe_v = torch.where(in_bounds, v, torch.zeros_like(v))
    x0 = torch.floor(safe_u).long()
    y0 = torch.floor(safe_v).long()
    x1 = x0 + 1
    y1 = y0 + 1
    four_valid = (
        observation_valid[y0, x0]
        & observation_valid[y0, x1]
        & observation_valid[y1, x0]
        & observation_valid[y1, x1]
    )
    normalized_x = 2.0 * safe_u / (width - 1) - 1.0
    normalized_y = 2.0 * safe_v / (height - 1) - 1.0
    grid = torch.stack((normalized_x, normalized_y), dim=-1)[None, None]
    sampled = functional.grid_sample(
        range_m[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(-1)
    valid = in_bounds & four_valid & torch.isfinite(sampled) & (sampled > 0.0)
    return torch.where(valid, sampled, torch.zeros_like(sampled)), valid


def _sample_scalar_image(image: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    height, width = image.shape
    u, v = pixels.unbind(dim=-1)
    grid = torch.stack(
        (2.0 * u / (width - 1) - 1.0, 2.0 * v / (height - 1) - 1.0), dim=-1
    )[None, None]
    return functional.grid_sample(
        image[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(-1)


def _safety_margin_valid(camera, pixels: torch.Tensor, margin_px: float) -> torch.Tensor:
    offsets = pixels.new_tensor(
        (
            (margin_px, 0.0),
            (-margin_px, 0.0),
            (0.0, margin_px),
            (0.0, -margin_px),
            (margin_px, margin_px),
            (margin_px, -margin_px),
            (-margin_px, margin_px),
            (-margin_px, -margin_px),
        )
    )
    candidates = pixels[:, None, :] + offsets[None]
    return (camera.valid_mask(candidates) & camera.image_valid(candidates)).all(dim=1)


def _gt_rig_from_world(frame: StereoFisheyeFrame, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.linalg.inv(frame.gt_T_world_from_rig.to(dtype=dtype, device=device))


def build_oracle_motion_problem(
    source_frame: StereoFisheyeFrame,
    target_frame: StereoFisheyeFrame,
    reprojector: FisheyeRigReprojector,
    range_provider: RangeProvider,
    *,
    fixed_source_pose: Optional[torch.Tensor] = None,
    stride: int = 16,
    target_safety_margin_px: float = 128.0,
    occlusion_absolute_m: float = 0.01,
    occlusion_relative: float = 0.01,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[OracleMotionProblem, torch.Tensor]:
    """Build fixed same-camera Oracle correspondences and return GT separately."""
    if stride < 1:
        raise ValueError("stride must be positive")
    if target_safety_margin_px < 0.0 or not math.isfinite(target_safety_margin_px):
        raise ValueError("target_safety_margin_px must be finite and non-negative")
    if occlusion_absolute_m <= 0.0 or occlusion_relative <= 0.0:
        raise ValueError("occlusion tolerances must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be float32 or float64")
    device = torch.device(device)
    source_observation = range_provider.provide(source_frame)
    target_observation = range_provider.provide(target_frame)
    source_gt_pose = _gt_rig_from_world(source_frame, dtype, device)
    target_gt_pose = _gt_rig_from_world(target_frame, dtype, device)
    if fixed_source_pose is None:
        optimizer_source_pose = source_gt_pose
    else:
        optimizer_source_pose = fixed_source_pose.to(dtype=dtype, device=device)
        _require_transform(optimizer_source_pose, "fixed_source_pose")

    groups: list[OracleCameraCorrespondences] = []
    for camera_index in range(2):
        source_inverse_range_image = source_observation.inverse_range[camera_index].to(
            dtype=dtype, device=device
        )
        source_valid_image = source_observation.observation_valid[camera_index].to(device=device)
        source_confidence_image = source_observation.confidence[camera_index].to(
            dtype=dtype, device=device
        )
        target_range_image = target_observation.range_m[camera_index].to(
            dtype=dtype, device=device
        )
        target_valid_image = target_observation.observation_valid[camera_index].to(device=device)
        target_confidence_image = target_observation.confidence[camera_index].to(
            dtype=dtype, device=device
        )
        height, width = source_inverse_range_image.shape
        offset = stride // 2
        rows = torch.arange(offset, height, stride, dtype=dtype, device=device)
        columns = torch.arange(offset, width, stride, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(rows, columns, indexing="ij")
        pixels = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        integer_x = xx.reshape(-1).long()
        integer_y = yy.reshape(-1).long()
        source_inverse_range = source_inverse_range_image[integer_y, integer_x]
        source_valid = source_valid_image[integer_y, integer_x]
        source_confidence = source_confidence_image[integer_y, integer_x]

        oracle = reprojector.reproject(
            pixels,
            source_inverse_range,
            source_gt_pose,
            target_gt_pose,
            camera_index,
            camera_index,
        )
        target_range, target_range_valid = bilinear_sample_range(
            target_range_image, target_valid_image, oracle.pixels
        )
        target_confidence = _sample_scalar_image(target_confidence_image, oracle.pixels)
        occlusion_tolerance = torch.maximum(
            torch.full_like(target_range, occlusion_absolute_m),
            occlusion_relative * target_range,
        )
        occlusion_valid = torch.abs(oracle.target_range - target_range) <= occlusion_tolerance
        safety_valid = _safety_margin_valid(
            reprojector.cameras[camera_index], oracle.pixels, target_safety_margin_px
        )
        fixed_valid = (
            source_valid
            & oracle.validity.geometric_valid
            & target_range_valid
            & occlusion_valid
            & safety_valid
            & torch.isfinite(target_confidence)
            & (target_confidence > 0.0)
        )
        base_weights = source_confidence * target_confidence
        selected_weights = base_weights[fixed_valid]
        positive = torch.isfinite(selected_weights) & (selected_weights > 0.0)
        selected_indices = torch.nonzero(fixed_valid, as_tuple=False).flatten()[positive]
        group = OracleCameraCorrespondences(
            source_pixels=pixels[selected_indices].detach().clone(),
            source_inverse_range=source_inverse_range[selected_indices].detach().clone(),
            observed_target_pixels=oracle.pixels[selected_indices].detach().clone(),
            source_camera_index=camera_index,
            target_camera_index=camera_index,
            fixed_validity=torch.ones(
                len(selected_indices), dtype=torch.bool, device=device
            ),
            base_weights=selected_weights[positive].detach().clone(),
        )
        groups.append(group)

    problem = OracleMotionProblem(
        T_rig_from_world_source=optimizer_source_pose.detach().clone(),
        front=groups[0],
        back=groups[1],
    )
    return problem, target_gt_pose.detach().clone()


def _huber_cost_and_weight(errors: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    threshold_tensor = errors.new_tensor(threshold)
    quadratic = errors <= threshold_tensor
    cost = torch.where(
        quadratic,
        0.5 * errors * errors,
        threshold_tensor * (errors - 0.5 * threshold_tensor),
    )
    weight = torch.where(
        quadratic,
        torch.ones_like(errors),
        threshold_tensor / torch.clamp(errors, min=torch.finfo(errors.dtype).tiny),
    )
    return cost, weight


def _left_perturb(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return SE3.exp(delta).matrix() @ pose


class OracleMotionOnlyTracker:
    """Levenberg-Marquardt reference that optimizes only the target rig pose."""

    def __init__(
        self,
        reprojector: FisheyeRigReprojector,
        config: OracleMotionOnlyConfig = OracleMotionOnlyConfig(),
    ) -> None:
        if not isinstance(reprojector, FisheyeRigReprojector):
            raise TypeError("reprojector must be a FisheyeRigReprojector")
        if not isinstance(config, OracleMotionOnlyConfig):
            raise TypeError("config must be OracleMotionOnlyConfig")
        self.reprojector = reprojector
        self.config = config

    def _camera_scales(self, problem: OracleMotionProblem) -> tuple[float, float]:
        if self.config.camera_weighting == "all" or len(problem.groups) == 1:
            return (1.0 if problem.front is not None else 0.0, 1.0 if problem.back is not None else 0.0)
        front_sum = float(problem.front.base_weights.sum())
        back_sum = float(problem.back.base_weights.sum())
        total = front_sum + back_sum
        return total / (2.0 * front_sum), total / (2.0 * back_sum)

    def _evaluate(
        self,
        problem: OracleMotionProblem,
        target_pose: torch.Tensor,
        camera_scales: tuple[float, float],
        compute_linearization: bool,
    ) -> _Evaluation:
        dtype = target_pose.dtype
        device = target_pose.device
        total_cost = torch.zeros((), dtype=dtype, device=device)
        hessian = torch.zeros((6, 6), dtype=dtype, device=device) if compute_linearization else None
        gradient = torch.zeros(6, dtype=dtype, device=device) if compute_linearization else None
        per_camera_valid = [0, 0]
        invalid_count = 0
        invalid_cost, _ = _huber_cost_and_weight(
            target_pose.new_tensor([self.config.invalid_residual_penalty_px]),
            self.config.huber_threshold_px,
        )
        for group in problem.groups:
            camera_index = group.source_camera_index
            scale = target_pose.new_tensor(camera_scales[camera_index])
            result = self.reprojector.reproject(
                group.source_pixels,
                group.source_inverse_range,
                problem.T_rig_from_world_source,
                target_pose,
                group.source_camera_index,
                group.target_camera_index,
                compute_jacobians=compute_linearization,
            )
            valid = group.fixed_validity & result.validity.geometric_valid
            residual = group.observed_target_pixels - result.pixels
            errors = torch.linalg.vector_norm(residual, dim=-1)
            robust_cost, robust_weight = _huber_cost_and_weight(
                errors, self.config.huber_threshold_px
            )
            point_cost = torch.where(valid, robust_cost, invalid_cost[0])
            fixed_weight = scale * group.base_weights
            total_cost = total_cost + torch.sum(fixed_weight * point_cost)
            valid_count = int(valid.sum())
            per_camera_valid[camera_index] = valid_count
            invalid_count += group.count - valid_count
            if compute_linearization and valid_count:
                jacobian = result.jacobians.target_pose[valid]
                residual_valid = residual[valid]
                weights = fixed_weight[valid] * robust_weight[valid]
                hessian = hessian + torch.einsum("nki,nk,nkj->ij", jacobian, weights[:, None] * torch.ones_like(residual_valid), jacobian)
                gradient = gradient + torch.einsum(
                    "nki,nk->i", jacobian, weights[:, None] * residual_valid
                )
        return _Evaluation(
            cost=total_cost / problem.fixed_count,
            valid_count=sum(per_camera_valid),
            invalid_count=invalid_count,
            per_camera_valid=tuple(per_camera_valid),
            hessian=hessian,
            gradient=gradient,
        )

    def _enough_observations(self, problem: OracleMotionProblem, evaluation: _Evaluation) -> bool:
        if evaluation.valid_count < self.config.minimum_total_observations:
            return False
        for camera_index, group in enumerate((problem.front, problem.back)):
            if group is not None and evaluation.per_camera_valid[camera_index] < self.config.minimum_observations_per_camera:
                return False
        return True

    @staticmethod
    def _hessian_diagnostics(hessian: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, float]:
        symmetric = 0.5 * (hessian + hessian.transpose(0, 1))
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
        condition = math.inf if minimum <= epsilon else maximum / minimum
        return eigenvalues, condition

    def _result(
        self,
        pose: torch.Tensor,
        status: TrackerStatus,
        initial_cost: float,
        evaluation: _Evaluation,
        scales: tuple[float, float],
        eigenvalues: torch.Tensor,
        condition: float,
        damping: float,
        history: Sequence[OracleMotionIteration],
        iterations: int,
    ) -> OracleMotionOnlyResult:
        if evaluation.hessian is not None and bool(torch.isfinite(evaluation.hessian).all()):
            try:
                eigenvalues, condition = self._hessian_diagnostics(
                    evaluation.hessian, self.config.diagonal_epsilon
                )
            except RuntimeError:
                pass
        return OracleMotionOnlyResult(
            T_rig_from_world_target=pose.detach().clone(),
            status=status,
            iterations=iterations,
            initial_cost=initial_cost,
            final_cost=float(evaluation.cost),
            valid_count=evaluation.valid_count,
            camera_scales=scales,
            hessian_eigenvalues=tuple(float(value) for value in eigenvalues),
            hessian_condition_number=condition,
            final_damping=damping,
            history=tuple(history),
        )

    def optimize(
        self,
        problem: OracleMotionProblem,
        initial_target_pose: torch.Tensor,
    ) -> OracleMotionOnlyResult:
        if not isinstance(problem, OracleMotionProblem):
            raise TypeError("problem must be OracleMotionProblem")
        _require_transform(initial_target_pose, "initial_target_pose")
        if initial_target_pose.dtype != problem.T_rig_from_world_source.dtype:
            raise TypeError("initial target pose and problem must use the same dtype")
        if initial_target_pose.device != problem.T_rig_from_world_source.device:
            raise ValueError("initial target pose and problem must use the same device")

        pose = initial_target_pose.detach().clone()
        scales = self._camera_scales(problem)
        current = self._evaluate(problem, pose, scales, compute_linearization=True)
        initial_cost = float(current.cost)
        zero_eigenvalues = pose.new_zeros(6)
        if not self._enough_observations(problem, current):
            return self._result(
                pose, "insufficient_observations", initial_cost, current, scales,
                zero_eigenvalues, math.inf, self.config.initial_damping, (), 0
            )
        damping = self.config.initial_damping
        history: list[OracleMotionIteration] = []
        last_eigenvalues = zero_eigenvalues
        last_condition = math.inf

        for iteration in range(self.config.maximum_iterations):
            if current.hessian is None or current.gradient is None:
                raise RuntimeError("internal linearization is missing")
            tensors = (current.cost, current.hessian, current.gradient)
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                return self._result(
                    pose, "numerical_failure", initial_cost, current, scales,
                    last_eigenvalues, last_condition, damping, history, iteration
                )
            try:
                last_eigenvalues, last_condition = self._hessian_diagnostics(
                    current.hessian, self.config.diagonal_epsilon
                )
            except RuntimeError:
                return self._result(
                    pose, "numerical_failure", initial_cost, current, scales,
                    last_eigenvalues, last_condition, damping, history, iteration
                )
            if not bool(torch.isfinite(last_eigenvalues).all()):
                return self._result(
                    pose, "numerical_failure", initial_cost, current, scales,
                    last_eigenvalues, last_condition, damping, history, iteration
                )

            accepted = False
            smallest_step = None
            for retry in range(self.config.maximum_retries):
                diagonal = torch.clamp(
                    torch.diagonal(current.hessian), min=self.config.diagonal_epsilon
                )
                damped = current.hessian + damping * torch.diag(diagonal)
                try:
                    delta = torch.linalg.solve(damped, current.gradient)
                except RuntimeError:
                    delta = pose.new_full((6,), float("nan"))
                if bool(torch.isfinite(delta).all()):
                    translation = delta[:3]
                    rotation = delta[3:]
                    translation_norm = torch.linalg.vector_norm(translation)
                    rotation_norm = torch.linalg.vector_norm(rotation)
                    if float(translation_norm) > self.config.maximum_translation_step_m:
                        translation = translation * (
                            self.config.maximum_translation_step_m / translation_norm
                        )
                    if float(rotation_norm) > self.config.maximum_rotation_step_rad:
                        rotation = rotation * (
                            self.config.maximum_rotation_step_rad / rotation_norm
                        )
                    delta = torch.cat((translation, rotation))
                    translation_step = float(torch.linalg.vector_norm(translation))
                    rotation_step = float(torch.linalg.vector_norm(rotation))
                    candidate_pose = _left_perturb(pose, delta)
                    candidate = self._evaluate(
                        problem, candidate_pose, scales, compute_linearization=False
                    )
                    invalid_fraction = candidate.invalid_count / problem.fixed_count
                    finite_candidate = bool(torch.isfinite(candidate.cost)) and bool(
                        torch.isfinite(candidate_pose).all()
                    )
                    candidate_cost = float(candidate.cost) if finite_candidate else math.inf
                    candidate_accepted = (
                        finite_candidate
                        and invalid_fraction <= self.config.maximum_candidate_invalid_fraction
                        and candidate_cost <= float(current.cost)
                    )
                else:
                    translation_step = math.inf
                    rotation_step = math.inf
                    candidate = current
                    candidate_cost = math.inf
                    candidate_accepted = False
                history.append(
                    OracleMotionIteration(
                        iteration=iteration,
                        retry=retry,
                        damping=damping,
                        current_cost=float(current.cost),
                        candidate_cost=candidate_cost,
                        accepted=candidate_accepted,
                        valid_count=current.valid_count,
                        candidate_invalid_count=candidate.invalid_count,
                        translation_step_m=translation_step,
                        rotation_step_rad=rotation_step,
                        hessian_minimum_eigenvalue=float(last_eigenvalues[0]),
                        hessian_maximum_eigenvalue=float(last_eigenvalues[-1]),
                        hessian_condition_number=last_condition,
                    )
                )
                smallest_step = (translation_step, rotation_step)
                if candidate_accepted:
                    previous_cost = float(current.cost)
                    pose = candidate_pose
                    damping *= self.config.accepted_damping_scale
                    current = self._evaluate(problem, pose, scales, compute_linearization=True)
                    accepted = True
                    relative_decrease = (previous_cost - float(current.cost)) / max(
                        abs(previous_cost), self.config.diagonal_epsilon
                    )
                    if (
                        translation_step < self.config.translation_tolerance_m
                        and rotation_step < self.config.rotation_tolerance_rad
                    ) or relative_decrease < self.config.relative_cost_tolerance:
                        return self._result(
                            pose, "converged", initial_cost, current, scales,
                            last_eigenvalues, last_condition, damping, history, iteration + 1
                        )
                    break
                damping *= self.config.rejected_damping_scale

            if not accepted:
                if smallest_step is not None and (
                    smallest_step[0] < self.config.translation_tolerance_m
                    and smallest_step[1] < self.config.rotation_tolerance_rad
                ):
                    return self._result(
                        pose, "converged", initial_cost, current, scales,
                        last_eigenvalues, last_condition, damping, history, iteration + 1
                    )
                return self._result(
                    pose, "numerical_failure", initial_cost, current, scales,
                    last_eigenvalues, last_condition, damping, history, iteration + 1
                )

        return self._result(
            pose, "max_iterations", initial_cost, current, scales,
            last_eigenvalues, last_condition, damping, history,
            self.config.maximum_iterations,
        )
