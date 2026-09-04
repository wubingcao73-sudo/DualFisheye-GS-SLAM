#!/usr/bin/env python3
"""Validate native fisheye rig reprojection and its analytic Jacobians."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from lietorch import SE3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera  # noqa: E402
from hislam2.data import DEFAULT_DATA_ROOT, StereoFisheyeDataset  # noqa: E402
from hislam2.geom.fisheye_reprojection import (  # noqa: E402
    FisheyeRigReprojector,
    se3_point_action_jacobian,
)
from hislam2.range import GroundTruthRangeProvider  # noqa: E402


CAMERA_LABELS = ("front", "back")
PAIR_CONFIGS = (
    ("temporal_front", 0, 0, 0, 1),
    ("temporal_back", 1, 1, 0, 1),
    ("stereo_front_to_back", 0, 1, 0, 0),
    ("stereo_back_to_front", 1, 0, 0, 0),
)


def distribution(values: torch.Tensor | np.ndarray) -> dict:
    if isinstance(values, torch.Tensor):
        values = values.detach().double().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot summarize an empty measurement")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def jacobian_error(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    absolute = torch.linalg.vector_norm(actual - reference, dim=(-2, -1))
    reference_norm = torch.linalg.vector_norm(reference, dim=(-2, -1))
    relative = absolute / torch.maximum(
        reference_norm, torch.full_like(reference_norm, 1e-12)
    )
    return {"absolute": distribution(absolute), "relative": distribution(relative)}


def make_cameras(dataset: StereoFisheyeDataset) -> list[DoubleSphereCamera]:
    return [
        DoubleSphereCamera(
            *dataset.camera_params[index],
            *dataset.image_size[index],
        )
        for index in range(2)
    ]


def T_rig_from_world(dataset: StereoFisheyeDataset, index: int) -> torch.Tensor:
    return torch.linalg.inv(
        torch.from_numpy(dataset.gt_T_world_from_rig[index]).to(torch.float64)
    )


def left_perturb_lietorch(transform: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return SE3.exp(delta).matrix() @ transform


def se3_exp_matrix(twist: torch.Tensor) -> torch.Tensor:
    """Differentiable SE3 exponential with lietorch [translation, rotation] order."""
    translation = twist[:3]
    x, y, z = twist[3:].unbind()
    zero = torch.zeros_like(x)
    rotation_hat = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero)
    ).reshape(3, 3)
    upper = torch.cat((rotation_hat, translation[:, None]), dim=1)
    bottom = torch.zeros((1, 4), dtype=twist.dtype, device=twist.device)
    return torch.matrix_exp(torch.cat((upper, bottom), dim=0))


def sample_inputs(camera: DoubleSphereCamera, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260904)
    pixels = torch.rand((count, 2), generator=generator, dtype=torch.float64)
    pixels *= torch.tensor((camera.width, camera.height), dtype=torch.float64)
    special = torch.tensor(
        [
            [camera.cx, camera.cy],
            [camera.cx - 1.0, camera.cy],
            [camera.cx + 1.0, camera.cy],
            [camera.cx, 0.0],
            [camera.cx, camera.height - 1.0],
            [0.0, camera.cy],
            [camera.width - 1.0, camera.cy],
        ],
        dtype=torch.float64,
    )
    domain_radius = np.sqrt(1.0 / (2.0 * camera.alpha - 1.0))
    domain_angles = torch.tensor(
        (np.pi / 4.0, 3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0, 7.0 * np.pi / 4.0),
        dtype=torch.float64,
    )
    domain_pixels = []
    for margin in (1e-2, 1e-4, 1e-6):
        radius = domain_radius * (1.0 - margin)
        domain_pixels.append(
            torch.stack(
                (
                    camera.cx + camera.fx * radius * torch.cos(domain_angles),
                    camera.cy + camera.fy * radius * torch.sin(domain_angles),
                ),
                dim=-1,
            )
        )
    pixels = torch.cat((pixels, special, *domain_pixels), dim=0)
    range_generator = torch.Generator().manual_seed(37)
    ranges = torch.exp(
        torch.empty(len(pixels), dtype=torch.float64).uniform_(
            np.log(0.5), np.log(10.0), generator=range_generator
        )
    )
    return pixels, ranges.reciprocal()


def finite_difference_pose(
    reprojector: FisheyeRigReprojector,
    pixels: torch.Tensor,
    inverse_range: torch.Tensor,
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    source_camera: int,
    target_camera: int,
    which_pose: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    base = reprojector.reproject(
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
        True,
    )
    common_valid = base.validity.geometric_valid.clone()
    columns = []
    step = 1e-6
    for axis in range(6):
        delta = torch.zeros(6, dtype=torch.float64)
        delta[axis] = step
        if which_pose == "source":
            plus_source = left_perturb_lietorch(source_pose, delta)
            minus_source = left_perturb_lietorch(source_pose, -delta)
            plus_target = minus_target = target_pose
        else:
            plus_source = minus_source = source_pose
            plus_target = left_perturb_lietorch(target_pose, delta)
            minus_target = left_perturb_lietorch(target_pose, -delta)
        plus = reprojector.reproject(
            pixels,
            inverse_range,
            plus_source,
            plus_target,
            source_camera,
            target_camera,
        )
        minus = reprojector.reproject(
            pixels,
            inverse_range,
            minus_source,
            minus_target,
            source_camera,
            target_camera,
        )
        common_valid &= plus.validity.geometric_valid & minus.validity.geometric_valid
        columns.append((plus.pixels - minus.pixels) / (2.0 * step))
    numerical = torch.stack(columns, dim=-1)
    analytic = getattr(base.jacobians, f"{which_pose}_pose")
    crossing = base.validity.geometric_valid & ~common_valid
    return analytic[common_valid], numerical[common_valid], int(crossing.sum())


def finite_difference_inverse_range(
    reprojector: FisheyeRigReprojector,
    pixels: torch.Tensor,
    inverse_range: torch.Tensor,
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    source_camera: int,
    target_camera: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    base = reprojector.reproject(
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
        True,
    )
    step = 1e-6 * torch.maximum(inverse_range, torch.ones_like(inverse_range))
    plus = reprojector.reproject(
        pixels,
        inverse_range + step,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
    )
    minus = reprojector.reproject(
        pixels,
        inverse_range - step,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
    )
    common_valid = (
        base.validity.geometric_valid
        & plus.validity.geometric_valid
        & minus.validity.geometric_valid
    )
    numerical = ((plus.pixels - minus.pixels) / (2.0 * step[..., None]))[..., None]
    crossing = base.validity.geometric_valid & ~common_valid
    return (
        base.jacobians.inverse_range[common_valid],
        numerical[common_valid],
        int(crossing.sum()),
    )


def autograd_metrics(
    reprojector: FisheyeRigReprojector,
    pixels: torch.Tensor,
    inverse_range: torch.Tensor,
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    source_camera: int,
    target_camera: int,
) -> dict:
    base = reprojector.reproject(
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
        True,
    )
    valid_indices = torch.nonzero(base.validity.geometric_valid, as_tuple=False).flatten()[:64]
    if len(valid_indices) < 16:
        raise AssertionError("not enough valid samples for autograd")
    selected_pixels = pixels[valid_indices]
    selected_inverse_range = inverse_range[valid_indices]
    selected = reprojector.reproject(
        selected_pixels,
        selected_inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
        True,
    )

    def pose_function(delta: torch.Tensor, which_pose: str) -> torch.Tensor:
        if which_pose == "source":
            perturbed_source = se3_exp_matrix(delta) @ source_pose
            perturbed_target = target_pose
        else:
            perturbed_source = source_pose
            perturbed_target = se3_exp_matrix(delta) @ target_pose
        return reprojector.reproject(
            selected_pixels,
            selected_inverse_range,
            perturbed_source,
            perturbed_target,
            source_camera,
            target_camera,
        ).pixels

    delta = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    source_autograd = torch.autograd.functional.jacobian(
        lambda value: pose_function(value, "source"), delta, vectorize=True
    )
    target_autograd = torch.autograd.functional.jacobian(
        lambda value: pose_function(value, "target"), delta, vectorize=True
    )

    differentiable_inverse_range = selected_inverse_range.detach().requires_grad_(True)
    inverse_pixels = reprojector.reproject(
        selected_pixels,
        differentiable_inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
    ).pixels
    inverse_rows = []
    for coordinate in range(2):
        inverse_rows.append(
            torch.autograd.grad(
                inverse_pixels[:, coordinate].sum(),
                differentiable_inverse_range,
                retain_graph=coordinate == 0,
            )[0]
        )
    inverse_autograd = torch.stack(inverse_rows, dim=-1)[..., None]
    return {
        "sample_count": len(valid_indices),
        "source_pose": jacobian_error(selected.jacobians.source_pose, source_autograd),
        "target_pose": jacobian_error(selected.jacobians.target_pose, target_autograd),
        "inverse_range": jacobian_error(
            selected.jacobians.inverse_range, inverse_autograd
        ),
    }


def precision_metrics(
    cameras,
    extrinsics64,
    pixels64,
    inverse_range64,
    source_pose64,
    target_pose64,
    source_camera,
    target_camera,
) -> tuple[dict, dict]:
    reference_reprojector = FisheyeRigReprojector(cameras, extrinsics64)
    reference = reference_reprojector.reproject(
        pixels64,
        inverse_range64,
        source_pose64,
        target_pose64,
        source_camera,
        target_camera,
        True,
    )
    reprojector32 = FisheyeRigReprojector(cameras, extrinsics64.float())
    result32 = reprojector32.reproject(
        pixels64.float(),
        inverse_range64.float(),
        source_pose64.float(),
        target_pose64.float(),
        source_camera,
        target_camera,
        True,
    )
    common = reference.validity.geometric_valid & result32.validity.geometric_valid
    cpu = {
        "valid_mask_mismatch_count": int(
            (reference.validity.geometric_valid != result32.validity.geometric_valid).sum()
        ),
        "common_valid_count": int(common.sum()),
        "pixel_error_px": distribution(
            torch.linalg.vector_norm(
                result32.pixels[common].double() - reference.pixels[common], dim=-1
            )
        ),
    }
    for name in ("source_pose", "target_pose", "inverse_range"):
        cpu[f"{name}_jacobian"] = jacobian_error(
            getattr(result32.jacobians, name)[common].double(),
            getattr(reference.jacobians, name)[common],
        )
    if (
        cpu["valid_mask_mismatch_count"] != 0
        or cpu["pixel_error_px"]["p99"] >= 1e-3
        or any(
            cpu[f"{name}_jacobian"]["relative"]["p99"] >= 1e-4
            for name in ("source_pose", "target_pose", "inverse_range")
        )
    ):
        raise AssertionError(f"CPU float32 mismatch: {cpu}")

    if not torch.cuda.is_available():
        return cpu, {"status": "unavailable", "reason": "CUDA is not available"}
    gpu_reprojector = FisheyeRigReprojector(cameras, extrinsics64.float().cuda())
    gpu = gpu_reprojector.reproject(
        pixels64.float().cuda(),
        inverse_range64.float().cuda(),
        source_pose64.float().cuda(),
        target_pose64.float().cuda(),
        source_camera,
        target_camera,
        True,
    )
    gpu_valid = gpu.validity.geometric_valid.cpu()
    common_gpu = result32.validity.geometric_valid & gpu_valid
    cuda = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "valid_masks_equal": bool(
            torch.equal(result32.validity.geometric_valid, gpu_valid)
        ),
        "pixel_error_px": distribution(
            torch.linalg.vector_norm(
                result32.pixels[common_gpu] - gpu.pixels.cpu()[common_gpu], dim=-1
            )
        ),
    }
    for name in ("source_pose", "target_pose", "inverse_range"):
        cuda[f"{name}_jacobian"] = jacobian_error(
            getattr(gpu.jacobians, name).cpu()[common_gpu],
            getattr(result32.jacobians, name)[common_gpu],
        )
    if (
        not cuda["valid_masks_equal"]
        or cuda["pixel_error_px"]["p99"] >= 1e-3
        or any(
            cuda[f"{name}_jacobian"]["relative"]["p99"] >= 1e-4
            for name in ("source_pose", "target_pose", "inverse_range")
        )
    ):
        cuda["status"] = "failed"
        raise AssertionError(f"CUDA float32 mismatch: {cuda}")
    return cpu, cuda


def twist_order_metrics() -> dict:
    points = torch.tensor(
        [[0.4, -0.7, 1.3], [-1.0, 0.2, -0.3]], dtype=torch.float64
    )
    analytic = se3_point_action_jacobian(points)
    columns = []
    step = 1e-6
    for axis in range(6):
        delta = torch.zeros(6, dtype=torch.float64)
        delta[axis] = step
        plus = SE3.exp(delta.expand(len(points), 6)) * points
        minus = SE3.exp((-delta).expand(len(points), 6)) * points
        columns.append((plus - minus) / (2.0 * step))
    numerical = torch.stack(columns, dim=-1)
    maximum_error = float(torch.abs(analytic - numerical).max())
    translation_block_error = float(
        torch.abs(analytic[..., :3] - torch.eye(3).expand(2, 3, 3)).max()
    )
    result = {
        "order": ["tx", "ty", "tz", "rx", "ry", "rz"],
        "perturbation": "T_new = Exp(delta) @ T",
        "lietorch_point_action_max_absolute_error": maximum_error,
        "translation_first_block_max_error": translation_block_error,
    }
    if maximum_error >= 1e-9 or translation_block_error != 0.0:
        raise AssertionError(f"lietorch twist convention mismatch: {result}")
    return result


def validate_math_pair(
    cameras,
    extrinsics64,
    source_pose,
    target_pose,
    source_camera,
    target_camera,
    sample_count,
) -> dict:
    reprojector = FisheyeRigReprojector(cameras, extrinsics64)
    pixels, inverse_range = sample_inputs(cameras[source_camera], sample_count)
    base = reprojector.reproject(
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
        True,
    )
    source_analytic, source_numerical, source_crossing = finite_difference_pose(
        reprojector,
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
        "source",
    )
    target_analytic, target_numerical, target_crossing = finite_difference_pose(
        reprojector,
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
        "target",
    )
    range_analytic, range_numerical, range_crossing = finite_difference_inverse_range(
        reprojector,
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
    )
    result = {
        "sample_count": len(pixels),
        "geometric_valid_count": int(base.validity.geometric_valid.sum()),
        "valid_ratio": float(base.validity.geometric_valid.double().mean()),
        "source_pose_finite_difference": jacobian_error(
            source_analytic, source_numerical
        ),
        "target_pose_finite_difference": jacobian_error(
            target_analytic, target_numerical
        ),
        "inverse_range_finite_difference": jacobian_error(
            range_analytic, range_numerical
        ),
        "finite_difference_domain_crossing_count": {
            "source_pose": source_crossing,
            "target_pose": target_crossing,
            "inverse_range": range_crossing,
        },
        "autograd": autograd_metrics(
            reprojector,
            pixels,
            inverse_range,
            source_pose,
            target_pose,
            source_camera,
            target_camera,
        ),
    }
    source_rays, _ = cameras[source_camera].unproject(pixels)
    target_rays, _ = cameras[target_camera].unproject(base.pixels)
    base_valid = base.validity.geometric_valid
    result["coverage"] = {
        "requested_source_domain_edge_count": 12,
        "source_domain_edge_model_valid_count": int(
            base.validity.source_model_valid[-12:].sum()
        ),
        "source_domain_edge_image_valid_count": int(
            base.validity.source_image_valid[-12:].sum()
        ),
        "source_negative_ray_z_count": int(
            (base_valid & (source_rays[:, 2] < 0.0)).sum()
        ),
        "target_negative_ray_z_count": int(
            (base_valid & (target_rays[:, 2] < 0.0)).sum()
        ),
    }
    cpu, cuda = precision_metrics(
        cameras,
        extrinsics64,
        pixels,
        inverse_range,
        source_pose,
        target_pose,
        source_camera,
        target_camera,
    )
    result["cpu_float32_vs_float64"] = cpu
    result["cuda_float32_vs_cpu_float32"] = cuda
    for name in ("source_pose", "target_pose", "inverse_range"):
        if result[f"{name}_finite_difference"]["relative"]["p99"] >= 1e-3:
            raise AssertionError(f"{name} finite-difference p99 exceeds 1e-3")
        if result["autograd"][name]["relative"]["p99"] >= 1e-9:
            raise AssertionError(f"{name} autograd p99 exceeds 1e-9")
    return result


def identity_metrics(
    reprojector: FisheyeRigReprojector,
    camera_index: int,
    pose: torch.Tensor,
    sample_count: int,
) -> dict:
    pixels, inverse_range = sample_inputs(
        reprojector.cameras[camera_index], sample_count
    )
    result = reprojector.reproject(
        pixels,
        inverse_range,
        pose,
        pose,
        camera_index,
        camera_index,
    )
    valid = result.validity.geometric_valid
    pixel_error = torch.linalg.vector_norm(result.pixels[valid] - pixels[valid], dim=-1)
    expected_range = inverse_range[valid].reciprocal()
    range_error = torch.abs(result.target_range[valid] - expected_range)
    metrics = {
        "valid_count": int(valid.sum()),
        "pixel_error_px": distribution(pixel_error),
        "target_range_max_absolute_error_m": float(range_error.max()),
    }
    if metrics["pixel_error_px"]["p99"] >= 1e-3:
        raise AssertionError("identity reprojection pixel p99 exceeds 1e-3 px")
    return metrics


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"failed to write {path}")


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    write_image(path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def real_reprojection(
    reprojector: FisheyeRigReprojector,
    source_frame,
    target_frame,
    source_observation,
    target_observation,
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
    source_camera: int,
    target_camera: int,
    output_directory: Path,
    pair_kind: str,
) -> dict:
    target_model = reprojector.cameras[target_camera]
    height, width = target_model.height, target_model.width
    source_height = reprojector.cameras[source_camera].height
    source_width = reprojector.cameras[source_camera].width
    source_inverse_range = source_observation.inverse_range[source_camera].double()
    source_observation_valid = source_observation.observation_valid[source_camera]
    target_range = target_observation.range_m[target_camera].numpy()
    target_observation_valid = target_observation.observation_valid[target_camera].numpy()
    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    row_chunk = 128

    def projected_chunks():
        for row_start in range(0, source_height, row_chunk):
            row_end = min(row_start + row_chunk, source_height)
            local_observation_valid = source_observation_valid[row_start:row_end]
            if not local_observation_valid.any():
                continue
            y, x = torch.meshgrid(
                torch.arange(row_start, row_end, dtype=torch.float64),
                torch.arange(source_width, dtype=torch.float64),
                indexing="ij",
            )
            local_pixels = torch.stack((x, y), dim=-1)[local_observation_valid]
            local_inverse_range = source_inverse_range[row_start:row_end][
                local_observation_valid
            ]
            result = reprojector.reproject(
                local_pixels,
                local_inverse_range,
                source_pose,
                target_pose,
                source_camera,
                target_camera,
            )
            valid = result.validity.geometric_valid
            if not valid.any():
                continue
            pixels = result.pixels[valid]
            predicted_range = result.target_range[valid].float()
            u = torch.round(pixels[:, 0]).long().clamp(0, width - 1)
            v = torch.round(pixels[:, 1]).long().clamp(0, height - 1)
            target_flat = (v * width + u).numpy()
            source_grid = torch.nonzero(local_observation_valid, as_tuple=False)[valid]
            source_flat = (
                (source_grid[:, 0] + row_start) * source_width + source_grid[:, 1]
            ).numpy()
            yield target_flat, predicted_range.numpy(), source_flat

    for target_flat, predicted_range, _ in projected_chunks():
        np.minimum.at(z_buffer, target_flat, predicted_range)
    winner_source = np.full(height * width, -1, dtype=np.int64)
    for target_flat, predicted_range, source_flat in projected_chunks():
        winner = np.abs(predicted_range - z_buffer[target_flat]) <= 1e-5
        winner_source[target_flat[winner]] = source_flat[winner]

    projected = winner_source >= 0
    predicted_range = z_buffer[projected]
    observed_range = target_range.reshape(-1)[projected]
    observed_valid = target_observation_valid.reshape(-1)[projected]
    relative_error = np.full_like(predicted_range, np.inf)
    relative_error[observed_valid] = np.abs(
        predicted_range[observed_valid] - observed_range[observed_valid]
    ) / observed_range[observed_valid]
    visible = observed_valid & (
        np.abs(predicted_range - observed_range)
        <= np.maximum(0.01, 0.01 * observed_range)
    )

    source_rgb = source_frame.rgb[source_camera].permute(1, 2, 0).numpy()
    target_rgb = target_frame.rgb[target_camera].permute(1, 2, 0).numpy()
    warped_flat = np.zeros((height * width, 3), dtype=np.uint8)
    warped_flat[projected] = source_rgb.reshape(-1, 3)[winner_source[projected]]
    warped = warped_flat.reshape(height, width, 3)
    visible_flat = np.zeros(height * width, dtype=bool)
    visible_flat[np.flatnonzero(projected)[visible]] = True
    visible_mask = visible_flat.reshape(height, width)

    color_error = np.mean(
        np.abs(warped.astype(np.float32) - target_rgb.astype(np.float32)), axis=-1
    )
    photometric = np.zeros((height, width), dtype=np.float32)
    photometric[visible_mask] = color_error[visible_mask]
    photo_scale = max(float(np.percentile(photometric[visible_mask], 95)), 1.0)
    photo_heat = cv2.applyColorMap(
        np.clip(photometric / photo_scale * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    photo_heat[~visible_mask] = 0

    range_error_flat = np.zeros(height * width, dtype=np.float32)
    projected_indices = np.flatnonzero(projected)
    range_error_flat[projected_indices[observed_valid]] = relative_error[observed_valid]
    range_heat = cv2.applyColorMap(
        np.clip(range_error_flat.reshape(height, width) / 0.01 * 255.0, 0, 255).astype(
            np.uint8
        ),
        cv2.COLORMAP_TURBO,
    )
    range_heat.reshape(-1, 3)[~visible_flat] = 0
    overlay = target_rgb.copy()
    overlay[visible_mask] = (
        0.5 * target_rgb[visible_mask] + 0.5 * warped[visible_mask]
    ).astype(np.uint8)

    output_directory.mkdir(parents=True, exist_ok=True)
    save_rgb(output_directory / "source_rgb.png", source_rgb)
    save_rgb(output_directory / "target_rgb.png", target_rgb)
    save_rgb(output_directory / "warped_source.png", warped)
    write_image(output_directory / "valid_mask.png", visible_mask.astype(np.uint8) * 255)
    write_image(output_directory / "range_error.png", range_heat)
    write_image(output_directory / "photometric_error.png", photo_heat)
    save_rgb(output_directory / "overlay.png", overlay)

    range_stats = distribution(relative_error[observed_valid])
    photo_stats = distribution(photometric[visible_mask])
    result = {
        "source_camera": CAMERA_LABELS[source_camera],
        "target_camera": CAMERA_LABELS[target_camera],
        "projected_pixel_count": int(projected.sum()),
        "target_observation_count": int(observed_valid.sum()),
        "visible_pixel_count": int(visible.sum()),
        "target_range_relative_error": range_stats,
        "visible_photometric_error_8bit": photo_stats,
        "output_directory": str(output_directory),
    }
    if pair_kind == "temporal" and range_stats["p90"] >= 0.01:
        raise AssertionError("temporal target-range p90 exceeds 1%")
    if pair_kind == "stereo" and range_stats["p95"] >= 0.10:
        raise AssertionError("stereo target-range p95 exceeds 10%")
    return result


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("debug/fisheye_reprojection"))
    parser.add_argument("--samples", type=int, default=8_000)
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "data_root": str(args.data_root),
        "pose_state": "T_rig_from_world",
        "twist_order": ["tx", "ty", "tz", "rx", "ry", "rz"],
        "perturbation": "T_new = Exp(delta) @ T",
        "jacobian_semantics": "d predicted_target_pixel / d variable",
        "thresholds": {
            "identity_pixel_p99_px": 1e-3,
            "float64_finite_difference_p99_relative": 1e-3,
            "float64_autograd_p99_relative": 1e-9,
            "float32_jacobian_p99_relative": 1e-4,
            "float32_pixel_p99_px": 1e-3,
            "temporal_target_range_p90_relative": 0.01,
            "stereo_target_range_p95_relative": 0.10,
        },
    }
    try:
        if args.samples < 100:
            raise ValueError("--samples must be at least 100")
        dataset = StereoFisheyeDataset(args.data_root)
        cameras = make_cameras(dataset)
        extrinsics64 = torch.from_numpy(dataset.T_rig_from_camera).double()
        poses = [T_rig_from_world(dataset, index) for index in (0, 1)]
        report["twist_convention_validation"] = twist_order_metrics()
        identity_reprojector = FisheyeRigReprojector(cameras, extrinsics64)
        report["identity"] = {
            CAMERA_LABELS[index]: identity_metrics(
                identity_reprojector, index, poses[0], args.samples
            )
            for index in range(2)
        }
        report["mathematical"] = {}
        for name, source_camera, target_camera, source_index, target_index in PAIR_CONFIGS:
            report["mathematical"][name] = validate_math_pair(
                cameras,
                extrinsics64,
                poses[source_index],
                poses[target_index],
                source_camera,
                target_camera,
                args.samples,
            )
            write_report(report_path, report)

        if not args.skip_visualization:
            provider = GroundTruthRangeProvider()
            frames = [dataset[index] for index in (0, 1)]
            observations = [provider.provide(frame) for frame in frames]
            reprojector = FisheyeRigReprojector(cameras, extrinsics64)
            visual_pairs = (
                ("temporal/front_0000_to_0001", 0, 0, 0, 1, "temporal"),
                ("temporal/back_0000_to_0001", 1, 1, 0, 1, "temporal"),
                ("stereo/front_to_back_0000", 0, 1, 0, 0, "stereo"),
                ("stereo/back_to_front_0000", 1, 0, 0, 0, "stereo"),
            )
            report["real_reprojection"] = {}
            for path, source_camera, target_camera, source_index, target_index, kind in visual_pairs:
                report["real_reprojection"][path.replace("/", "_")] = real_reprojection(
                    reprojector,
                    frames[source_index],
                    frames[target_index],
                    observations[source_index],
                    observations[target_index],
                    poses[source_index],
                    poses[target_index],
                    source_camera,
                    target_camera,
                    args.output / path,
                    kind,
                )
                write_report(report_path, report)

        cuda_complete = all(
            result["cuda_float32_vs_cpu_float32"]["status"] == "passed"
            for result in report["mathematical"].values()
        )
        report["status"] = "passed" if cuda_complete else "cpu_passed_cuda_unavailable"
        write_report(report_path, report)
        print(f"Fisheye rig reprojection validation: {report['status']}")
        print(f"Report: {report_path}")
        return 0 if cuda_complete else 2
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        write_report(report_path, report)
        print(f"Fisheye rig reprojection validation failed: {error}", file=sys.stderr)
        print(f"Partial report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
