#!/usr/bin/env python3
"""Validate Double Sphere Camera Geometry V0 on the real classroom dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera  # noqa: E402
from hislam2.data import DEFAULT_DATA_ROOT, StereoFisheyeDataset  # noqa: E402
from hislam2.range import GroundTruthRangeProvider  # noqa: E402


CAMERA_LABELS = ("front", "back")
REPROJECTION_PAIRS = ((0, 1), (150, 151))


def distribution(values: torch.Tensor | np.ndarray) -> dict:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
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


def make_cameras(dataset: StereoFisheyeDataset) -> list[DoubleSphereCamera]:
    return [
        DoubleSphereCamera(
            *dataset.camera_params[index],
            *dataset.image_size[index],
        )
        for index in range(2)
    ]


def random_pixels(camera: DoubleSphereCamera, count: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260903)
    random = torch.stack(
        (
            torch.rand(count, generator=generator, dtype=dtype) * camera.width,
            torch.rand(count, generator=generator, dtype=dtype) * camera.height,
        ),
        dim=-1,
    )
    special = torch.tensor(
        [
            [camera.cx, camera.cy],
            [camera.cx - 1.0, camera.cy],
            [camera.cx + 1.0, camera.cy],
            [camera.cx, camera.cy - 1.0],
            [camera.cx, camera.cy + 1.0],
            [0.0, 0.0],
            [camera.width - 1.0, 0.0],
            [0.0, camera.height - 1.0],
            [camera.width - 1.0, camera.height - 1.0],
            [camera.cx, 0.0],
            [camera.cx, camera.height - 1.0],
            [0.0, camera.cy],
            [camera.width - 1.0, camera.cy],
        ],
        dtype=dtype,
    )
    return torch.cat((random, special), dim=0)


def mathematical_metrics(
    camera: DoubleSphereCamera,
    camera_name: str,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    pixels64 = random_pixels(camera, 250_000, torch.float64)
    rays64, unproject_valid64 = camera.unproject(pixels64)
    reconstructed_pixels64, project_valid64 = camera.project(rays64)
    roundtrip_valid64 = unproject_valid64 & project_valid64
    pixel_error64 = torch.linalg.vector_norm(
        reconstructed_pixels64[roundtrip_valid64] - pixels64[roundtrip_valid64], dim=-1
    )

    generator = torch.Generator().manual_seed(31)
    ranges64 = 0.25 + 19.75 * torch.rand(
        len(pixels64), generator=generator, dtype=torch.float64
    )
    points64 = rays64 * ranges64[:, None]
    point_pixels64, point_project_valid64 = camera.project(points64)
    rebuilt_rays64, point_unproject_valid64 = camera.unproject(point_pixels64)
    point_valid64 = unproject_valid64 & point_project_valid64 & point_unproject_valid64
    rebuilt_points64 = rebuilt_rays64 * torch.linalg.vector_norm(
        points64, dim=-1, keepdim=True
    )
    point_relative_error64 = torch.linalg.vector_norm(
        rebuilt_points64[point_valid64] - points64[point_valid64], dim=-1
    ) / torch.linalg.vector_norm(points64[point_valid64], dim=-1)
    ray_norm_error64 = torch.abs(
        torch.linalg.vector_norm(rays64[unproject_valid64], dim=-1) - 1.0
    )

    pixels32 = pixels64.float()
    rays32, unproject_valid32 = camera.unproject(pixels32)
    reconstructed_pixels32, project_valid32 = camera.project(rays32)
    common_valid = unproject_valid64 & unproject_valid32
    float_ray_difference = torch.abs(rays64.float()[common_valid] - rays32[common_valid])
    pixel_error32 = torch.linalg.vector_norm(
        reconstructed_pixels32[unproject_valid32 & project_valid32]
        - pixels32[unproject_valid32 & project_valid32],
        dim=-1,
    )

    lut64, lut_valid64 = camera.get_ray_lut(dtype=torch.float64)
    valid_lut_rays = lut64[lut_valid64]
    lut_norm_error64 = torch.abs(torch.linalg.vector_norm(valid_lut_rays, dim=-1) - 1.0)
    negative_ray_z_count = int((valid_lut_rays[:, 2] < 0.0).sum())
    closest_to_zero_z = float(torch.min(torch.abs(valid_lut_rays[:, 2])))

    if not torch.isfinite(valid_lut_rays).all():
        raise AssertionError(f"{camera_name}: valid float64 LUT contains NaN or Inf")

    metrics = {
        "width": camera.width,
        "height": camera.height,
        "xi": camera.xi,
        "alpha": camera.alpha,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "valid_pixel_count": int(lut_valid64.sum()),
        "valid_ratio": float(lut_valid64.double().mean()),
        "negative_ray_z_count": negative_ray_z_count,
        "negative_ray_z_ratio": negative_ray_z_count / int(lut_valid64.sum()),
        "closest_valid_ray_z_to_zero": closest_to_zero_z,
        "pixel_roundtrip_float64_px": distribution(pixel_error64),
        "point_reconstruction_float64_relative": distribution(point_relative_error64),
        "ray_norm_float64_max_error": float(ray_norm_error64.max()),
        "lut_ray_norm_float64_max_error": float(lut_norm_error64.max()),
        "pixel_roundtrip_float32_px": distribution(pixel_error32),
        "float32_vs_float64_ray_max_abs": float(float_ray_difference.max()),
        "lut_memory_bytes": {
            "float64_rays_and_mask": int(
                lut64.nelement() * lut64.element_size()
                + lut_valid64.nelement() * lut_valid64.element_size()
            ),
            "float32_rays_and_mask": camera.width
            * camera.height
            * (3 * torch.tensor([], dtype=torch.float32).element_size() + 1),
        },
        "special_regions": {
            "principal_point_valid": bool(camera.valid_mask(pixels64[-13:-12])[0]),
            "has_model_invalid_image_corner": bool(
                (~camera.valid_mask(pixels64[-8:-4])).any()
            ),
            "has_ray_z_near_zero": closest_to_zero_z < 1e-3,
            "has_negative_ray_z": negative_ray_z_count > 0,
        },
    }
    pixel_stats = metrics["pixel_roundtrip_float64_px"]
    point_stats = metrics["point_reconstruction_float64_relative"]
    metrics.update(
        {
            "pixel_roundtrip_mean_px": pixel_stats["mean"],
            "pixel_roundtrip_median_px": pixel_stats["median"],
            "pixel_roundtrip_p95_px": pixel_stats["p95"],
            "pixel_roundtrip_p99_px": pixel_stats["p99"],
            "pixel_roundtrip_max_px": pixel_stats["max"],
            "ray_norm_max_error": metrics["ray_norm_float64_max_error"],
            "point_reconstruction_mean_relative_error": point_stats["mean"],
            "point_reconstruction_p99_relative_error": point_stats["p99"],
        }
    )

    if metrics["pixel_roundtrip_float64_px"]["p99"] >= 1e-3:
        raise AssertionError(f"{camera_name}: pixel round-trip p99 exceeds 1e-3 px")
    if metrics["point_reconstruction_float64_relative"]["p99"] >= 1e-5:
        raise AssertionError(f"{camera_name}: point reconstruction p99 exceeds 1e-5")
    if metrics["ray_norm_float64_max_error"] >= 1e-6:
        raise AssertionError(f"{camera_name}: ray norm error exceeds 1e-6")
    if not all(metrics["special_regions"].values()):
        raise AssertionError(f"{camera_name}: one or more special regions were not covered")

    return metrics, lut64, lut_valid64


def cuda_metrics(
    camera: DoubleSphereCamera,
    cpu_lut32: torch.Tensor,
    cpu_valid32: torch.Tensor,
) -> dict:
    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "torch.cuda.is_available() is false"}

    device = torch.device("cuda:0")
    gpu_lut, gpu_valid = camera.get_ray_lut(device=device, dtype=torch.float32)
    cpu_lut_on_gpu = cpu_lut32.to(device)
    cpu_valid_on_gpu = cpu_valid32.to(device)
    valid_equal = bool(torch.equal(gpu_valid, cpu_valid_on_gpu))
    common_valid = gpu_valid & cpu_valid_on_gpu
    ray_error = torch.max(
        torch.abs(gpu_lut[common_valid] - cpu_lut_on_gpu[common_valid]), dim=-1
    )
    ray_error_stats = distribution(ray_error.values)
    valid_rays = gpu_lut[gpu_valid]
    finite = bool(torch.isfinite(valid_rays).all().cpu())
    norm_max_error = float(
        torch.max(torch.abs(torch.linalg.vector_norm(valid_rays, dim=-1) - 1.0)).cpu()
    )
    y, x = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=torch.float32),
        torch.arange(camera.width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    reference_pixels = torch.stack((x, y), dim=-1)
    reconstructed_pixels, projection_valid = camera.project(gpu_lut)
    roundtrip_valid = gpu_valid & projection_valid
    gpu_pixel_error = torch.linalg.vector_norm(
        reconstructed_pixels[roundtrip_valid] - reference_pixels[roundtrip_valid], dim=-1
    )
    gpu_pixel_stats = distribution(gpu_pixel_error)
    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "valid_equal_to_cpu": valid_equal,
        "ray_abs_difference_from_cpu": ray_error_stats,
        "pixel_roundtrip_px": gpu_pixel_stats,
        "ray_norm_max_error": norm_max_error,
        "valid_rays_finite": finite,
    }
    if (
        not valid_equal
        or not finite
        or ray_error_stats["p99"] >= 1e-5
        or ray_error_stats["max"] >= 5e-5
        or gpu_pixel_stats["p99"] >= 1e-3
        or norm_max_error >= 1e-5
    ):
        result["status"] = "failed"
        raise AssertionError(f"CUDA LUT does not match CPU float32: {result}")
    return result


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def write_binary_ply(path: Path, point_sets: Iterable[tuple[np.ndarray, np.ndarray]]) -> int:
    point_sets = list(point_sets)
    count = sum(len(points) for points, _ in point_sets)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    with path.open("wb") as handle:
        handle.write(header)
        for points, colors in point_sets:
            for start in range(0, len(points), 1_000_000):
                end = min(start + 1_000_000, len(points))
                vertices = np.empty(end - start, dtype=vertex_dtype)
                vertices["x"] = points[start:end, 0]
                vertices["y"] = points[start:end, 1]
                vertices["z"] = points[start:end, 2]
                vertices["red"] = colors[start:end, 0]
                vertices["green"] = colors[start:end, 1]
                vertices["blue"] = colors[start:end, 2]
                handle.write(vertices.tobytes())
    return path.stat().st_size


def save_pointcloud_preview(
    path: Path,
    point_sets: Sequence[tuple[np.ndarray, np.ndarray, str]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    planes = ((0, 1, "world X", "world Y"), (0, 2, "world X", "world Z"), (1, 2, "world Y", "world Z"))
    for points, colors, label in point_sets:
        stride = max(1, len(points) // 80_000)
        sample = points[::stride]
        sample_colors = colors[::stride].astype(np.float32) / 255.0
        for axis, (first, second, xlabel, ylabel) in zip(axes, planes):
            axis.scatter(
                sample[:, first],
                sample[:, second],
                s=0.25,
                c=sample_colors,
                alpha=0.65,
                label=label,
            )
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.set_aspect("equal", adjustable="box")
    axes[0].legend(markerscale=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def create_pointclouds(
    dataset: StereoFisheyeDataset,
    cameras: Sequence[DoubleSphereCamera],
    range_provider: GroundTruthRangeProvider,
    frame_index: int,
    output_dir: Path,
) -> dict:
    frame = dataset[frame_index]
    range_observation = range_provider.provide(frame)
    world_sets = []
    result = {}
    for camera_index, label in enumerate(CAMERA_LABELS):
        camera = cameras[camera_index]
        rays, model_valid = camera.get_ray_lut(dtype=torch.float32)
        observation_valid = range_observation.observation_valid[camera_index]
        pixels_in_image = torch.ones_like(model_valid)
        valid = model_valid & pixels_in_image & observation_valid
        points_camera = (
            frame.gt_range[camera_index][..., None] * rays
        )[valid].numpy()
        colors = frame.rgb[camera_index].permute(1, 2, 0)[valid].numpy()
        points_world = transform_points(
            points_camera, dataset.gt_T_world_from_camera[frame_index, camera_index]
        ).astype(np.float32)

        camera_path = output_dir / f"{label}_frame_{frame.index:04d}.ply"
        camera_size = write_binary_ply(camera_path, [(points_camera, colors)])
        world_sets.append((points_world, colors, label))
        result[label] = {
            "point_count": len(points_camera),
            "camera_ply": str(camera_path),
            "camera_ply_bytes": camera_size,
            "camera_bounds_min": points_camera.min(axis=0).tolist(),
            "camera_bounds_max": points_camera.max(axis=0).tolist(),
            "world_bounds_min": points_world.min(axis=0).tolist(),
            "world_bounds_max": points_world.max(axis=0).tolist(),
        }

    dual_path = output_dir / f"dual_frame_{frame.index:04d}_world.ply"
    dual_size = write_binary_ply(
        dual_path, [(points, colors) for points, colors, _ in world_sets]
    )
    preview_path = output_dir / f"dual_frame_{frame.index:04d}_world_preview.png"
    save_pointcloud_preview(preview_path, world_sets)
    result["dual_world"] = {
        "point_count": sum(len(points) for points, _, _ in world_sets),
        "ply": str(dual_path),
        "ply_bytes": dual_size,
        "preview": str(preview_path),
    }
    return result


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {path}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise IOError(f"failed to write {path}")


def forward_reprojection(
    source_frame,
    target_frame,
    range_provider: GroundTruthRangeProvider,
    camera: DoubleSphereCamera,
    camera_index: int,
    T_target_from_source: np.ndarray,
    output_dir: Path,
) -> dict:
    height, width = camera.height, camera.width
    rays, model_valid = camera.get_ray_lut(dtype=torch.float32)
    source_range_observation = range_provider.provide(source_frame)
    target_range_observation = range_provider.provide(target_frame)
    source_range = source_range_observation.range_m[camera_index]
    source_observation = source_range_observation.observation_valid[camera_index]
    source_valid = model_valid & source_observation
    target_range = target_range_observation.range_m[camera_index].numpy()
    target_observation = target_range_observation.observation_valid[camera_index].numpy()

    z_buffer = np.full(height * width, np.inf, dtype=np.float32)
    row_chunk = 128

    def projected_chunks():
        for row_start in range(0, height, row_chunk):
            row_end = min(row_start + row_chunk, height)
            local_valid = source_valid[row_start:row_end]
            if not local_valid.any():
                continue
            local_points = (
                source_range[row_start:row_end, ..., None]
                * rays[row_start:row_end]
            )[local_valid]
            rotation = torch.from_numpy(T_target_from_source[:3, :3]).to(torch.float32)
            translation = torch.from_numpy(T_target_from_source[:3, 3]).to(torch.float32)
            target_points = local_points @ rotation.T + translation
            pixels, projection_model_valid = camera.project(target_points)
            projection_image_valid = camera.image_valid(pixels)
            projected_valid = projection_model_valid & projection_image_valid
            if not projected_valid.any():
                continue
            pixels = pixels[projected_valid]
            target_points = target_points[projected_valid]
            u = torch.round(pixels[:, 0]).long().clamp(0, width - 1)
            v = torch.round(pixels[:, 1]).long().clamp(0, height - 1)
            target_flat = (v * width + u).numpy()
            predicted_range = torch.linalg.vector_norm(target_points, dim=-1).numpy()
            source_grid = torch.nonzero(local_valid, as_tuple=False)
            source_grid = source_grid[projected_valid]
            source_flat = (
                (source_grid[:, 0] + row_start) * width + source_grid[:, 1]
            ).numpy()
            yield target_flat, predicted_range, source_flat

    for target_flat, predicted_range, _ in projected_chunks():
        np.minimum.at(z_buffer, target_flat, predicted_range)

    winner_source = np.full(height * width, -1, dtype=np.int64)
    for target_flat, predicted_range, source_flat in projected_chunks():
        winner = np.abs(predicted_range - z_buffer[target_flat]) <= 1e-5
        winner_source[target_flat[winner]] = source_flat[winner]

    projected = winner_source >= 0
    target_flat_indices = np.flatnonzero(projected)
    predicted_range = z_buffer[projected]
    observed_range = target_range.reshape(-1)[projected]
    observed_valid = target_observation.reshape(-1)[projected]
    relative_error = np.full_like(predicted_range, np.inf, dtype=np.float32)
    relative_error[observed_valid] = np.abs(
        predicted_range[observed_valid] - observed_range[observed_valid]
    ) / observed_range[observed_valid]
    visible = observed_valid & (
        np.abs(predicted_range - observed_range)
        <= np.maximum(0.01, 0.01 * observed_range)
    )

    source_rgb = source_frame.rgb[camera_index].permute(1, 2, 0).numpy()
    target_rgb = target_frame.rgb[camera_index].permute(1, 2, 0).numpy()
    warped_flat = np.zeros((height * width, 3), dtype=np.uint8)
    warped_flat[projected] = source_rgb.reshape(-1, 3)[winner_source[projected]]
    warped = warped_flat.reshape(height, width, 3)
    visible_mask_flat = np.zeros(height * width, dtype=bool)
    visible_mask_flat[target_flat_indices[visible]] = True
    visible_mask = visible_mask_flat.reshape(height, width)

    photometric = np.zeros((height, width), dtype=np.float32)
    color_error = np.mean(
        np.abs(warped.astype(np.float32) - target_rgb.astype(np.float32)), axis=-1
    )
    photometric[visible_mask] = color_error[visible_mask]
    scale = max(float(np.percentile(photometric[visible_mask], 95)), 1.0)
    heat = cv2.applyColorMap(
        np.clip(photometric / scale * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heat[~visible_mask] = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    label = CAMERA_LABELS[camera_index]
    save_rgb(output_dir / f"{label}_t.png", source_rgb)
    save_rgb(output_dir / f"{label}_t1.png", target_rgb)
    save_rgb(output_dir / "warped_t_to_t1.png", warped)
    save_mask(output_dir / "valid_mask.png", visible_mask)
    if not cv2.imwrite(str(output_dir / "photometric_error.png"), heat):
        raise IOError("failed to write photometric error image")

    geometric_values = relative_error[observed_valid]
    metrics = {
        "projected_pixel_count": int(projected.sum()),
        "target_observation_count": int(observed_valid.sum()),
        "visible_pixel_count": int(visible.sum()),
        "target_range_relative_error": distribution(geometric_values),
        "visible_photometric_error_8bit": distribution(photometric[visible_mask]),
        "output_directory": str(output_dir),
    }
    if metrics["target_range_relative_error"]["p90"] >= 0.01:
        raise AssertionError("target-range reprojection p90 exceeds 1%")
    return metrics


def reprojection_suite(
    dataset: StereoFisheyeDataset,
    cameras: Sequence[DoubleSphereCamera],
    range_provider: GroundTruthRangeProvider,
    output_dir: Path,
) -> dict:
    results = {}
    for source_index, target_index in REPROJECTION_PAIRS:
        source_frame = dataset[source_index]
        target_frame = dataset[target_index]
        for camera_index, label in enumerate(CAMERA_LABELS):
            T_world_from_source = dataset.gt_T_world_from_camera[source_index, camera_index]
            T_world_from_target = dataset.gt_T_world_from_camera[target_index, camera_index]
            T_target_from_source = np.linalg.inv(T_world_from_target) @ T_world_from_source
            name = f"{label}_{source_index:04d}_to_{target_index:04d}"
            results[name] = forward_reprojection(
                source_frame,
                target_frame,
                range_provider,
                cameras[camera_index],
                camera_index,
                T_target_from_source,
                output_dir / name,
            )
        del source_frame, target_frame
    return results


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("debug/camera_model"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--skip-pointclouds", action="store_true")
    parser.add_argument("--skip-reprojection", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "data_root": str(args.data_root),
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "image_coordinate_convention": "u-right, v-down, integer pixel centers",
        "range_semantics": "Euclidean distance from camera center in meters",
        "transform_convention": "T_destination_from_source",
        "blender_camera_from_ds_camera": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
    }
    try:
        dataset = StereoFisheyeDataset(args.data_root)
        cameras = make_cameras(dataset)
        range_provider = GroundTruthRangeProvider()
        report["pose_composition_max_error"] = dataset.pose_composition_max_error
        for camera, name in zip(cameras, ("cam0", "cam1")):
            metrics, lut64, valid64 = mathematical_metrics(camera, name)
            cpu_lut32, cpu_valid32 = camera.get_ray_lut(dtype=torch.float32)
            metrics["cuda_float32"] = cuda_metrics(camera, cpu_lut32, cpu_valid32)
            report[name] = metrics
            write_report(report_path, report)
            del lut64, valid64, cpu_lut32, cpu_valid32
            camera.clear_ray_lut_cache()

        if not args.skip_pointclouds:
            report["pointclouds"] = create_pointclouds(
                dataset, cameras, range_provider, args.frame_index, args.output
            )
            write_report(report_path, report)
        if not args.skip_reprojection:
            report["reprojection"] = reprojection_suite(
                dataset, cameras, range_provider, args.output / "reprojection"
            )
            write_report(report_path, report)

        cuda_complete = all(
            report[name]["cuda_float32"]["status"] == "passed"
            for name in ("cam0", "cam1")
        )
        report["status"] = "passed" if cuda_complete else "cpu_passed_cuda_unavailable"
        write_report(report_path, report)
        print(f"Camera Geometry V0 validation: {report['status']}")
        print(f"Report: {report_path}")
        return 0 if cuda_complete else 2
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        write_report(report_path, report)
        print(f"Camera Geometry V0 validation failed: {error}", file=sys.stderr)
        print(f"Partial report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
