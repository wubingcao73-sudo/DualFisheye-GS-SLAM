#!/usr/bin/env python3
"""Validate the analytic Double Sphere projection Jacobian for both cameras."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera  # noqa: E402
from hislam2.data import DEFAULT_DATA_ROOT, StereoFisheyeDataset  # noqa: E402


def distribution(values: torch.Tensor) -> dict:
    values = values.detach().double().cpu().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty measurement")
    return {
        "mean": float(values.mean()),
        "median": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    difference = torch.linalg.vector_norm(actual - reference, dim=(-2, -1))
    reference_norm = torch.linalg.vector_norm(reference, dim=(-2, -1))
    relative = difference / torch.maximum(
        reference_norm, torch.full_like(reference_norm, 1e-12)
    )
    return {"absolute": distribution(difference), "relative": distribution(relative)}


def make_camera(dataset: StereoFisheyeDataset, index: int) -> DoubleSphereCamera:
    return DoubleSphereCamera(
        *dataset.camera_params[index],
        *dataset.image_size[index],
    )


def sample_points(camera: DoubleSphereCamera, count: int) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(20260903)
    random_pixels = torch.rand((count, 2), generator=generator, dtype=torch.float64)
    random_pixels *= torch.tensor((camera.width, camera.height), dtype=torch.float64)
    pixel_rays, pixel_valid = camera.unproject(random_pixels)
    rays = [pixel_rays[pixel_valid]]

    # Explicitly cover the principal point, image axes, ray_z ~= 0 and the
    # mathematical projection boundary. Boundary points are useful even when
    # finite-difference perturbations cross out of the valid domain.
    special_pixels = torch.tensor(
        [
            [camera.cx, camera.cy],
            [camera.cx - 1.0, camera.cy],
            [camera.cx + 1.0, camera.cy],
            [camera.cx, 0.0],
            [camera.cx, camera.height - 1.0],
            [0.0, camera.cy],
            [camera.width - 1.0, camera.cy],
            [0.0, 0.0],
            [camera.width - 1.0, 0.0],
            [0.0, camera.height - 1.0],
            [camera.width - 1.0, camera.height - 1.0],
        ],
        dtype=torch.float64,
    )
    special_rays, special_valid = camera.unproject(special_pixels)
    rays.append(special_rays[special_valid])

    angles = torch.linspace(0.0, 2.0 * torch.pi, 17, dtype=torch.float64)[:-1]
    for z in (-1e-8, 1e-8):
        radial = np.sqrt(1.0 - z * z)
        rays.append(
            torch.stack(
                (radial * torch.cos(angles), radial * torch.sin(angles), torch.full_like(angles, z)),
                dim=-1,
            )
        )
    for domain_margin in (1e-2, 1e-4, 1e-6):
        z = -camera._w2 + domain_margin
        radial = np.sqrt(1.0 - z * z)
        rays.append(
            torch.stack(
                (radial * torch.cos(angles), radial * torch.sin(angles), torch.full_like(angles, z)),
                dim=-1,
            )
        )

    unit_rays = torch.cat(rays, dim=0)
    range_generator = torch.Generator().manual_seed(31)
    ranges = torch.exp(
        torch.empty(len(unit_rays), dtype=torch.float64).uniform_(
            np.log(0.25), np.log(20.0), generator=range_generator
        )
    )
    points = unit_rays * ranges[:, None]
    _, valid = camera.project(points)
    points = points[valid]
    direction_z = points[:, 2] / torch.linalg.vector_norm(points, dim=-1)
    coverage = {
        "requested_random_pixels": count,
        "valid_random_pixels": int(pixel_valid.sum()),
        "valid_special_pixels": int(special_valid.sum()),
        "valid_point_count": len(points),
        "negative_z_count": int((direction_z < 0.0).sum()),
        "near_zero_z_count": int((torch.abs(direction_z) < 1e-6).sum()),
        "near_projection_domain_boundary_count": int(
            ((direction_z + camera._w2) < 1e-3).sum()
        ),
        "model_invalid_image_corner_count": int(
            (~camera.valid_mask(special_pixels[-4:])).sum()
        ),
    }
    return points, coverage


def autograd_jacobian(
    camera: DoubleSphereCamera, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = points.detach().requires_grad_(True)
    pixels, valid = camera.project(points)
    rows = []
    for output_index in range(2):
        rows.append(
            torch.autograd.grad(
                pixels[:, output_index].sum(),
                points,
                retain_graph=output_index == 0,
            )[0]
        )
    return pixels.detach(), valid.detach(), torch.stack(rows, dim=-2).detach()


def finite_difference_jacobian(
    camera: DoubleSphereCamera, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    step = 1e-6 * torch.maximum(
        torch.ones(len(points), dtype=points.dtype, device=points.device),
        torch.linalg.vector_norm(points, dim=-1),
    )
    columns = []
    perturbations_valid = torch.ones(len(points), dtype=torch.bool, device=points.device)
    for axis in range(3):
        delta = torch.zeros_like(points)
        delta[:, axis] = step
        plus, plus_valid = camera.project(points + delta)
        minus, minus_valid = camera.project(points - delta)
        perturbations_valid &= plus_valid & minus_valid
        columns.append((plus - minus) / (2.0 * step[:, None]))
    return torch.stack(columns, dim=-1), perturbations_valid


def cuda_metrics(camera: DoubleSphereCamera, points32: torch.Tensor) -> dict:
    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "torch.cuda.is_available() is false"}
    cpu_pixels, cpu_valid, cpu_jacobian = camera.project_jacobian(points32)
    gpu_pixels, gpu_valid, gpu_jacobian = camera.project_jacobian(points32.cuda())
    gpu_pixels = gpu_pixels.cpu()
    gpu_valid = gpu_valid.cpu()
    gpu_jacobian = gpu_jacobian.cpu()
    common_valid = cpu_valid & gpu_valid
    metrics = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "valid_masks_equal": bool(torch.equal(cpu_valid, gpu_valid)),
        "pixel_max_absolute_error": float(
            torch.abs(cpu_pixels[common_valid] - gpu_pixels[common_valid]).max()
        ),
        "jacobian_error": error_metrics(
            gpu_jacobian[common_valid], cpu_jacobian[common_valid]
        ),
    }
    if (
        not metrics["valid_masks_equal"]
        or metrics["pixel_max_absolute_error"] >= 1e-3
        or metrics["jacobian_error"]["relative"]["p99"] >= 1e-5
    ):
        metrics["status"] = "failed"
        raise AssertionError(f"CUDA float32 Jacobian mismatch: {metrics}")
    return metrics


def validate_camera(camera: DoubleSphereCamera, sample_count: int) -> dict:
    points64, coverage = sample_points(camera, sample_count)
    pixels64, analytic_valid64, analytic64 = camera.project_jacobian(points64)
    autograd_pixels64, autograd_valid64, autograd64 = autograd_jacobian(camera, points64)
    finite_difference64, perturbations_valid = finite_difference_jacobian(camera, points64)
    common64 = analytic_valid64 & autograd_valid64

    points32 = points64.float()
    pixels32, valid32, analytic32 = camera.project_jacobian(points32)
    common_precision = analytic_valid64 & valid32
    precision_metrics = {
        "valid_mask_mismatch_count": int((analytic_valid64 != valid32).sum()),
        "common_valid_count": int(common_precision.sum()),
        "pixel_absolute_error": distribution(
            torch.linalg.vector_norm(
                pixels32[common_precision].double() - pixels64[common_precision], dim=-1
            )
        ),
        "jacobian_error": error_metrics(
            analytic32[common_precision].double(), analytic64[common_precision]
        ),
    }

    result = {
        "width": camera.width,
        "height": camera.height,
        "xi": camera.xi,
        "alpha": camera.alpha,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "coverage": coverage,
        "project_and_project_jacobian_pixels_equal": bool(
            torch.equal(pixels64, camera.project(points64)[0])
        ),
        "float64_autograd": error_metrics(analytic64[common64], autograd64[common64]),
        "float64_autograd_pixels_equal": bool(torch.equal(pixels64, autograd_pixels64)),
        "float64_finite_difference": error_metrics(
            analytic64[perturbations_valid], finite_difference64[perturbations_valid]
        ),
        "finite_difference_usable_count": int(perturbations_valid.sum()),
        "finite_difference_domain_crossing_count": int((~perturbations_valid).sum()),
        "cpu_float32_vs_float64": precision_metrics,
        "cuda_float32_vs_cpu_float32": cuda_metrics(camera, points32),
    }
    if not result["project_and_project_jacobian_pixels_equal"]:
        raise AssertionError("project() and project_jacobian() pixels differ")
    if not result["float64_autograd_pixels_equal"]:
        raise AssertionError("autograd projection pixels differ")
    if result["float64_autograd"]["relative"]["p99"] >= 1e-10:
        raise AssertionError("analytic Jacobian does not match float64 autograd")
    if result["float64_finite_difference"]["relative"]["p99"] >= 1e-3:
        raise AssertionError("analytic Jacobian p99 relative error exceeds 1e-3")
    if coverage["negative_z_count"] == 0 or coverage["near_zero_z_count"] == 0:
        raise AssertionError("special ray_z regions were not covered")
    if coverage["near_projection_domain_boundary_count"] == 0:
        raise AssertionError("projection domain boundary was not covered")
    return result


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("debug/camera_model_v1"))
    parser.add_argument("--samples", type=int, default=20_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 100:
        raise ValueError("--samples must be at least 100")
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "data_root": str(args.data_root),
        "jacobian_definition": "d(u,v) / d(X,Y,Z)",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "thresholds": {
            "float64_autograd_p99_relative": 1e-10,
            "float64_finite_difference_p99_relative": 1e-3,
            "cuda_float32_p99_relative": 1e-5,
        },
    }
    try:
        dataset = StereoFisheyeDataset(args.data_root)
        for index, name in enumerate(("cam0", "cam1")):
            report[name] = validate_camera(make_camera(dataset, index), args.samples)
            write_report(report_path, report)
        cuda_complete = all(
            report[name]["cuda_float32_vs_cpu_float32"]["status"] == "passed"
            for name in ("cam0", "cam1")
        )
        report["status"] = "passed" if cuda_complete else "cpu_passed_cuda_unavailable"
        write_report(report_path, report)
        print(f"Camera Geometry V1 validation: {report['status']}")
        print(f"Report: {report_path}")
        return 0 if cuda_complete else 2
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        write_report(report_path, report)
        print(f"Camera Geometry V1 validation failed: {error}", file=sys.stderr)
        print(f"Partial report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
