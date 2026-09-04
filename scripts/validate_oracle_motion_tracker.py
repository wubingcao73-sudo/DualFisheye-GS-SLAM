#!/usr/bin/env python3
"""Validate the independent Oracle motion-only dual-fisheye rig tracker."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
import cv2
import numpy as np
from lietorch import SE3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera  # noqa: E402
from hislam2.data import DEFAULT_DATA_ROOT, StereoFisheyeDataset  # noqa: E402
from hislam2.geom.fisheye_reprojection import FisheyeRigReprojector  # noqa: E402
from hislam2.range import GroundTruthRangeProvider  # noqa: E402
from hislam2.tracking import (  # noqa: E402
    OracleCameraCorrespondences,
    OracleMotionOnlyConfig,
    OracleMotionOnlyTracker,
    OracleMotionProblem,
    build_oracle_motion_problem,
)


PAIR_INDICES = ((0, 1), (50, 51), (150, 151))
CAMERA_MODES = ("front", "back", "both")
CAMERA_LABELS = ("front", "back")


def _pyplot():
    """Import plotting lazily so it cannot interfere with CUDA discovery."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def distribution(values) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {name: math.nan for name in ("mean", "median", "p90", "p99", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def make_cameras(dataset: StereoFisheyeDataset) -> list[DoubleSphereCamera]:
    return [
        DoubleSphereCamera(
            *dataset.camera_params[index], *dataset.image_size[index]
        )
        for index in range(2)
    ]


def gt_rig_from_world(frame, dtype=torch.float64, device="cpu") -> torch.Tensor:
    return torch.linalg.inv(frame.gt_T_world_from_rig.to(dtype=dtype, device=device))


def perturb_pose(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return SE3.exp(delta).matrix() @ pose


def relative_pose_error(
    estimate: torch.Tensor, ground_truth: torch.Tensor
) -> tuple[float, float]:
    relative = estimate.double().cpu() @ torch.linalg.inv(ground_truth.double().cpu())
    translation = float(torch.linalg.vector_norm(relative[:3, 3]))
    rotation_matrix = relative[:3, :3]
    cosine = float((torch.trace(rotation_matrix) - 1.0) / 2.0)
    sine = 0.5 * float(
        torch.linalg.vector_norm(
            torch.stack(
                (
                    rotation_matrix[2, 1] - rotation_matrix[1, 2],
                    rotation_matrix[0, 2] - rotation_matrix[2, 0],
                    rotation_matrix[1, 0] - rotation_matrix[0, 1],
                )
            )
        )
    )
    rotation = math.atan2(sine, max(-1.0, min(1.0, cosine)))
    return translation, rotation


def world_position(pose_rig_from_world: torch.Tensor) -> np.ndarray:
    return torch.linalg.inv(pose_rig_from_world.double().cpu())[:3, 3].numpy()


def accepted_costs_monotonic(result) -> bool:
    costs = [item.candidate_cost for item in result.history if item.accepted]
    return all(right <= left + 1e-12 for left, right in zip(costs, costs[1:]))


def result_metrics(result, initial_pose, ground_truth) -> dict:
    initial_translation, initial_rotation = relative_pose_error(initial_pose, ground_truth)
    final_translation, final_rotation = relative_pose_error(
        result.T_rig_from_world_target, ground_truth
    )
    accepted = [item for item in result.history if item.accepted]
    return {
        "status": result.status,
        "iterations": result.iterations,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "cost_reduction_fraction": (
            (result.initial_cost - result.final_cost) / max(result.initial_cost, 1e-30)
            if result.initial_cost > 0.0
            else 0.0
        ),
        "initial_translation_error_m": initial_translation,
        "final_translation_error_m": final_translation,
        "translation_error_reduction_fraction": (
            (initial_translation - final_translation) / initial_translation
            if initial_translation > 1e-12
            else None
        ),
        "initial_rotation_error_deg": math.degrees(initial_rotation),
        "final_rotation_error_deg": math.degrees(final_rotation),
        "rotation_error_reduction_fraction": (
            (initial_rotation - final_rotation) / initial_rotation
            if initial_rotation > 1e-12
            else None
        ),
        "valid_count": result.valid_count,
        "camera_scales": list(result.camera_scales),
        "hessian_eigenvalues": list(result.hessian_eigenvalues),
        "hessian_condition_number": result.hessian_condition_number,
        "final_damping": result.final_damping,
        "accepted_costs_monotonic": accepted_costs_monotonic(result),
        "accepted_steps": len(accepted),
        "retry_count": len(result.history) - len(accepted),
        "history": [asdict(item) for item in result.history],
    }


def standard_passes(metrics: dict, translation_perturbed: bool, rotation_perturbed: bool) -> bool:
    if metrics["status"] != "converged" or not metrics["accepted_costs_monotonic"]:
        return False
    if metrics["final_translation_error_m"] >= 0.005:
        return False
    if metrics["final_rotation_error_deg"] >= 0.05:
        return False
    if translation_perturbed and metrics["translation_error_reduction_fraction"] < 0.90:
        return False
    if rotation_perturbed and metrics["rotation_error_reduction_fraction"] < 0.90:
        return False
    return True


def perturbation_scenarios() -> list[tuple[str, torch.Tensor, str]]:
    scenarios = [("gt_initial", torch.zeros(6, dtype=torch.float64), "gt")]
    names = ("tx", "ty", "tz", "rx", "ry", "rz")
    for axis, name in enumerate(names):
        magnitude = 0.05 if axis < 3 else math.radians(2.0)
        for sign, label in ((-1.0, "negative"), (1.0, "positive")):
            delta = torch.zeros(6, dtype=torch.float64)
            delta[axis] = sign * magnitude
            scenarios.append((f"axis_{name}_{label}", delta, "standard_axis"))
    translation_direction = torch.tensor((1.0, -2.0, 1.0), dtype=torch.float64)
    translation_direction /= torch.linalg.vector_norm(translation_direction)
    rotation_direction = torch.tensor((1.0, 1.0, -1.0), dtype=torch.float64)
    rotation_direction /= torch.linalg.vector_norm(rotation_direction)
    standard = torch.cat(
        (0.05 * translation_direction, math.radians(2.0) * rotation_direction)
    )
    difficult = torch.cat(
        (0.10 * translation_direction, math.radians(5.0) * rotation_direction)
    )
    scenarios.append(("mixed_standard", standard, "standard_mixed"))
    scenarios.append(("mixed_difficult", difficult, "difficult"))
    return scenarios


def _resize_for_output(image: np.ndarray, maximum_height: int = 1440) -> np.ndarray:
    if image.shape[0] <= maximum_height:
        return image
    scale = maximum_height / image.shape[0]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _resize_for_output(image)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {path}")


def _camera_canvases(frame) -> list[np.ndarray]:
    return [frame.rgb[index].permute(1, 2, 0).numpy().copy() for index in range(2)]


def _join_cameras(canvases: list[np.ndarray]) -> np.ndarray:
    height = max(canvas.shape[0] for canvas in canvases)
    normalized = []
    for canvas in canvases:
        if canvas.shape[0] != height:
            scale = height / canvas.shape[0]
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
        normalized.append(canvas)
    return np.concatenate(normalized, axis=1)


def render_pose_state(
    reprojector,
    problem,
    target_pose,
    source_frame,
    target_frame,
) -> tuple[np.ndarray, np.ndarray]:
    overlays = _camera_canvases(target_frame)
    warps = [np.zeros_like(image) for image in overlays]
    source_images = _camera_canvases(source_frame)
    for group in problem.groups:
        index = group.source_camera_index
        projected = reprojector.reproject(
            group.source_pixels,
            group.source_inverse_range,
            problem.T_rig_from_world_source,
            target_pose,
            index,
            index,
        )
        valid_indices = torch.nonzero(projected.validity.geometric_valid, as_tuple=False).flatten()
        if not len(valid_indices):
            continue
        predicted = projected.pixels[valid_indices].detach().cpu().numpy()
        observed = group.observed_target_pixels[valid_indices].detach().cpu().numpy()
        source_pixels = group.source_pixels[valid_indices].detach().cpu().numpy()
        source_rgb = source_images[index]
        height, width = warps[index].shape[:2]
        for source_pixel, target_pixel in zip(source_pixels, predicted):
            u, v = np.rint(target_pixel).astype(np.int64)
            su, sv = np.rint(source_pixel).astype(np.int64)
            if 0 <= u < width and 0 <= v < height:
                color = source_rgb[np.clip(sv, 0, height - 1), np.clip(su, 0, width - 1)]
                warps[index][max(0, v - 1) : min(height, v + 2), max(0, u - 1) : min(width, u + 2)] = color
        sample_count = min(800, len(predicted))
        selected = np.linspace(0, len(predicted) - 1, sample_count, dtype=np.int64)
        for predicted_pixel, observed_pixel in zip(predicted[selected], observed[selected]):
            p0 = tuple(np.rint(predicted_pixel).astype(np.int32))
            p1 = tuple(np.rint(observed_pixel).astype(np.int32))
            cv2.arrowedLine(overlays[index], p0, p1, (255, 0, 0), 2, tipLength=0.2)
            cv2.circle(overlays[index], p1, 3, (0, 255, 0), -1)
    return _join_cameras(overlays), _join_cameras(warps)


def plot_convergence(path: Path, result) -> None:
    plt = _pyplot()
    accepted = [item for item in result.history if item.accepted]
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if accepted:
        x = np.arange(1, len(accepted) + 1)
        axes[0].semilogy(x, [max(item.candidate_cost, 1e-30) for item in accepted], marker="o")
        axes[1].semilogy(x, [item.damping for item in accepted], marker="o")
    axes[0].set_ylabel("robust mean cost")
    axes[1].set_ylabel("LM damping")
    axes[1].set_xlabel("accepted step")
    axes[0].grid(True)
    axes[1].grid(True)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def visualize_case(output, reprojector, problem, source_frame, target_frame, initial, result) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    initial_overlay, initial_warp = render_pose_state(
        reprojector, problem, initial, source_frame, target_frame
    )
    optimized_overlay, optimized_warp = render_pose_state(
        reprojector,
        problem,
        result.T_rig_from_world_target,
        source_frame,
        target_frame,
    )
    target_rgb = _join_cameras(_camera_canvases(target_frame))
    _write_rgb(output / "initial_overlay.png", initial_overlay)
    _write_rgb(output / "optimized_overlay.png", optimized_overlay)
    _write_rgb(output / "initial_warp.png", initial_warp)
    _write_rgb(output / "optimized_warp.png", optimized_warp)
    _write_rgb(output / "target_rgb.png", target_rgb)
    plot_convergence(output / "convergence.png", result)
    return {
        "initial_overlay": str(output / "initial_overlay.png"),
        "optimized_overlay": str(output / "optimized_overlay.png"),
        "initial_warp": str(output / "initial_warp.png"),
        "optimized_warp": str(output / "optimized_warp.png"),
        "target_rgb": str(output / "target_rgb.png"),
        "convergence": str(output / "convergence.png"),
    }


def cast_group(group, dtype, device):
    if group is None:
        return None
    return OracleCameraCorrespondences(
        source_pixels=group.source_pixels.to(dtype=dtype, device=device),
        source_inverse_range=group.source_inverse_range.to(dtype=dtype, device=device),
        observed_target_pixels=group.observed_target_pixels.to(dtype=dtype, device=device),
        source_camera_index=group.source_camera_index,
        target_camera_index=group.target_camera_index,
        fixed_validity=group.fixed_validity.to(device=device),
        base_weights=group.base_weights.to(dtype=dtype, device=device),
    )


def cast_problem(problem, dtype, device):
    return OracleMotionProblem(
        problem.T_rig_from_world_source.to(dtype=dtype, device=device),
        cast_group(problem.front, dtype, device),
        cast_group(problem.back, dtype, device),
    )


def precision_validation(cameras, extrinsics64, problem64, gt64, delta64, config):
    reference = OracleMotionOnlyTracker(
        FisheyeRigReprojector(cameras, extrinsics64), config
    ).optimize(problem64, perturb_pose(gt64, delta64))
    problem32 = cast_problem(problem64, torch.float32, "cpu")
    gt32 = gt64.float()
    delta32 = delta64.float()
    cpu = OracleMotionOnlyTracker(
        FisheyeRigReprojector(cameras, extrinsics64.float()), config
    ).optimize(problem32, perturb_pose(gt32, delta32))
    cpu_translation, cpu_rotation = relative_pose_error(
        cpu.T_rig_from_world_target, reference.T_rig_from_world_target
    )
    report = {
        "cpu_float32": {
            "status": cpu.status,
            "vs_float64_translation_m": cpu_translation,
            "vs_float64_rotation_deg": math.degrees(cpu_rotation),
        }
    }
    if cpu_translation >= 0.001 or math.degrees(cpu_rotation) >= 0.01:
        raise AssertionError(f"CPU float32 optimizer mismatch: {report}")
    if not torch.cuda.is_available():
        report["cuda_float32"] = {"status": "unavailable"}
        return report
    device = torch.device("cuda")
    cuda = OracleMotionOnlyTracker(
        FisheyeRigReprojector(cameras, extrinsics64.float().to(device)), config
    ).optimize(
        cast_problem(problem64, torch.float32, device),
        perturb_pose(gt64.float().to(device), delta64.float().to(device)),
    )
    cuda_translation, cuda_rotation = relative_pose_error(
        cuda.T_rig_from_world_target, reference.T_rig_from_world_target
    )
    cpu_cuda_translation, cpu_cuda_rotation = relative_pose_error(
        cuda.T_rig_from_world_target, cpu.T_rig_from_world_target
    )
    report["cuda_float32"] = {
        "status": cuda.status,
        "device": torch.cuda.get_device_name(0),
        "vs_float64_translation_m": cuda_translation,
        "vs_float64_rotation_deg": math.degrees(cuda_rotation),
        "vs_cpu_float32_translation_m": cpu_cuda_translation,
        "vs_cpu_float32_rotation_deg": math.degrees(cpu_cuda_rotation),
    }
    if (
        cuda_translation >= 0.001
        or math.degrees(cuda_rotation) >= 0.01
        or cpu_cuda_translation >= 0.001
        or math.degrees(cpu_cuda_rotation) >= 0.01
    ):
        raise AssertionError(f"CUDA float32 optimizer mismatch: {report}")
    return report


def rpe(estimate_source, estimate_target, gt_source, gt_target):
    estimate_relative = estimate_target @ torch.linalg.inv(estimate_source)
    gt_relative = gt_target @ torch.linalg.inv(gt_source)
    return relative_pose_error(estimate_relative, gt_relative)


def validate_sequence(
    dataset,
    reprojector,
    provider,
    config,
    stride,
    length,
    output,
):
    tracker = OracleMotionOnlyTracker(reprojector, config)
    source_frame = dataset[0]
    estimate = gt_rig_from_world(source_frame)
    estimates = [estimate.detach().clone()]
    ground_truth = [estimate.detach().clone()]
    rows = []
    translation_errors = [0.0]
    rotation_errors = [0.0]
    translation_rpe = [0.0]
    rotation_rpe = [0.0]
    condition_numbers = [math.nan]
    iterations = [0]
    valid_counts = [0]
    cost_reductions = [0.0]
    for target_index in range(1, length):
        target_frame = dataset[target_index]
        problem, target_gt = build_oracle_motion_problem(
            source_frame,
            target_frame,
            reprojector,
            provider,
            fixed_source_pose=estimate,
            stride=stride,
        )
        result = tracker.optimize(problem, estimate)
        if result.status != "converged":
            raise AssertionError(
                f"sequence pair {target_index - 1}->{target_index} failed: {result.status}"
            )
        if not accepted_costs_monotonic(result):
            raise AssertionError("accepted sequence costs are not monotonic")
        previous_estimate = estimate
        previous_gt = ground_truth[-1]
        estimate = result.T_rig_from_world_target.detach().clone()
        estimates.append(estimate)
        ground_truth.append(target_gt)
        translation, rotation = relative_pose_error(estimate, target_gt)
        pair_translation, pair_rotation = rpe(
            previous_estimate, estimate, previous_gt, target_gt
        )
        reduction = (
            (result.initial_cost - result.final_cost) / max(result.initial_cost, 1e-30)
            if result.initial_cost > 0.0
            else 0.0
        )
        translation_errors.append(translation)
        rotation_errors.append(math.degrees(rotation))
        translation_rpe.append(pair_translation)
        rotation_rpe.append(math.degrees(pair_rotation))
        condition_numbers.append(result.hessian_condition_number)
        iterations.append(result.iterations)
        valid_counts.append(result.valid_count)
        cost_reductions.append(reduction)
        position_estimate = world_position(estimate)
        position_gt = world_position(target_gt)
        rows.append(
            {
                "frame": target_index,
                "source_frame": target_index - 1,
                "status": result.status,
                "iterations": result.iterations,
                "retries": len(result.history) - sum(item.accepted for item in result.history),
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "cost_reduction_fraction": reduction,
                "valid_count": result.valid_count,
                "condition_number": result.hessian_condition_number,
                "translation_error_m": translation,
                "rotation_error_deg": math.degrees(rotation),
                "translation_rpe_m": pair_translation,
                "rotation_rpe_deg": math.degrees(pair_rotation),
                "estimated_world_x": position_estimate[0],
                "estimated_world_y": position_estimate[1],
                "estimated_world_z": position_estimate[2],
                "gt_world_x": position_gt[0],
                "gt_world_y": position_gt[1],
                "gt_world_z": position_gt[2],
            }
        )
        source_frame = target_frame

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    estimate_positions = np.stack([world_position(pose) for pose in estimates])
    gt_positions = np.stack([world_position(pose) for pose in ground_truth])
    plot_trajectory(output / "trajectory.png", estimate_positions, gt_positions)
    plot_sequence_errors(
        output / "trajectory_error.png", translation_errors, rotation_errors,
        translation_rpe, rotation_rpe
    )
    plot_sequence_diagnostics(
        output / "sequence_diagnostics.png", iterations, valid_counts,
        condition_numbers, cost_reductions
    )
    ate = np.linalg.norm(estimate_positions - gt_positions, axis=1)
    finite_conditions = [value for value in condition_numbers if math.isfinite(value)]
    return {
        "frame_count": length,
        "pair_count": length - 1,
        "anchoring": "frame 0 GT only; each later fixed source and target initial are previous estimate",
        "trajectory_position": "inverse(T_rig_from_world)[:3,3]",
        "alignment": "none",
        "ate_rmse_m": float(np.sqrt(np.mean(ate * ate))),
        "absolute_translation_error_m": distribution(translation_errors),
        "absolute_rotation_error_deg": distribution(rotation_errors),
        "translation_rpe_m": distribution(translation_rpe[1:]),
        "rotation_rpe_deg": distribution(rotation_rpe[1:]),
        "iterations": distribution(iterations[1:]),
        "valid_count": distribution(valid_counts[1:]),
        "hessian_condition_number": distribution(finite_conditions),
        "cost_reduction_fraction": distribution(cost_reductions[1:]),
        "outputs": {
            "trajectory": str(output / "trajectory.png"),
            "trajectory_error": str(output / "trajectory_error.png"),
            "sequence_diagnostics": str(output / "sequence_diagnostics.png"),
            "csv": str(csv_path),
        },
    }


def plot_trajectory(path, estimate, ground_truth):
    plt = _pyplot()
    figure = plt.figure(figsize=(12, 9))
    axis3d = figure.add_subplot(221, projection="3d")
    axis3d.plot(
        *estimate.T,
        color="tab:orange",
        linewidth=3.0,
        label="Estimated (solid)",
        zorder=2,
    )
    axis3d.plot(
        *ground_truth.T,
        color="tab:blue",
        linewidth=2.0,
        linestyle=(0, (4, 3)),
        label="GT (dashed)",
        zorder=3,
    )
    axis3d.scatter(*ground_truth[0], color="limegreen", marker="o", s=45, label="Start")
    axis3d.scatter(*ground_truth[-1], color="red", marker="x", s=55, label="End")
    axis3d.set_title("GT is dashed above Estimated; gaps reveal the solid estimate")
    axis3d.legend()
    for panel, axes, labels in (
        (222, (0, 1), ("x", "y")),
        (223, (0, 2), ("x", "z")),
        (224, (1, 2), ("y", "z")),
    ):
        axis = figure.add_subplot(panel)
        axis.plot(
            estimate[:, axes[0]],
            estimate[:, axes[1]],
            color="tab:orange",
            linewidth=3.0,
            label="Estimated (solid)",
            zorder=2,
        )
        axis.plot(
            ground_truth[:, axes[0]],
            ground_truth[:, axes[1]],
            color="tab:blue",
            linewidth=2.0,
            linestyle=(0, (4, 3)),
            label="GT (dashed)",
            zorder=3,
        )
        axis.scatter(
            ground_truth[0, axes[0]], ground_truth[0, axes[1]],
            color="limegreen", marker="o", s=35, zorder=4
        )
        axis.scatter(
            ground_truth[-1, axes[0]], ground_truth[-1, axes[1]],
            color="red", marker="x", s=45, zorder=4
        )
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.axis("equal")
        axis.grid(True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_sequence_errors(path, translation, rotation, translation_rpe, rotation_rpe):
    plt = _pyplot()
    frames = np.arange(len(translation))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(frames, translation)
    axes[0, 0].set_ylabel("absolute translation [m]")
    axes[0, 1].plot(frames, rotation)
    axes[0, 1].set_ylabel("absolute rotation [deg]")
    axes[1, 0].plot(frames, translation_rpe)
    axes[1, 0].set_ylabel("translation RPE [m]")
    axes[1, 1].plot(frames, rotation_rpe)
    axes[1, 1].set_ylabel("rotation RPE [deg]")
    for axis in axes.flat:
        axis.grid(True)
        axis.set_xlabel("frame")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_sequence_diagnostics(path, iterations, valid_counts, conditions, reductions):
    plt = _pyplot()
    frames = np.arange(len(iterations))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(frames, iterations)
    axes[0, 0].set_ylabel("iterations")
    axes[0, 1].plot(frames, valid_counts)
    axes[0, 1].set_ylabel("valid correspondences")
    axes[1, 0].semilogy(frames, conditions)
    axes[1, 0].set_ylabel("Hessian condition")
    axes[1, 1].plot(frames, reductions)
    axes[1, 1].set_ylabel("cost reduction")
    for axis in axes.flat:
        axis.grid(True)
        axis.set_xlabel("frame")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("debug/oracle_motion_tracker"))
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "data_root": str(args.data_root),
        "pose_state": "T_rig_from_world",
        "left_update": "G_new = Exp(delta) @ G",
        "twist_order": ["tx", "ty", "tz", "rx", "ry", "rz"],
        "oracle_boundary": {
            "builder": "GT range and source/target GT poses generate fixed correspondences",
            "optimizer": "fixed problem, fixed source pose, and target initial pose only",
            "target_gt_in_problem": False,
        },
        "builder": {
            "stride": args.stride,
            "safety_margin_px": 128.0,
            "range_sampling": "bilinear align_corners=True; xn=2u/(W-1)-1, yn=2v/(H-1)-1",
            "occlusion": "abs(predicted-target_gt) <= max(0.01m, 0.01*target_gt)",
        },
        "config": asdict(OracleMotionOnlyConfig()),
        "thresholds": {
            "perturbed_component_reduction_fraction": 0.90,
            "final_translation_m": 0.005,
            "final_rotation_deg": 0.05,
            "float32_vs_float64_translation_m": 0.001,
            "float32_vs_float64_rotation_deg": 0.01,
        },
    }
    try:
        if args.stride < 1:
            raise ValueError("--stride must be positive")
        if not 2 <= args.sequence_length <= 300:
            raise ValueError("--sequence-length must lie in [2, 300]")
        dataset = StereoFisheyeDataset(args.data_root)
        cameras = make_cameras(dataset)
        extrinsics64 = torch.from_numpy(dataset.T_rig_from_camera).double()
        reprojector = FisheyeRigReprojector(cameras, extrinsics64)
        provider = GroundTruthRangeProvider()
        config = OracleMotionOnlyConfig()
        tracker = OracleMotionOnlyTracker(reprojector, config)
        scenarios = perturbation_scenarios()
        standard_failures = []
        report["frame_pairs"] = {}
        for source_index, target_index in PAIR_INDICES:
            source_frame = dataset[source_index]
            target_frame = dataset[target_index]
            full_problem, target_gt = build_oracle_motion_problem(
                source_frame,
                target_frame,
                reprojector,
                provider,
                stride=args.stride,
            )
            pair_key = f"{source_index:04d}_to_{target_index:04d}"
            pair_report = {
                "fixed_correspondences": {
                    "front": full_problem.front.count,
                    "back": full_problem.back.count,
                },
                "modes": {},
            }
            if "precision" not in report:
                mixed_delta = next(
                    delta for name, delta, _ in scenarios if name == "mixed_standard"
                )
                # Run CUDA before the first lazy plotting import.  Some OpenCV/
                # Matplotlib builds load libraries that interfere with CUDA discovery.
                report["precision"] = precision_validation(
                    cameras,
                    extrinsics64,
                    full_problem,
                    target_gt,
                    mixed_delta,
                    config,
                )
                write_report(report_path, report)
            for mode in CAMERA_MODES:
                problem = full_problem.camera_mode(mode)
                mode_report = {}
                for scenario_name, delta, kind in scenarios:
                    initial = perturb_pose(target_gt, delta)
                    result = tracker.optimize(problem, initial)
                    metrics = result_metrics(result, initial, target_gt)
                    metrics["kind"] = kind
                    if kind == "gt":
                        metrics["passed"] = (
                            metrics["status"] == "converged"
                            and metrics["initial_cost"] < 1e-16
                        )
                    elif kind.startswith("standard"):
                        metrics["passed"] = standard_passes(
                            metrics,
                            translation_perturbed=bool(float(torch.linalg.vector_norm(delta[:3])) > 0.0),
                            rotation_perturbed=bool(float(torch.linalg.vector_norm(delta[3:])) > 0.0),
                        )
                        if mode == "both" and not metrics["passed"]:
                            standard_failures.append(f"{pair_key}/{scenario_name}")
                    else:
                        metrics["passed"] = standard_passes(metrics, True, True)
                    if (
                        not args.skip_visualization
                        and scenario_name == "mixed_standard"
                    ):
                        metrics["visualization"] = visualize_case(
                            args.output / "pairs" / pair_key / mode / scenario_name,
                            reprojector,
                            problem,
                            source_frame,
                            target_frame,
                            initial,
                            result,
                        )
                    mode_report[scenario_name] = metrics
                pair_report["modes"][mode] = mode_report
            report["frame_pairs"][pair_key] = pair_report
            write_report(report_path, report)

        if standard_failures:
            raise AssertionError(f"standard Front+Back cases failed: {standard_failures}")
        report["sequence"] = validate_sequence(
            dataset,
            reprojector,
            provider,
            config,
            args.stride,
            args.sequence_length,
            args.output / "sequence",
        )
        report["difficult_success"] = {
            mode: {
                "success_count": sum(
                    report["frame_pairs"][key]["modes"][mode]["mixed_difficult"]["passed"]
                    for key in report["frame_pairs"]
                ),
                "total": len(PAIR_INDICES),
            }
            for mode in CAMERA_MODES
        }
        report["status"] = "passed"
        write_report(report_path, report)
        print("Oracle motion-only rig tracker validation: passed")
        print(f"Report: {report_path}")
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        write_report(report_path, report)
        print(f"Oracle motion-only rig tracker validation failed: {error}", file=sys.stderr)
        print(f"Partial report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
