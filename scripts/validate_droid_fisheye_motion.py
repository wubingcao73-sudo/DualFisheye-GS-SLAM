#!/usr/bin/env python3
"""Validate frozen DROID correspondences with dual-fisheye motion-only LM."""

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

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera  # noqa: E402
from hislam2.data import DEFAULT_DATA_ROOT, StereoFisheyeDataset  # noqa: E402
from hislam2.geom.fisheye_reprojection import FisheyeRigReprojector  # noqa: E402
from hislam2.range import GroundTruthRangeProvider  # noqa: E402
from hislam2.tracking import (  # noqa: E402
    DroidFisheyeMotionConfig,
    DroidFisheyeMotionTracker,
    feature_to_processed_pixels,
    load_pretrained_droid,
)
from hislam2.tracking.oracle_motion_only import OracleMotionOnlyConfig  # noqa: E402


PAIR_INDICES = ((0, 1), (50, 51), (150, 151))
CAMERA_MODES = ("front", "back", "both")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def write_report(path: Path, report: dict) -> None:
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=convert), encoding="utf-8"
    )
    os.replace(temporary, path)


def make_cameras(dataset) -> tuple[DoubleSphereCamera, DoubleSphereCamera]:
    return tuple(
        DoubleSphereCamera(*dataset.camera_params[index], *dataset.image_size[index])
        for index in range(2)
    )


def gt_pose(frame, device="cpu") -> torch.Tensor:
    return torch.linalg.inv(frame.gt_T_world_from_rig.float().to(device))


def pose_error(estimate: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    error = estimate.double().cpu() @ torch.linalg.inv(truth.double().cpu())
    translation = float(torch.linalg.vector_norm(error[:3, 3]))
    rotation = error[:3, :3]
    cosine = max(-1.0, min(1.0, float((torch.trace(rotation) - 1.0) / 2.0)))
    sine = 0.5 * float(
        torch.linalg.vector_norm(
            torch.tensor(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ],
                dtype=torch.float64,
            )
        )
    )
    return translation, math.degrees(math.atan2(sine, cosine))


def world_position(pose: torch.Tensor) -> np.ndarray:
    return torch.linalg.inv(pose.double().cpu())[:3, 3].numpy()


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {key: None for key in ("mean", "median", "p90", "p99", "maximum")}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(values.max()),
    }


def serialize_lm(result) -> dict:
    return {
        "status": result.status,
        "iterations": result.iterations,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "valid_count": result.valid_count,
        "camera_scales": list(result.camera_scales),
        "hessian_eigenvalues": list(result.hessian_eigenvalues),
        "hessian_condition_number": result.hessian_condition_number,
        "final_damping": result.final_damping,
        "accepted_costs_monotonic": all(
            right <= left + 1e-9
            for left, right in zip(
                [item.candidate_cost for item in result.history if item.accepted],
                [item.candidate_cost for item in result.history if item.accepted][1:],
            )
        ),
        "history": [asdict(item) for item in result.history],
    }


def serialize_tracking(result, initial, truth) -> dict:
    initial_translation, initial_rotation = pose_error(initial, truth)
    final_translation, final_rotation = pose_error(result.T_rig_from_world_target, truth)
    return {
        "status": result.status,
        "outer_iterations": result.iterations,
        "initial_translation_error_m": initial_translation,
        "initial_rotation_error_deg": initial_rotation,
        "final_translation_error_m": final_translation,
        "final_rotation_error_deg": final_rotation,
        "finite_pose": bool(torch.isfinite(result.T_rig_from_world_target).all()),
        "outer_history": [
            {
                "iteration": item.iteration,
                "status": item.status,
                "correspondence_count": item.correspondence_count,
                "front_count": item.front_count,
                "back_count": item.back_count,
                "weight_mean": item.weight_mean,
                "weight_minimum": item.weight_minimum,
                "weight_maximum": item.weight_maximum,
                "lm": serialize_lm(item.lm_result),
            }
            for item in result.history
        ],
    }


def oracle_correspondence_errors(tracker, problem, source_gt, target_gt):
    per_camera = {}
    raw = {}
    for group in problem.groups:
        camera = "front" if group.source_camera_index == 0 else "back"
        oracle = tracker.feature_reprojector.reproject(
            group.source_pixels,
            group.source_inverse_range,
            source_gt,
            target_gt,
            group.source_camera_index,
            group.target_camera_index,
        )
        valid = oracle.validity.geometric_valid
        errors = torch.linalg.vector_norm(
            group.observed_target_pixels[valid] - oracle.pixels[valid], dim=-1
        ).detach().cpu().numpy()
        metrics = distribution(errors)
        metrics["count"] = int(valid.sum())
        metrics["inlier_fraction_below_1px"] = float(np.mean(errors < 1.0)) if len(errors) else 0.0
        per_camera[camera] = metrics
        raw[camera] = (errors, group, oracle, valid)
    return per_camera, raw


def _processed_rgb(frame, height, width) -> list[np.ndarray]:
    images = []
    for camera in frame.rgb:
        image = camera.permute(1, 2, 0).numpy()
        images.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
    return images


def _join(images):
    return np.concatenate(images, axis=1)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise IOError(f"failed to write {path}")


def render_correspondence_overlay(tracker, problem, source_gt, target_gt, target_frame):
    canvases = _processed_rgb(
        target_frame, tracker.config.processed_height, tracker.config.processed_width
    )
    for group in problem.groups:
        index = group.source_camera_index
        oracle = tracker.feature_reprojector.reproject(
            group.source_pixels,
            group.source_inverse_range,
            source_gt,
            target_gt,
            index,
            index,
        )
        valid = oracle.validity.geometric_valid
        observed = feature_to_processed_pixels(group.observed_target_pixels[valid]).cpu().numpy()
        truth = feature_to_processed_pixels(oracle.pixels[valid]).cpu().numpy()
        if not len(observed):
            continue
        chosen = np.linspace(0, len(observed) - 1, min(800, len(observed)), dtype=np.int64)
        for prediction, gt in zip(observed[chosen], truth[chosen]):
            p0 = tuple(np.rint(prediction).astype(np.int32))
            p1 = tuple(np.rint(gt).astype(np.int32))
            cv2.arrowedLine(canvases[index], p0, p1, (255, 40, 40), 1, tipLength=0.25)
            cv2.circle(canvases[index], p1, 2, (40, 255, 40), -1)
    return _join(canvases)


def render_pose_overlay(tracker, problem, pose, target_frame):
    canvases = _processed_rgb(
        target_frame, tracker.config.processed_height, tracker.config.processed_width
    )
    for group in problem.groups:
        index = group.source_camera_index
        projected = tracker.feature_reprojector.reproject(
            group.source_pixels,
            group.source_inverse_range,
            problem.T_rig_from_world_source,
            pose,
            index,
            index,
        )
        valid = projected.validity.geometric_valid
        predicted = feature_to_processed_pixels(projected.pixels[valid]).cpu().numpy()
        observed = feature_to_processed_pixels(group.observed_target_pixels[valid]).cpu().numpy()
        chosen = np.linspace(0, len(predicted) - 1, min(800, len(predicted)), dtype=np.int64)
        for prediction, observation in zip(predicted[chosen], observed[chosen]):
            p0 = tuple(np.rint(prediction).astype(np.int32))
            p1 = tuple(np.rint(observation).astype(np.int32))
            cv2.arrowedLine(canvases[index], p0, p1, (255, 50, 50), 1, tipLength=0.25)
            cv2.circle(canvases[index], p1, 2, (50, 255, 50), -1)
    return _join(canvases)


def render_scalar_maps(tracker, raw, name):
    maps = []
    for camera in ("front", "back"):
        canvas = np.full(
            (tracker.config.feature_height, tracker.config.feature_width), np.nan, np.float32
        )
        if camera in raw:
            errors, group, _, valid = raw[camera]
            pixels = group.source_pixels[valid].cpu().numpy()
            x = np.rint(pixels[:, 0]).astype(np.int64)
            y = np.rint(pixels[:, 1]).astype(np.int64)
            values = errors if name == "epe" else group.base_weights[valid].cpu().numpy()
            canvas[y, x] = values
        finite = np.isfinite(canvas)
        if finite.any():
            maximum = np.percentile(canvas[finite], 99) if name == "epe" else 1.0
            normalized = np.nan_to_num(canvas / max(float(maximum), 1e-6), nan=0.0)
        else:
            normalized = np.zeros_like(canvas)
        colored = cv2.applyColorMap(
            np.clip(255.0 * normalized, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO
        )
        maps.append(cv2.cvtColor(cv2.resize(
            colored,
            (tracker.config.processed_width, tracker.config.processed_height),
            interpolation=cv2.INTER_NEAREST,
        ), cv2.COLOR_BGR2RGB))
    return _join(maps)


def plot_convergence(path, result):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 5))
    offset = 0
    for outer in result.history:
        costs = [item.candidate_cost for item in outer.lm_result.history if item.accepted]
        if costs:
            x = np.arange(offset, offset + len(costs))
            axis.semilogy(x, costs, marker="o", label=f"DROID/LM outer {outer.iteration + 1}")
            offset += len(costs)
    axis.set_xlabel("accepted LM step")
    axis.set_ylabel("fixed-denominator robust cost")
    axis.grid(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_pair_visuals(output, tracker, result, raw, source_gt, target_gt, target_frame):
    output.mkdir(parents=True, exist_ok=True)
    problem = result.final_problem
    initial = result.history[0].pose_before
    _write_rgb(
        output / "droid_correspondence_overlay.png",
        render_correspondence_overlay(tracker, problem, source_gt, target_gt, target_frame),
    )
    _write_rgb(output / "droid_epe_heatmap.png", render_scalar_maps(tracker, raw, "epe"))
    _write_rgb(output / "weight_heatmap.png", render_scalar_maps(tracker, raw, "weight"))
    _write_rgb(
        output / "initial_pose_overlay.png",
        render_pose_overlay(tracker, problem, initial, target_frame),
    )
    _write_rgb(
        output / "optimized_pose_overlay.png",
        render_pose_overlay(tracker, problem, result.T_rig_from_world_target, target_frame),
    )
    plot_convergence(output / "convergence.png", result)


def pair_validation(dataset, tracker, output):
    results = {}
    visual_written = False
    for source_index, target_index in PAIR_INDICES:
        source_frame, target_frame = dataset[source_index], dataset[target_index]
        source_features = tracker.extract_features(source_frame.rgb, source_index)
        target_features = tracker.extract_features(target_frame.rgb, target_index)
        geometry = tracker.prepare_source_geometry(source_frame)
        source_gt = gt_pose(source_frame, tracker.device)
        target_gt = gt_pose(target_frame, tracker.device)
        pair_key = f"{source_index}->{target_index}"
        results[pair_key] = {}
        for mode in CAMERA_MODES:
            tracked = tracker.track_pair(
                source_features, target_features, geometry, source_gt, source_gt, camera_mode=mode
            )
            metrics = serialize_tracking(tracked, source_gt, target_gt)
            if mode == "both":
                correspondence, raw = oracle_correspondence_errors(
                    tracker, tracked.final_problem, source_gt, target_gt
                )
                metrics["correspondence_feature_px"] = correspondence
                if not visual_written:
                    write_pair_visuals(
                        output / "pair_000000_000001",
                        tracker,
                        tracked,
                        raw,
                        source_gt,
                        target_gt,
                        target_frame,
                    )
                    visual_written = True
            results[pair_key][mode] = metrics
        print(f"validated pair {pair_key}", flush=True)
    return results


def relative_pose(start, end):
    return end @ torch.linalg.inv(start)


def sequence_metrics(poses, truths):
    positions = np.stack([world_position(pose) for pose in poses])
    truth_positions = np.stack([world_position(pose) for pose in truths])
    absolute = np.linalg.norm(positions - truth_positions, axis=1)
    translation_rpe, rotation_rpe = [], []
    for index in range(len(poses) - 1):
        estimate_relative = relative_pose(poses[index], poses[index + 1])
        truth_relative = relative_pose(truths[index], truths[index + 1])
        translation, rotation = pose_error(estimate_relative, truth_relative)
        translation_rpe.append(translation)
        rotation_rpe.append(rotation)
    return {
        "positions": positions,
        "truth_positions": truth_positions,
        "absolute_translation_errors_m": absolute,
        "translation_rpe_m": np.asarray(translation_rpe),
        "rotation_rpe_deg": np.asarray(rotation_rpe),
        "ate_rmse_m": float(np.sqrt(np.mean(absolute ** 2))),
        "translation_rpe_rmse_m": float(np.sqrt(np.mean(np.square(translation_rpe)))),
        "rotation_rpe_rmse_deg": float(np.sqrt(np.mean(np.square(rotation_rpe)))),
    }


def plot_trajectory(path, mode_results, truth_positions):
    plt = _pyplot()
    figure = plt.figure(figsize=(11, 9))
    axis = figure.add_subplot(111, projection="3d")
    colors = {"front": "tab:orange", "back": "tab:green", "both": "tab:red"}
    plotted = [truth_positions]
    for mode, metrics in mode_results.items():
        points = metrics["numeric"]["positions"]
        plotted.append(points)
        axis.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color=colors[mode], linewidth=1.7, label=mode,
        )
    axis.plot(
        truth_positions[:, 0],
        truth_positions[:, 1],
        truth_positions[:, 2],
        "b--",
        linewidth=2.4,
        label="GT (drawn on top)",
    )
    axis.scatter(
        *truth_positions[0], c="lime", edgecolors="black", s=65, depthshade=False,
        label="start",
    )
    axis.scatter(
        *truth_positions[-1], c="red", edgecolors="black", s=65, depthshade=False,
        label="end",
    )
    all_points = np.concatenate(plotted, axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_zlabel("world z [m]")
    axis.view_init(elev=24, azim=-55)
    axis.grid(True)
    axis.legend(loc="upper left")
    figure.suptitle("Rig world position = translation of inverse(T_rig_from_world); no alignment")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_trajectory_from_csv(csv_path: Path, output_path: Path) -> None:
    """Regenerate the 3-D trajectory without rerunning network inference."""
    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    truth_positions = np.column_stack((table["gt_x"], table["gt_y"], table["gt_z"]))
    mode_results = {}
    for mode in CAMERA_MODES:
        positions = np.column_stack(
            (table[f"{mode}_x"], table[f"{mode}_y"], table[f"{mode}_z"])
        )
        mode_results[mode] = {"numeric": {"positions": positions}}
    plot_trajectory(output_path, mode_results, truth_positions)


def plot_sequence_errors(path, mode_results):
    plt = _pyplot()
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for mode, metrics in mode_results.items():
        numeric = metrics["numeric"]
        axes[0].plot(numeric["absolute_translation_errors_m"] * 1000.0, label=mode)
        axes[1].plot(np.arange(1, len(numeric["translation_rpe_m"]) + 1), numeric["translation_rpe_m"] * 1000.0, label=mode)
        axes[2].plot(np.arange(1, len(numeric["rotation_rpe_deg"]) + 1), numeric["rotation_rpe_deg"], label=mode)
    axes[0].set_ylabel("absolute t [mm]")
    axes[1].set_ylabel("translation RPE [mm]")
    axes[2].set_ylabel("rotation RPE [deg]")
    axes[2].set_xlabel("frame")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_diagnostics(path, mode_results):
    plt = _pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for mode, metrics in mode_results.items():
        records = metrics["records"]
        axes[0].plot([record["minimum_correspondences"] for record in records], label=mode)
        axes[1].semilogy(
            [max(record["maximum_hessian_condition"], 1.0) for record in records], label=mode
        )
    axes[0].set_ylabel("minimum fixed correspondences")
    axes[1].set_ylabel("maximum Hessian condition")
    axes[1].set_xlabel("pair index")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_trajectory_csv(path, mode_results, truths, timestamps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["frame", "timestamp", "gt_x", "gt_y", "gt_z"]
        for mode in CAMERA_MODES:
            fieldnames.extend(
                [f"{mode}_x", f"{mode}_y", f"{mode}_z", f"{mode}_translation_error_m", f"{mode}_rotation_error_deg"]
            )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, truth in enumerate(truths):
            gt = world_position(truth)
            row = {"frame": index, "timestamp": timestamps[index], "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2]}
            for mode in CAMERA_MODES:
                pose = mode_results[mode]["poses"][index]
                position = world_position(pose)
                translation, rotation = pose_error(pose, truth)
                row.update(
                    {
                        f"{mode}_x": position[0], f"{mode}_y": position[1], f"{mode}_z": position[2],
                        f"{mode}_translation_error_m": translation,
                        f"{mode}_rotation_error_deg": rotation,
                    }
                )
            writer.writerow(row)


def sequence_validation(dataset, tracker, output, length):
    output.mkdir(parents=True, exist_ok=True)
    features, geometries, truths, timestamps = [], [], [], []
    for index in range(length):
        frame = dataset[index]
        features.append(tracker.extract_features(frame.rgb, index).to("cpu"))
        geometries.append(tracker.prepare_source_geometry(frame).to("cpu"))
        truths.append(gt_pose(frame, tracker.device))
        timestamps.append(frame.timestamp)
        print(f"cached frame {index + 1}/{length}", flush=True)
    modes = {}
    for mode in CAMERA_MODES:
        poses = [truths[0].detach().clone()]
        records = []
        for source_index in range(length - 1):
            source_pose = poses[-1]
            target_initial = source_pose.detach().clone()
            tracked = tracker.track_pair(
                features[source_index],
                features[source_index + 1],
                geometries[source_index],
                source_pose,
                target_initial,
                camera_mode=mode,
            )
            target_truth = truths[source_index + 1]
            pre_t, pre_r = pose_error(target_initial, target_truth)
            post_t, post_r = pose_error(tracked.T_rig_from_world_target, target_truth)
            poses.append(tracked.T_rig_from_world_target.detach().clone())
            conditions = [item.lm_result.hessian_condition_number for item in tracked.history]
            records.append(
                {
                    "source_frame": source_index,
                    "target_frame": source_index + 1,
                    "status": tracked.status,
                    "pre_translation_error_m": pre_t,
                    "pre_rotation_error_deg": pre_r,
                    "post_translation_error_m": post_t,
                    "post_rotation_error_deg": post_r,
                    "minimum_correspondences": min(item.correspondence_count for item in tracked.history),
                    "maximum_hessian_condition": max(conditions),
                }
            )
            print(f"sequence {mode} {source_index}->{source_index + 1}: {tracked.status}", flush=True)
        numeric = sequence_metrics(poses, truths)
        modes[mode] = {"poses": poses, "records": records, "numeric": numeric}

    truth_positions = modes["both"]["numeric"]["truth_positions"]
    zero_poses = [truths[0]] * length
    baseline = sequence_metrics(zero_poses, truths)
    plot_trajectory(output / "trajectory.png", modes, truth_positions)
    plot_sequence_errors(output / "trajectory_error.png", modes)
    plot_diagnostics(output / "sequence_diagnostics.png", modes)
    write_trajectory_csv(output / "trajectory.csv", modes, truths, timestamps)
    serializable = {
        "length": length,
        "frame_zero_gt_anchored": True,
        "intermediate_gt_reanchoring": False,
        "trajectory_position_definition": "inverse(T_rig_from_world)[:3,3]",
        "trajectory_alignment": "none",
        "zero_motion_baseline": {
            "ate_rmse_m": baseline["ate_rmse_m"],
            "translation_rpe_rmse_m": baseline["translation_rpe_rmse_m"],
            "rotation_rpe_rmse_deg": baseline["rotation_rpe_rmse_deg"],
        },
        "modes": {},
    }
    for mode, metrics in modes.items():
        numeric = metrics["numeric"]
        serializable["modes"][mode] = {
            "ate_rmse_m": numeric["ate_rmse_m"],
            "translation_rpe_rmse_m": numeric["translation_rpe_rmse_m"],
            "rotation_rpe_rmse_deg": numeric["rotation_rpe_rmse_deg"],
            "absolute_translation_error_m": distribution(numeric["absolute_translation_errors_m"]),
            "translation_rpe_m": distribution(numeric["translation_rpe_m"]),
            "rotation_rpe_deg": distribution(numeric["rotation_rpe_deg"]),
            "bad_status_count": sum(record["status"] not in ("converged", "max_iterations") for record in metrics["records"]),
            "records": metrics["records"],
        }
    return serializable


def acceptance(pair_results, sequence):
    checks = {}
    for pair, modes in pair_results.items():
        for mode, metrics in modes.items():
            checks[f"pair_{pair}_{mode}"] = (
                metrics["status"] == "converged"
                and metrics["finite_pose"]
                and metrics["final_translation_error_m"] < 0.005
                and metrics["final_rotation_error_deg"] < 0.05
                and all(item["lm"]["accepted_costs_monotonic"] for item in metrics["outer_history"])
            )
        for camera, metrics in modes["both"]["correspondence_feature_px"].items():
            checks[f"correspondence_{pair}_{camera}"] = (
                metrics["median"] < 0.5
                and metrics["p90"] < 1.0
                and metrics["inlier_fraction_below_1px"] > 0.90
            )
    baseline = sequence["zero_motion_baseline"]
    both = sequence["modes"]["both"]
    checks["sequence_no_bad_status"] = both["bad_status_count"] == 0
    checks["sequence_ate_beats_zero_motion"] = both["ate_rmse_m"] < baseline["ate_rmse_m"]
    checks["sequence_translation_rpe_reduced_50_percent"] = (
        both["translation_rpe_rmse_m"] < 0.5 * baseline["translation_rpe_rmse_m"]
    )
    checks["sequence_rotation_rpe_reduced_50_percent"] = (
        both["rotation_rpe_rmse_deg"] < 0.5 * baseline["rotation_rpe_rmse_deg"]
    )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "pretrained_models/droid.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "debug/droid_fisheye_motion")
    parser.add_argument("--sequence-length", type=int, default=300)
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--regenerate-trajectory-only",
        action="store_true",
        help="regenerate the 3-D trajectory from the existing trajectory.csv",
    )
    args = parser.parse_args()
    if args.regenerate_trajectory_only:
        sequence_output = args.output / "sequence"
        plot_trajectory_from_csv(
            sequence_output / "trajectory.csv", sequence_output / "trajectory.png"
        )
        print(f"regenerated {sequence_output / 'trajectory.png'}")
        return 0
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "oracle_information_boundary": {
            "source_gt_range": "used",
            "target_gt_pose_in_tracker": False,
            "target_gt_range": "unused",
            "target_gt_pose": "external evaluation only",
        },
    }
    write_report(report_path, report)
    try:
        if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
            raise RuntimeError("CUDA is required by AltCorrBlock")
        dataset = StereoFisheyeDataset(args.data_root)
        if args.sequence_length < 2 or args.sequence_length > len(dataset):
            raise ValueError("sequence length must be between 2 and the dataset length")
        cameras = make_cameras(dataset)
        native_reprojector = FisheyeRigReprojector(
            cameras, torch.from_numpy(dataset.T_rig_from_camera).float()
        )
        network = load_pretrained_droid(args.checkpoint, args.device)
        solver = OracleMotionOnlyConfig(
            camera_weighting="balanced",
            huber_threshold_px=1.0,
            invalid_residual_penalty_px=20.0,
            minimum_total_observations=2000,
            minimum_observations_per_camera=500,
            maximum_candidate_invalid_fraction=0.01,
            maximum_iterations=20,
        )
        config = DroidFisheyeMotionConfig(
            outer_iterations=args.outer_iterations,
            solver=solver,
        )
        tracker = DroidFisheyeMotionTracker(
            native_reprojector,
            GroundTruthRangeProvider(),
            network,
            config,
            device=args.device,
        )
        report["config"] = {
            **asdict(config),
            "checkpoint": str(args.checkpoint),
            "device": str(args.device),
            "native_resolution": list(dataset.image_size[0]),
            "feature_resolution": [config.feature_width, config.feature_height],
        }
        report["pair_validation"] = pair_validation(dataset, tracker, args.output)
        write_report(report_path, report)
        report["sequence"] = sequence_validation(
            dataset, tracker, args.output / "sequence", args.sequence_length
        )
        report["acceptance"] = acceptance(report["pair_validation"], report["sequence"])
        report["status"] = "passed" if all(report["acceptance"].values()) else "failed"
        write_report(report_path, report)
        print(json.dumps({"status": report["status"], "acceptance": report["acceptance"]}, indent=2))
        return 0 if report["status"] == "passed" else 1
    except Exception as error:
        report["status"] = "error"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        write_report(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
