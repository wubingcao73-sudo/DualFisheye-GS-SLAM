#!/usr/bin/env python3
"""Numerically and visually validate the GT Euclidean range provider."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.data import DEFAULT_DATA_ROOT, StereoFisheyeDataset  # noqa: E402
from hislam2.range import GroundTruthRangeProvider  # noqa: E402


CAMERA_LABELS = ("front", "back")


def sampled_distribution(values: torch.Tensor, maximum_samples: int = 250_000) -> dict:
    values = values.detach().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    stride = max(1, values.numel() // maximum_samples)
    sample = values[::stride].double().cpu()
    return {
        "sample_count": len(sample),
        "min": float(values.min()),
        "mean": float(sample.mean()),
        "median": float(torch.quantile(sample, 0.50)),
        "p01": float(torch.quantile(sample, 0.01)),
        "p95": float(torch.quantile(sample, 0.95)),
        "p99": float(torch.quantile(sample, 0.99)),
        "max": float(values.max()),
    }


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"failed to write image: {path}")


def colorize_scalar(
    values: np.ndarray, valid: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    normalized = np.zeros(values.shape, dtype=np.uint8)
    if upper <= lower:
        raise ValueError(f"invalid visualization range [{lower}, {upper}]")
    normalized[valid] = np.clip(
        (values[valid] - lower) / (upper - lower) * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def labeled_preview(image: np.ndarray, title: str, size: int = 720) -> np.ndarray:
    preview = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(preview, (0, 0), (size, 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        preview,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def create_visualization(
    frame,
    observation,
    camera_index: int,
    output_directory: Path,
) -> dict:
    label = CAMERA_LABELS[camera_index]
    prefix = f"{label}_frame_{frame.index:04d}"
    rgb = frame.rgb[camera_index].permute(1, 2, 0).numpy()
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    range_m = observation.range_m[camera_index].numpy()
    inverse_range = observation.inverse_range[camera_index].numpy()
    valid = observation.observation_valid[camera_index].numpy()

    valid_range = range_m[valid]
    valid_inverse = inverse_range[valid]
    range_limits = np.percentile(valid_range[:: max(1, len(valid_range) // 250_000)], [1, 99])
    inverse_limits = np.percentile(
        valid_inverse[:: max(1, len(valid_inverse) // 250_000)], [1, 99]
    )
    range_color = colorize_scalar(range_m, valid, *range_limits)
    inverse_color = colorize_scalar(inverse_range, valid, *inverse_limits)
    valid_image = valid.astype(np.uint8) * 255
    valid_bgr = cv2.cvtColor(valid_image, cv2.COLOR_GRAY2BGR)

    rgb_path = output_directory / f"{prefix}_rgb.png"
    range_path = output_directory / f"{prefix}_range.png"
    inverse_path = output_directory / f"{prefix}_inverse_range.png"
    valid_path = output_directory / f"{prefix}_valid_mask.png"
    panel_path = output_directory / f"{prefix}_panel.png"
    write_image(rgb_path, rgb_bgr)
    write_image(range_path, range_color)
    write_image(inverse_path, inverse_color)
    write_image(valid_path, valid_image)
    panel = np.concatenate(
        (
            labeled_preview(rgb_bgr, "RGB"),
            labeled_preview(range_color, "Euclidean range (p01-p99)"),
            labeled_preview(inverse_color, "Inverse range (p01-p99)"),
            labeled_preview(valid_bgr, "Observation valid"),
        ),
        axis=1,
    )
    write_image(panel_path, panel)
    return {
        "rgb": str(rgb_path),
        "range": str(range_path),
        "inverse_range": str(inverse_path),
        "valid_mask": str(valid_path),
        "panel": str(panel_path),
        "range_color_limits_m": [float(value) for value in range_limits],
        "inverse_range_color_limits_per_m": [float(value) for value in inverse_limits],
        "invalid_pixels_are_black": True,
    }


def camera_metrics(
    raw_range: torch.Tensor,
    observation_valid: torch.Tensor,
    inverse_range: torch.Tensor,
    confidence: torch.Tensor,
) -> dict:
    finite = torch.isfinite(raw_range)
    nonpositive = finite & (raw_range <= 0.0)
    sentinel = finite & (raw_range >= 1.0e10 * (1.0 - 1.0e-6))
    valid_range = raw_range[observation_valid]
    valid_inverse = inverse_range[observation_valid]
    identity_error = torch.abs(valid_range * valid_inverse - 1.0)
    result = {
        "shape": list(raw_range.shape),
        "dtype": str(raw_range.dtype),
        "device": str(raw_range.device),
        "pixel_count": raw_range.numel(),
        "valid_pixel_count": int(observation_valid.sum()),
        "valid_ratio": float(observation_valid.float().mean()),
        "invalid_nonfinite_count": int((~finite).sum()),
        "invalid_nonpositive_count": int(nonpositive.sum()),
        "invalid_sentinel_count": int(sentinel.sum()),
        "valid_range_m": sampled_distribution(valid_range),
        "valid_inverse_range_per_m": sampled_distribution(valid_inverse),
        "range_times_inverse_range_max_error": float(identity_error.max()),
        "invalid_inverse_range_zero": bool((inverse_range[~observation_valid] == 0).all()),
        "valid_confidence_one": bool((confidence[observation_valid] == 1).all()),
        "invalid_confidence_zero": bool((confidence[~observation_valid] == 0).all()),
    }
    if result["range_times_inverse_range_max_error"] >= 1e-6:
        raise AssertionError("range * inverse_range identity error exceeds 1e-6")
    if not result["invalid_inverse_range_zero"]:
        raise AssertionError("invalid inverse-range values are not zero")
    if not result["valid_confidence_one"] or not result["invalid_confidence_zero"]:
        raise AssertionError("GT confidence does not match observation validity")
    return result


def synthetic_metrics(provider: GroundTruthRangeProvider) -> dict:
    raw = torch.tensor(
        [[1.0, 2.0, 0.0, -1.0, float("nan"), float("inf"), 1.0e10, 9.0e9]],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [[True, True, False, False, False, False, False, True]]
    )
    observation = provider.from_range(raw)
    mask_equal = bool(torch.equal(observation.observation_valid, expected))
    result = {
        "mask_equal": mask_equal,
        "actual_valid": observation.observation_valid.tolist(),
        "expected_valid": expected.tolist(),
    }
    if not mask_equal:
        raise AssertionError(f"synthetic validity classification failed: {result}")
    return result


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("debug/range_provider"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.output / "report.json"
    report = {
        "status": "running",
        "data_root": str(args.data_root),
        "range_semantics": "Euclidean distance from camera center in meters",
        "invalid_sentinel": 1.0e10,
        "sentinel_relative_margin": 1.0e-6,
    }
    try:
        provider = GroundTruthRangeProvider()
        report["synthetic"] = synthetic_metrics(provider)
        dataset = StereoFisheyeDataset(args.data_root)
        frame = dataset[args.frame_index]
        observation = provider.provide(frame)
        report["frame"] = {
            "index": frame.index,
            "frame_number": frame.frame_number,
            "timestamp": frame.timestamp,
        }
        for camera_index, label in enumerate(CAMERA_LABELS):
            report[label] = camera_metrics(
                observation.range_m[camera_index],
                observation.observation_valid[camera_index],
                observation.inverse_range[camera_index],
                observation.confidence[camera_index],
            )
            if not args.skip_visualization:
                report[label]["visualization"] = create_visualization(
                    frame,
                    observation,
                    camera_index,
                    args.output / "visualization",
                )
            write_report(report_path, report)
        report["status"] = "passed"
        write_report(report_path, report)
        print("GT RangeProvider validation: passed")
        print(f"Report: {report_path}")
        if not args.skip_visualization:
            print(f"Visualization: {args.output / 'visualization'}")
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        write_report(report_path, report)
        print(f"GT RangeProvider validation failed: {error}", file=sys.stderr)
        print(f"Partial report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
