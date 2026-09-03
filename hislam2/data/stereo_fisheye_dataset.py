"""Native-resolution reader for the classroom stereo Double Sphere dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch

from .frame_types import StereoFisheyeFrame


DEFAULT_DATA_ROOT = Path(
    "/media/nonchalance/data/Google_Downloads/sim/sim_data/classroom"
)

_CAMERAS = ("cam0", "cam1")
_RGB_PATTERN = re.compile(r"^(\d{6})RGB\.png$")
_RANGE_PATTERN = re.compile(r"^(\d{6})Depth\.exr$")
_DEPTH_VIS_PATTERN = re.compile(r"^(\d{6})DepthVis\.png$")
_INVALID_RANGE_SENTINEL = 1.0e10
_T_BLENDER_CAMERA_FROM_DS_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def _numeric_tokens(path: Path) -> List[float]:
    text = "\n".join(
        line.split("#", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    values = [token for token in re.split(r"[\s,]+", text.strip()) if token]
    try:
        return [float(value) for value in values]
    except ValueError as error:
        raise ValueError(f"{path} contains a non-numeric calibration value") from error


def load_ds_calibration(path: Path) -> np.ndarray:
    """Read ``xi alpha fx fy cx cy width height`` from one text file."""
    values = _numeric_tokens(path)
    if len(values) != 8:
        raise ValueError(f"{path} must contain exactly 8 values, found {len(values)}")

    calibration = np.asarray(values, dtype=np.float64)
    xi, alpha, fx, fy, cx, cy, width, height = calibration
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{path}: DS alpha must be in (0, 1), got {alpha}")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"{path}: fx and fy must be positive")
    if width <= 0.0 or height <= 0.0 or not width.is_integer() or not height.is_integer():
        raise ValueError(f"{path}: width and height must be positive integers")
    if not 0.0 <= cx < width or not 0.0 <= cy < height:
        raise ValueError(f"{path}: principal point lies outside the calibrated image")
    return calibration


def load_extrinsics(path: Path) -> Dict[str, np.ndarray]:
    """Read named 4x4 matrices from the dataset extrinsics file."""
    matrices: Dict[str, np.ndarray] = {}
    current_name = None
    rows: List[List[float]] = []

    def finish_matrix() -> None:
        nonlocal rows
        if current_name is None:
            return
        if len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ValueError(f"{path}: {current_name} is not a 4x4 matrix")
        matrices[current_name] = np.asarray(rows, dtype=np.float64)
        rows = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# T_"):
            finish_matrix()
            current_name = line[2:].strip()
            continue
        if line.startswith("#"):
            continue
        if current_name is None:
            raise ValueError(f"{path}: matrix data appears before a named T_* heading")
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError as error:
            raise ValueError(f"{path}: invalid value in {current_name}") from error
    finish_matrix()

    required = {"T_rig_from_cam0", "T_rig_from_cam1", "T_cam1_from_cam0"}
    missing = required.difference(matrices)
    if missing:
        raise ValueError(f"{path}: missing matrices: {', '.join(sorted(missing))}")
    for name, matrix in matrices.items():
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"{path}: {name} has an invalid homogeneous bottom row")
    return matrices


def _indexed_files(directory: Path, pattern: re.Pattern[str]) -> Dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"required directory does not exist: {directory}")
    indexed: Dict[int, Path] = {}
    unexpected = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            unexpected.append(path.name)
            continue
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"duplicate frame index {index} in {directory}")
        indexed[index] = path
    if unexpected:
        raise ValueError(f"unexpected files in {directory}: {', '.join(unexpected[:5])}")
    return indexed


def _load_pose_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.loadtxt(path, comments="#", dtype=np.float64)
    table = np.atleast_2d(table)
    if table.shape[1] != 19:
        raise ValueError(f"{path} must have 19 columns, found {table.shape[1]}")
    indices = table[:, 0].astype(np.int64)
    if not np.array_equal(table[:, 0], indices):
        raise ValueError(f"{path} contains non-integral indices")
    poses = table[:, 3:].reshape(-1, 4, 4)
    if not np.allclose(poses[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{path} contains invalid homogeneous transforms")
    return indices, table[:, 2], poses


def _read_exr_v(path: Path) -> np.ndarray:
    """Read Blender's single-channel ``V`` OpenEXR output as float32."""
    try:
        import OpenEXR
    except ImportError as error:
        raise RuntimeError(
            "OpenEXR is required to read GT range; install the project requirements"
        ) from error

    if hasattr(OpenEXR, "File"):
        exr = OpenEXR.File(str(path))
        try:
            channels = exr.channels()
            if "V" not in channels:
                raise ValueError(f"{path} does not contain the required V channel")
            pixels = getattr(channels["V"], "pixels", channels["V"])
            image = np.asarray(pixels, dtype=np.float32)
        finally:
            close = getattr(exr, "close", None)
            if close is not None:
                close()
    else:
        try:
            import Imath
        except ImportError as error:
            raise RuntimeError("the installed OpenEXR package is missing Imath") from error
        exr = OpenEXR.InputFile(str(path))
        try:
            header = exr.header()
            if "V" not in header["channels"]:
                raise ValueError(f"{path} does not contain the required V channel")
            data_window = header["dataWindow"]
            height = data_window.max.y - data_window.min.y + 1
            width = data_window.max.x - data_window.min.x + 1
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            image = np.frombuffer(exr.channel("V", pixel_type), dtype=np.float32)
            image = image.reshape(height, width)
        finally:
            exr.close()

    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D V channel, got shape {image.shape}")
    return np.ascontiguousarray(image, dtype=np.float32)


class StereoFisheyeDataset(Sequence[StereoFisheyeFrame]):
    """Strict reader for synchronized native-resolution stereo fisheye frames."""

    camera_model = "double_sphere"

    def __init__(self, data_root: str | Path = DEFAULT_DATA_ROOT):
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")

        calibration_dir = self.root / "calibration"
        calibrations = np.stack(
            [load_ds_calibration(calibration_dir / f"{camera}.txt") for camera in _CAMERAS]
        )
        self.camera_params = calibrations[:, :6]
        self.image_size = calibrations[:, [6, 7]].astype(np.int64)
        extrinsics = load_extrinsics(calibration_dir / "extrinsics.txt")
        self.T_rig_from_blender_camera = np.stack(
            [extrinsics[f"T_rig_from_{camera}"] for camera in _CAMERAS]
        )
        self.T_rig_from_camera = (
            self.T_rig_from_blender_camera @ _T_BLENDER_CAMERA_FROM_DS_CAMERA
        )
        expected_raw_cam1_from_cam0 = (
            np.linalg.inv(self.T_rig_from_blender_camera[1])
            @ self.T_rig_from_blender_camera[0]
        )
        if not np.allclose(
            expected_raw_cam1_from_cam0, extrinsics["T_cam1_from_cam0"], atol=1e-6
        ):
            raise ValueError("extrinsics are inconsistent with T_cam1_from_cam0")

        self.rgb_files = [
            _indexed_files(self.root / camera / "rgb", _RGB_PATTERN) for camera in _CAMERAS
        ]
        self.range_files = [
            _indexed_files(self.root / camera / "depth_exr", _RANGE_PATTERN)
            for camera in _CAMERAS
        ]
        self.depth_vis_files = [
            _indexed_files(self.root / camera / "depth_png", _DEPTH_VIS_PATTERN)
            for camera in _CAMERAS
        ]

        timestamp_table = np.loadtxt(self.root / "timestamps.txt", comments="#", dtype=np.float64)
        timestamp_table = np.atleast_2d(timestamp_table)
        if timestamp_table.shape[1] != 3:
            raise ValueError("timestamps.txt must contain index, frame, timestamp")
        self.indices = timestamp_table[:, 0].astype(np.int64)
        self.frame_numbers = timestamp_table[:, 1].astype(np.int64)
        self.timestamps = timestamp_table[:, 2]
        if not np.array_equal(timestamp_table[:, :2], timestamp_table[:, :2].astype(np.int64)):
            raise ValueError("timestamps.txt contains non-integral indices or frame numbers")
        if not np.array_equal(self.indices, np.arange(len(self.indices))):
            raise ValueError("timestamps.txt indices must be contiguous and start at zero")
        if np.any(np.diff(self.timestamps) <= 0.0):
            raise ValueError("timestamps must be strictly increasing")

        expected_indices = set(self.indices.tolist())
        for camera_index, camera in enumerate(_CAMERAS):
            for kind, files in (
                ("RGB", self.rgb_files[camera_index]),
                ("EXR range", self.range_files[camera_index]),
                ("depth preview", self.depth_vis_files[camera_index]),
            ):
                actual_indices = set(files)
                if actual_indices != expected_indices:
                    missing = sorted(expected_indices - actual_indices)
                    extra = sorted(actual_indices - expected_indices)
                    raise ValueError(
                        f"{camera} {kind} indices do not match timestamps; "
                        f"missing={missing[:5]}, extra={extra[:5]}"
                    )

        pose_tables = {}
        for name in ("rig", "cam0", "cam1"):
            indices, timestamps, poses = _load_pose_table(self.root / "poses" / f"{name}.txt")
            if not np.array_equal(indices, self.indices):
                raise ValueError(f"poses/{name}.txt indices do not match timestamps.txt")
            if not np.allclose(timestamps, self.timestamps, atol=1e-9):
                raise ValueError(f"poses/{name}.txt timestamps do not match timestamps.txt")
            pose_tables[name] = poses
        self.gt_T_world_from_rig = pose_tables["rig"]
        self.gt_T_world_from_blender_camera = np.stack(
            [pose_tables["cam0"], pose_tables["cam1"]], axis=1
        )
        self.gt_T_world_from_camera = (
            self.gt_T_world_from_blender_camera
            @ _T_BLENDER_CAMERA_FROM_DS_CAMERA
        )

        composed = self.gt_T_world_from_rig[:, None] @ self.T_rig_from_camera[None]
        self.pose_composition_max_error = float(
            np.max(np.abs(composed - self.gt_T_world_from_camera))
        )
        if self.pose_composition_max_error >= 1e-6:
            raise ValueError(
                "camera poses are inconsistent with rig poses and extrinsics: "
                f"max error={self.pose_composition_max_error:.3e}"
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> StereoFisheyeFrame:
        if item < 0:
            item += len(self)
        if item < 0 or item >= len(self):
            raise IndexError(item)
        index = int(self.indices[item])

        rgb_images = []
        ranges = []
        for camera_index, camera in enumerate(_CAMERAS):
            bgr = cv2.imread(str(self.rgb_files[camera_index][index]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"failed to read {camera} RGB frame {index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt_range = _read_exr_v(self.range_files[camera_index][index])
            expected_width, expected_height = self.image_size[camera_index]
            expected_shape = (int(expected_height), int(expected_width))
            if rgb.shape[:2] != expected_shape:
                raise ValueError(
                    f"{camera} RGB frame {index} has shape {rgb.shape[:2]}, "
                    f"calibration expects {expected_shape}"
                )
            if gt_range.shape != expected_shape:
                raise ValueError(
                    f"{camera} range frame {index} has shape {gt_range.shape}, "
                    f"calibration expects {expected_shape}"
                )
            rgb_images.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
            ranges.append(torch.from_numpy(gt_range))

        rgb_tensor = torch.stack(rgb_images)
        range_tensor = torch.stack(ranges)
        observation_valid = (
            torch.isfinite(range_tensor)
            & (range_tensor > 0.0)
            & (range_tensor < _INVALID_RANGE_SENTINEL * (1.0 - 1.0e-6))
        )
        return StereoFisheyeFrame(
            index=index,
            frame_number=int(self.frame_numbers[item]),
            timestamp=float(self.timestamps[item]),
            rgb=rgb_tensor,
            camera_model=self.camera_model,
            camera_params=torch.from_numpy(self.camera_params.astype(np.float32)),
            image_size=torch.from_numpy(self.image_size.copy()),
            T_rig_from_camera=torch.from_numpy(self.T_rig_from_camera.copy()),
            gt_T_world_from_rig=torch.from_numpy(self.gt_T_world_from_rig[item].copy()),
            gt_range=range_tensor,
            range_observation_valid=observation_valid,
        )

    @property
    def baseline_m(self) -> float:
        T_cam1_from_cam0 = np.linalg.inv(self.T_rig_from_camera[1]) @ self.T_rig_from_camera[0]
        return float(np.linalg.norm(T_cam1_from_cam0[:3, 3]))


def _preview_range_correlation(
    dataset: StereoFisheyeDataset, frame: StereoFisheyeFrame
) -> List[float]:
    correlations = []
    for camera_index in range(2):
        preview = cv2.imread(
            str(dataset.depth_vis_files[camera_index][frame.index]), cv2.IMREAD_GRAYSCALE
        )
        if preview is None:
            raise ValueError(f"failed to read depth preview for frame {frame.index}")
        gt_range = frame.gt_range[camera_index].numpy()
        valid = frame.range_observation_valid[camera_index].numpy()
        stride = max(1, int(np.sqrt(gt_range.size / 100_000)))
        sampled_range = gt_range[::stride, ::stride][valid[::stride, ::stride]]
        sampled_preview = preview[::stride, ::stride][valid[::stride, ::stride]]
        if sampled_range.size < 2 or np.std(sampled_range) == 0 or np.std(sampled_preview) == 0:
            raise ValueError("range/preview correlation cannot be computed")
        correlation = float(np.corrcoef(sampled_range, sampled_preview)[0, 1])
        if abs(correlation) < 0.8:
            raise ValueError(
                f"range and depth preview are inconsistent: correlation={correlation:.3f}"
            )
        correlations.append(correlation)
    return correlations


def validate_dataset(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    sample_indices: Iterable[int] | None = None,
) -> dict:
    """Validate the manifest and decode representative native-resolution frames."""
    dataset = StereoFisheyeDataset(data_root)
    if sample_indices is None:
        sample_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})

    samples = []
    for item in sample_indices:
        frame = dataset[item]
        valid_ratios = frame.range_observation_valid.float().mean(dim=(1, 2)).tolist()
        if not frame.range_observation_valid.any():
            raise ValueError(f"frame {frame.index} has no valid range pixels")
        samples.append(
            {
                "index": frame.index,
                "frame_number": frame.frame_number,
                "timestamp": frame.timestamp,
                "rgb_shape": list(frame.rgb.shape),
                "range_shape": list(frame.gt_range.shape),
                "range_valid_ratio": valid_ratios,
                "range_preview_correlation": _preview_range_correlation(dataset, frame),
            }
        )

    report = {
        "data_root": str(dataset.root),
        "camera_model": dataset.camera_model,
        "frame_count": len(dataset),
        "image_size": dataset.image_size.tolist(),
        "camera_params": dataset.camera_params.tolist(),
        "baseline_m": dataset.baseline_m,
        "pose_composition_max_error": dataset.pose_composition_max_error,
        "timestamp_step_seconds": np.diff(dataset.timestamps).tolist()[:5],
        "samples": samples,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = validate_dataset(args.data_root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Validated {report['frame_count']} stereo frames at {report['data_root']}")
        print(f"Camera model: {report['camera_model']}")
        print(f"Native image size: {report['image_size']}")
        print(f"Baseline: {report['baseline_m']:.9f} m")
        print(f"Pose composition max error: {report['pose_composition_max_error']:.3e}")
        for sample in report["samples"]:
            print(
                f"frame {sample['index']:06d}: rgb={sample['rgb_shape']} "
                f"range={sample['range_shape']} valid={sample['range_valid_ratio']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
