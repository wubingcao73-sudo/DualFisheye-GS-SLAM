import argparse
import json
import os
import sys
from pathlib import Path


sys.path.append(os.path.join(os.path.dirname(__file__), "hislam2"))

from data.stereo_fisheye_dataset import DEFAULT_DATA_ROOT, validate_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HI-SLAM2 stereo fisheye entry point")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"stereo fisheye dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/classroom_stereo_fisheye.yaml"),
        help="algorithm and stereo input configuration",
    )
    parser.add_argument(
        "--validate-input",
        action="store_true",
        help="validate stereo RGB/range/calibration/poses without loading SLAM",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the input validation report as JSON",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.validate_input:
        parser.error(
            "the stereo Double Sphere input contract is implemented, but tracking and "
            "mapping are not DS-aware yet; run with --validate-input for this milestone"
        )

    if not args.config.is_file():
        parser.error(f"configuration file does not exist: {args.config}")

    report = validate_dataset(args.data_root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Validated {report['frame_count']} synchronized stereo frames")
        print(f"Data root: {report['data_root']}")
        print(f"Camera model: {report['camera_model']}")
        print(f"Native sizes (width, height): {report['image_size']}")
        print(f"Baseline: {report['baseline_m']:.9f} m")
        print(f"Pose composition max error: {report['pose_composition_max_error']:.3e}")
        for sample in report["samples"]:
            print(
                f"Frame {sample['index']:06d}: rgb={sample['rgb_shape']}, "
                f"range={sample['range_shape']}, "
                f"valid={sample['range_valid_ratio']}, "
                f"preview_corr={sample['range_preview_correlation']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
