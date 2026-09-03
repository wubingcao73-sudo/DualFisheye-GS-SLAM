import tempfile
import unittest
from pathlib import Path

import numpy as np

from hislam2.data.stereo_fisheye_dataset import (
    StereoFisheyeDataset,
    load_ds_calibration,
    load_extrinsics,
)


class StereoFisheyeCalibrationTest(unittest.TestCase):
    def test_ds_calibration_accepts_ascii_commas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cam0.txt"
            path.write_text(
                "-0.12, 0.56, 681.5, 682.1, 1445.7, 1444.3, 2880.0, 2880.0\n",
                encoding="utf-8",
            )
            calibration = load_ds_calibration(path)
        self.assertEqual(calibration.shape, (8,))
        np.testing.assert_allclose(calibration[-2:], [2880.0, 2880.0])

    def test_ds_calibration_rejects_wrong_value_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cam0.txt"
            path.write_text("0 0.5 1 1 1 1 10\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 8"):
                load_ds_calibration(path)

    def test_ds_calibration_rejects_chinese_comma(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cam0.txt"
            path.write_text(
                "-0.12, 0.56, 681.5, 682.1, 1445.7, 1444.3, 2880.0，2880.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-numeric"):
                load_ds_calibration(path)

    def test_extrinsics_are_parsed_by_explicit_names(self):
        identity = "\n".join(
            ["1 0 0 0", "0 1 0 0", "0 0 1 0", "0 0 0 1"]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extrinsics.txt"
            path.write_text(
                "# T_rig_from_cam0\n"
                + identity
                + "\n# T_rig_from_cam1\n"
                + identity
                + "\n# T_cam1_from_cam0\n"
                + identity
                + "\n",
                encoding="utf-8",
            )
            matrices = load_extrinsics(path)
        self.assertEqual(
            set(matrices),
            {"T_rig_from_cam0", "T_rig_from_cam1", "T_cam1_from_cam0"},
        )
        np.testing.assert_allclose(matrices["T_rig_from_cam0"], np.eye(4))

    def test_dataset_converts_blender_camera_axes_to_ds_axes(self):
        dataset = StereoFisheyeDataset()
        ds_forward = np.array([0.0, 0.0, 1.0])
        front_in_rig = dataset.T_rig_from_camera[0, :3, :3] @ ds_forward
        back_in_rig = dataset.T_rig_from_camera[1, :3, :3] @ ds_forward
        np.testing.assert_allclose(front_in_rig, [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(back_in_rig, [0.0, -1.0, 0.0], atol=1e-6)
        self.assertLess(dataset.pose_composition_max_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
