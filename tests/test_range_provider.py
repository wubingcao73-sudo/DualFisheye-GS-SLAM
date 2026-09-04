import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.data import StereoFisheyeDataset
from hislam2.range import GroundTruthRangeProvider, RangeObservation


class GroundTruthRangeProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = GroundTruthRangeProvider()

    def test_classifies_synthetic_observations(self):
        range_m = torch.tensor(
            [
                [1.0, 2.5, 0.0, -1.0],
                [float("nan"), float("inf"), 1.0e10, 9.0e9],
            ],
            dtype=torch.float32,
        )
        observation = self.provider.from_range(range_m)
        expected = torch.tensor(
            [[True, True, False, False], [False, False, False, True]]
        )
        self.assertTrue(torch.equal(observation.observation_valid, expected))
        self.assertTrue(
            torch.allclose(
                observation.inverse_range[expected], range_m[expected].reciprocal()
            )
        )
        self.assertTrue(bool((observation.inverse_range[~expected] == 0.0).all()))
        self.assertTrue(bool((observation.confidence[expected] == 1.0).all()))
        self.assertTrue(bool((observation.confidence[~expected] == 0.0).all()))
        self.assertIs(observation.range_m, range_m)

    def test_provide_reads_raw_gt_range_from_frame(self):
        range_m = torch.tensor([[1.0, 4.0], [2.0, 8.0]], dtype=torch.float64)
        observation = self.provider.provide(SimpleNamespace(gt_range=range_m))
        self.assertIsInstance(observation, RangeObservation)
        self.assertEqual(observation.range_m.dtype, torch.float64)
        self.assertTrue(bool(observation.observation_valid.all()))
        self.assertTrue(
            torch.allclose(
                observation.range_m * observation.inverse_range,
                torch.ones_like(range_m),
            )
        )

    def test_rejects_invalid_inputs_and_configuration(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            self.provider.from_range([[1.0]])
        with self.assertRaisesRegex(TypeError, "float32 or float64"):
            self.provider.from_range(torch.ones((2, 2), dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
            self.provider.from_range(torch.ones(2, dtype=torch.float32))
        with self.assertRaisesRegex(TypeError, "gt_range"):
            self.provider.provide(SimpleNamespace())
        with self.assertRaises(ValueError):
            GroundTruthRangeProvider(invalid_sentinel=-1.0)
        with self.assertRaises(ValueError):
            GroundTruthRangeProvider(sentinel_relative_margin=1.0)

    def test_real_stereo_frame_shapes_and_identity(self):
        frame = StereoFisheyeDataset()[0]
        self.assertFalse(hasattr(frame, "range_observation_valid"))
        observation = self.provider.provide(frame)
        self.assertEqual(tuple(observation.range_m.shape), (2, 2880, 2880))
        self.assertEqual(observation.range_m.dtype, torch.float32)
        self.assertEqual(observation.inverse_range.dtype, torch.float32)
        self.assertEqual(observation.observation_valid.dtype, torch.bool)
        self.assertEqual(observation.confidence.dtype, torch.float32)
        self.assertEqual(observation.range_m.device.type, "cpu")
        valid = observation.observation_valid
        identity_error = torch.abs(
            observation.range_m[valid] * observation.inverse_range[valid] - 1.0
        )
        self.assertLess(float(identity_error.max()), 1e-6)
        self.assertTrue(bool(valid[0].any()))
        self.assertTrue(bool(valid[1].any()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_preserves_cuda_device(self):
        range_m = torch.tensor([[1.0, 0.0], [2.0, 4.0]], device="cuda")
        observation = self.provider.from_range(range_m)
        self.assertEqual(observation.range_m.device.type, "cuda")
        self.assertEqual(observation.inverse_range.device.type, "cuda")
        self.assertEqual(observation.observation_valid.device.type, "cuda")
        self.assertEqual(observation.confidence.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
