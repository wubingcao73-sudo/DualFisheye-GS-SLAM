import math
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera
from hislam2.data import DEFAULT_DATA_ROOT, StereoFisheyeDataset
from hislam2.data.frame_types import StereoFisheyeFrame
from hislam2.geom.fisheye_reprojection import FisheyeRigReprojector
from hislam2.range import GroundTruthRangeProvider
from hislam2.tracking.droid_fisheye_motion import (
    DroidFisheyeMotionConfig,
    DroidFisheyeMotionTracker,
    DroidFrameFeatures,
    build_feature_camera,
    feature_to_native_pixels,
    feature_to_processed_pixels,
    load_pretrained_droid,
)
from hislam2.tracking.oracle_motion_only import OracleMotionOnlyConfig, bilinear_sample_range


class _ZeroCorr:
    def __init__(self, feature_maps, num_levels, radius):
        self.shape = feature_maps.shape
        self.channels = num_levels * (2 * radius + 1) ** 2

    def __call__(self, coordinates, ii, jj):
        batch, edges, height, width, _ = coordinates.shape
        return coordinates.new_zeros(batch, edges, self.channels, height, width)


class _ZeroUpdate(torch.nn.Module):
    def extract_features(self, images):
        batch, cameras, _, height, width = images.shape
        shape = (batch, cameras, 128, height // 8, width // 8)
        zeros = images.new_zeros(shape)
        return zeros, zeros, zeros

    def update(self, net, context, correlation, motion):
        shape = (net.shape[0], net.shape[1], net.shape[-2], net.shape[-1], 2)
        return net, net.new_zeros(shape), net.new_ones(shape)


def _identity_frame(camera, size=64):
    identity = torch.eye(4, dtype=torch.float64)
    return StereoFisheyeFrame(
        index=0,
        frame_number=0,
        timestamp=0.0,
        rgb=torch.zeros((2, 3, size, size), dtype=torch.uint8),
        camera_model="double_sphere",
        camera_params=torch.tensor([camera.parameters, camera.parameters], dtype=torch.float32),
        image_size=torch.tensor([[size, size], [size, size]], dtype=torch.int64),
        T_rig_from_camera=torch.stack((identity, identity)),
        gt_T_world_from_rig=identity,
        gt_range=torch.full((2, size, size), 4.0, dtype=torch.float32),
    )


class FeatureCoordinateTest(unittest.TestCase):
    def setUp(self):
        self.camera = DoubleSphereCamera(-0.12, 0.56, 681.5, 682.1, 1445.7, 1444.3, 2880, 2880)
        self.feature_camera = build_feature_camera(self.camera, 720, 720, 8)

    def test_feature_calibration_matches_exact_half_pixel_mapping(self):
        self.assertAlmostEqual(self.feature_camera.fx, 0.25 * self.camera.fx / 8.0)
        self.assertAlmostEqual(
            self.feature_camera.cx, (0.25 * (self.camera.cx + 0.5) - 0.5) / 8.0
        )
        feature_pixels = torch.tensor(
            [[0.0, 0.0], [12.0, 31.0], [44.5, 50.25], [89.0, 89.0]],
            dtype=torch.float64,
        )
        native_pixels = feature_to_native_pixels(feature_pixels, self.camera, 720, 720, 8)
        expected = (8.0 * feature_pixels + 0.5) / 0.25 - 0.5
        torch.testing.assert_close(native_pixels, expected)
        feature_rays, feature_valid = self.feature_camera.unproject(feature_pixels)
        native_rays, native_valid = self.camera.unproject(native_pixels)
        self.assertTrue(bool((feature_valid == native_valid).all()))
        torch.testing.assert_close(feature_rays, native_rays, atol=2e-12, rtol=2e-12)

    def test_feature_to_processed_pixel_centers(self):
        feature_pixels = torch.tensor([[0.0, 0.0], [89.0, 89.0]])
        expected = torch.tensor([[0.5, 0.5], [712.5, 712.5]])
        torch.testing.assert_close(feature_to_processed_pixels(feature_pixels), expected)

    def test_native_integer_sampling_is_identity(self):
        height, width = 7, 9
        image = torch.arange(height * width, dtype=torch.float64).reshape(height, width)
        valid = torch.ones_like(image, dtype=torch.bool)
        rows, columns = torch.meshgrid(
            torch.arange(height - 1, dtype=torch.float64),
            torch.arange(width - 1, dtype=torch.float64),
            indexing="ij",
        )
        pixels = torch.stack((columns.flatten(), rows.flatten()), dim=-1)
        sampled, sampled_valid = bilinear_sample_range(image + 1.0, valid, pixels)
        self.assertTrue(bool(sampled_valid.all()))
        torch.testing.assert_close(
            sampled,
            (image + 1.0)[rows.long(), columns.long()].flatten(),
            atol=8.0 * torch.finfo(torch.float64).eps,
            rtol=8.0 * torch.finfo(torch.float64).eps,
        )


class DroidFisheyeTrackerTest(unittest.TestCase):
    def setUp(self):
        camera = DoubleSphereCamera(0.0, 0.55, 24.0, 24.0, 31.5, 31.5, 64, 64)
        self.frame = _identity_frame(camera)
        reprojector = FisheyeRigReprojector(
            (camera, camera), torch.stack((torch.eye(4), torch.eye(4))).float()
        )
        solver = OracleMotionOnlyConfig(
            camera_weighting="balanced",
            huber_threshold_px=1.0,
            invalid_residual_penalty_px=20.0,
            minimum_total_observations=2,
            minimum_observations_per_camera=1,
            maximum_iterations=3,
        )
        config = DroidFisheyeMotionConfig(
            processed_height=64,
            processed_width=64,
            outer_iterations=2,
            target_safety_margin_feature_px=0.0,
            use_amp=False,
            solver=solver,
        )
        self.tracker = DroidFisheyeMotionTracker(
            reprojector,
            GroundTruthRangeProvider(),
            _ZeroUpdate(),
            config,
            corr_block_factory=_ZeroCorr,
            device="cpu",
        )

    def test_pose_free_features_and_source_geometry_track_identity(self):
        features = self.tracker.extract_features(self.frame.rgb, self.frame.index)
        self.assertEqual(features.feature_maps.shape, (2, 128, 8, 8))
        self.assertFalse(hasattr(features, "gt_T_world_from_rig"))
        geometry = self.tracker.prepare_source_geometry(self.frame)
        pose = torch.eye(4)
        result = self.tracker.track_pair(features, features, geometry, pose, pose)
        self.assertEqual(result.status, "converged")
        self.assertEqual(result.iterations, 2)
        self.assertIsNotNone(result.final_problem)
        self.assertFalse(hasattr(result.final_problem, "target_gt_pose"))
        torch.testing.assert_close(result.T_rig_from_world_target, pose)
        self.assertTrue(all(item.correspondence_count > 0 for item in result.history))

    def test_front_and_back_modes_keep_only_requested_camera(self):
        features = self.tracker.extract_features(self.frame.rgb, self.frame.index)
        geometry = self.tracker.prepare_source_geometry(self.frame)
        pose = torch.eye(4)
        front = self.tracker.track_pair(
            features, features, geometry, pose, pose, camera_mode="front"
        )
        back = self.tracker.track_pair(
            features, features, geometry, pose, pose, camera_mode="back"
        )
        self.assertIsNotNone(front.final_problem.front)
        self.assertIsNone(front.final_problem.back)
        self.assertIsNone(back.final_problem.front)
        self.assertIsNotNone(back.final_problem.back)

    def test_config_rejects_non_divisible_network_resolution(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            DroidFisheyeMotionConfig(processed_height=719)


def _pose_error(estimate, truth):
    relative = estimate.double().cpu() @ torch.linalg.inv(truth.double().cpu())
    translation = float(torch.linalg.vector_norm(relative[:3, 3]))
    rotation = relative[:3, :3]
    cosine = max(-1.0, min(1.0, float((torch.trace(rotation) - 1.0) / 2.0)))
    sine = 0.5 * float(
        torch.linalg.vector_norm(
            torch.stack(
                (
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                )
            )
        )
    )
    return translation, math.degrees(math.atan2(sine, cosine))


@unittest.skipUnless(
    torch.cuda.is_available()
    and DEFAULT_DATA_ROOT.is_dir()
    and (PROJECT_ROOT / "pretrained_models/droid.pth").is_file(),
    "real DROID fisheye validation requires CUDA, dataset, and checkpoint",
)
class RealDroidFisheyeTrackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = StereoFisheyeDataset()
        cls.cameras = tuple(
            DoubleSphereCamera(
                *cls.dataset.camera_params[index], *cls.dataset.image_size[index]
            )
            for index in range(2)
        )
        reprojector = FisheyeRigReprojector(
            cls.cameras, torch.from_numpy(cls.dataset.T_rig_from_camera).float()
        )
        network = load_pretrained_droid(PROJECT_ROOT / "pretrained_models/droid.pth", "cuda:0")
        cls.tracker = DroidFisheyeMotionTracker(
            reprojector,
            GroundTruthRangeProvider(),
            network,
            DroidFisheyeMotionConfig(),
            device="cuda:0",
        )

    def test_three_real_pairs_all_camera_modes(self):
        for source_index, target_index in ((0, 1), (50, 51), (150, 151)):
            source = self.dataset[source_index]
            target = self.dataset[target_index]
            source_features = self.tracker.extract_features(source.rgb, source_index)
            target_features = self.tracker.extract_features(target.rgb, target_index)
            geometry = self.tracker.prepare_source_geometry(source)
            source_gt = torch.linalg.inv(source.gt_T_world_from_rig.float()).cuda()
            target_gt = torch.linalg.inv(target.gt_T_world_from_rig.float()).cuda()
            for mode in ("front", "back", "both"):
                result = self.tracker.track_pair(
                    source_features,
                    target_features,
                    geometry,
                    source_gt,
                    source_gt,
                    camera_mode=mode,
                )
                translation, rotation = _pose_error(
                    result.T_rig_from_world_target, target_gt
                )
                self.assertEqual(result.status, "converged", (source_index, mode))
                self.assertLess(translation, 0.005, (source_index, mode))
                self.assertLess(rotation, 0.05, (source_index, mode))
                self.assertTrue(bool(torch.isfinite(result.T_rig_from_world_target).all()))
                for outer in result.history:
                    costs = [
                        item.candidate_cost
                        for item in outer.lm_result.history
                        if item.accepted
                    ]
                    self.assertTrue(
                        all(right <= left + 1e-9 for left, right in zip(costs, costs[1:]))
                    )


if __name__ == "__main__":
    unittest.main()
