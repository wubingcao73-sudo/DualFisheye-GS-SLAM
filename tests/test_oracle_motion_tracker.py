import math
import sys
import unittest
from pathlib import Path

import torch
from lietorch import SE3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera
from hislam2.data import StereoFisheyeDataset
from hislam2.geom.fisheye_reprojection import FisheyeRigReprojector
from hislam2.range import GroundTruthRangeProvider
from hislam2.tracking import (
    OracleCameraCorrespondences,
    OracleMotionOnlyConfig,
    OracleMotionOnlyTracker,
    OracleMotionProblem,
    bilinear_sample_range,
    build_oracle_motion_problem,
)


def pose_error(estimate: torch.Tensor, ground_truth: torch.Tensor) -> tuple[float, float]:
    relative = estimate @ torch.linalg.inv(ground_truth)
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


def perturb_pose(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return SE3.exp(delta).matrix() @ pose


def move_group(group: OracleCameraCorrespondences, device: torch.device) -> OracleCameraCorrespondences:
    return OracleCameraCorrespondences(
        source_pixels=group.source_pixels.to(device),
        source_inverse_range=group.source_inverse_range.to(device),
        observed_target_pixels=group.observed_target_pixels.to(device),
        source_camera_index=group.source_camera_index,
        target_camera_index=group.target_camera_index,
        fixed_validity=group.fixed_validity.to(device),
        base_weights=group.base_weights.to(device),
    )


class OracleMotionOnlyTrackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = StereoFisheyeDataset()
        cls.frames = (cls.dataset[0], cls.dataset[1])
        cls.cameras = [
            DoubleSphereCamera(
                *cls.dataset.camera_params[index], *cls.dataset.image_size[index]
            )
            for index in range(2)
        ]
        cls.extrinsics64 = torch.from_numpy(cls.dataset.T_rig_from_camera).double()
        cls.reprojector = FisheyeRigReprojector(cls.cameras, cls.extrinsics64)
        cls.problem, cls.target_gt = build_oracle_motion_problem(
            cls.frames[0],
            cls.frames[1],
            cls.reprojector,
            GroundTruthRangeProvider(),
            stride=32,
        )

    def tracker(self, **overrides) -> OracleMotionOnlyTracker:
        defaults = {
            "minimum_total_observations": 500,
            "minimum_observations_per_camera": 200,
        }
        defaults.update(overrides)
        return OracleMotionOnlyTracker(
            self.reprojector, OracleMotionOnlyConfig(**defaults)
        )

    def test_align_corners_integer_pixel_identity(self):
        image = torch.arange(42, dtype=torch.float64).reshape(6, 7)
        valid = torch.ones_like(image, dtype=torch.bool)
        pixels = torch.tensor(
            [[1.0, 1.0], [2.0, 3.0], [5.0, 4.0]], dtype=torch.float64
        )
        sampled, sampled_valid = bilinear_sample_range(image, valid, pixels)
        expected = image[pixels[:, 1].long(), pixels[:, 0].long()]
        self.assertTrue(bool(sampled_valid.all()))
        self.assertTrue(torch.equal(sampled, expected))

    def test_problem_contains_no_target_ground_truth(self):
        self.assertFalse(hasattr(self.problem, "gt_target_pose"))
        self.assertFalse(hasattr(self.problem, "T_rig_from_world_target"))
        self.assertGreater(self.problem.front.count, 2000)
        self.assertGreater(self.problem.back.count, 2000)
        self.assertEqual(self.problem.front.source_camera_index, 0)
        self.assertEqual(self.problem.back.source_camera_index, 1)

    def test_ground_truth_initialization_has_zero_update(self):
        result = self.tracker().optimize(self.problem, self.target_gt)
        translation, rotation = pose_error(
            result.T_rig_from_world_target, self.target_gt
        )
        self.assertEqual(result.status, "converged")
        self.assertLess(result.initial_cost, 1e-16)
        self.assertLess(result.final_cost, 1e-16)
        self.assertLess(translation, 1e-10)
        self.assertLess(rotation, 1e-10)

    def test_positive_and_negative_single_axis_perturbations(self):
        tracker = self.tracker()
        for axis in range(6):
            magnitude = 0.05 if axis < 3 else math.radians(2.0)
            for sign in (-1.0, 1.0):
                delta = torch.zeros(6, dtype=torch.float64)
                delta[axis] = sign * magnitude
                initial = perturb_pose(self.target_gt, delta)
                initial_translation, initial_rotation = pose_error(initial, self.target_gt)
                result = tracker.optimize(self.problem, initial)
                final_translation, final_rotation = pose_error(
                    result.T_rig_from_world_target, self.target_gt
                )
                self.assertEqual(result.status, "converged", (axis, sign, result))
                self.assertLess(final_translation, 0.005)
                self.assertLess(final_rotation, math.radians(0.05))
                if axis < 3:
                    self.assertLess(final_translation, 0.1 * initial_translation)
                else:
                    self.assertLess(final_rotation, 0.1 * initial_rotation)
                accepted_costs = [
                    item.candidate_cost for item in result.history if item.accepted
                ]
                self.assertTrue(
                    all(
                        right <= left
                        for left, right in zip(accepted_costs, accepted_costs[1:])
                    )
                )

    def test_balanced_scales_are_fixed_from_base_weights(self):
        tracker = self.tracker(camera_weighting="balanced")
        result = tracker.optimize(self.problem, self.target_gt)
        front_sum = float(self.problem.front.base_weights.sum())
        back_sum = float(self.problem.back.base_weights.sum())
        total = front_sum + back_sum
        expected = (total / (2.0 * front_sum), total / (2.0 * back_sum))
        self.assertAlmostEqual(result.camera_scales[0], expected[0], places=12)
        self.assertAlmostEqual(result.camera_scales[1], expected[1], places=12)
        self.assertAlmostEqual(
            result.camera_scales[0] * front_sum,
            result.camera_scales[1] * back_sum,
            places=8,
        )

    def test_front_only_checks_only_front_minimum(self):
        front_problem = self.problem.camera_mode("front")
        result = self.tracker().optimize(front_problem, self.target_gt)
        self.assertEqual(result.status, "converged")
        self.assertEqual(result.camera_scales, (1.0, 0.0))

    def test_insufficient_observations(self):
        group = self.problem.front
        count = 100
        small = OracleCameraCorrespondences(
            source_pixels=group.source_pixels[:count],
            source_inverse_range=group.source_inverse_range[:count],
            observed_target_pixels=group.observed_target_pixels[:count],
            source_camera_index=0,
            target_camera_index=0,
            fixed_validity=group.fixed_validity[:count],
            base_weights=group.base_weights[:count],
        )
        problem = OracleMotionProblem(self.problem.T_rig_from_world_source, small, None)
        result = self.tracker().optimize(problem, self.target_gt)
        self.assertEqual(result.status, "insufficient_observations")

    def test_cost_uses_fixed_denominator_with_invalid_penalty(self):
        tracker = self.tracker(maximum_candidate_invalid_fraction=0.99)
        scales = tracker._camera_scales(self.problem)
        valid = tracker._evaluate(self.problem, self.target_gt, scales, False)
        delta = torch.tensor(
            [0.0, 0.0, 0.0, math.radians(40.0), 0.0, 0.0], dtype=torch.float64
        )
        invalid = tracker._evaluate(
            self.problem, perturb_pose(self.target_gt, delta), scales, False
        )
        self.assertGreater(invalid.invalid_count, 0)
        self.assertEqual(self.problem.fixed_count, valid.valid_count)
        self.assertGreater(float(invalid.cost), float(valid.cost))

    def test_rejects_invalid_candidates_and_reports_numerical_failure(self):
        tracker = self.tracker(maximum_candidate_invalid_fraction=0.0)
        delta = torch.tensor(
            [0.0, 0.0, 0.0, math.radians(40.0), 0.0, 0.0], dtype=torch.float64
        )
        result = tracker.optimize(self.problem, perturb_pose(self.target_gt, delta))
        self.assertEqual(result.status, "numerical_failure")
        self.assertTrue(result.history)
        self.assertTrue(all(not item.accepted for item in result.history))
        self.assertTrue(all(item.candidate_invalid_count > 0 for item in result.history))

    def test_iteration_limit_reports_max_iterations(self):
        tracker = self.tracker(maximum_iterations=1)
        delta = torch.tensor(
            [0.05, 0.0, 0.0, 0.0, math.radians(2.0), 0.0], dtype=torch.float64
        )
        result = tracker.optimize(self.problem, perturb_pose(self.target_gt, delta))
        self.assertEqual(result.status, "max_iterations")
        self.assertEqual(result.iterations, 1)
        self.assertLess(result.final_cost, result.initial_cost)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_float32_matches_cpu_float32(self):
        problem32, target32 = build_oracle_motion_problem(
            self.frames[0],
            self.frames[1],
            FisheyeRigReprojector(self.cameras, self.extrinsics64.float()),
            GroundTruthRangeProvider(),
            stride=32,
            dtype=torch.float32,
        )
        delta = torch.tensor(
            [0.03, -0.02, 0.01, math.radians(1.0), 0.0, math.radians(-1.0)],
            dtype=torch.float32,
        )
        initial_cpu = perturb_pose(target32, delta)
        config = OracleMotionOnlyConfig(
            minimum_total_observations=500,
            minimum_observations_per_camera=200,
        )
        cpu = OracleMotionOnlyTracker(
            FisheyeRigReprojector(self.cameras, self.extrinsics64.float()), config
        ).optimize(problem32, initial_cpu)
        device = torch.device("cuda")
        problem_cuda = OracleMotionProblem(
            problem32.T_rig_from_world_source.to(device),
            move_group(problem32.front, device),
            move_group(problem32.back, device),
        )
        cuda = OracleMotionOnlyTracker(
            FisheyeRigReprojector(self.cameras, self.extrinsics64.float().to(device)),
            config,
        ).optimize(problem_cuda, initial_cpu.to(device))
        translation, rotation = pose_error(
            cuda.T_rig_from_world_target.cpu(), cpu.T_rig_from_world_target
        )
        self.assertEqual(cpu.status, "converged")
        self.assertEqual(cuda.status, "converged")
        self.assertLess(translation, 1e-3)
        self.assertLess(rotation, math.radians(0.01))


if __name__ == "__main__":
    unittest.main()
