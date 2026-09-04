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
from hislam2.geom.fisheye_reprojection import (
    FisheyeRigReprojector,
    se3_point_action_jacobian,
)


def inverse_transform(transform: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(transform)


def left_perturb(transform: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return SE3.exp(delta).matrix() @ transform


def relative_jacobian_error(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    difference = torch.linalg.vector_norm(actual - reference, dim=(-2, -1))
    scale = torch.linalg.vector_norm(reference, dim=(-2, -1))
    return difference / torch.maximum(scale, torch.full_like(scale, 1e-12))


class FisheyeRigReprojectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = StereoFisheyeDataset()
        cls.dataset = dataset
        cls.cameras = [
            DoubleSphereCamera(
                *dataset.camera_params[index],
                *dataset.image_size[index],
            )
            for index in range(2)
        ]
        cls.extrinsics64 = torch.from_numpy(dataset.T_rig_from_camera).double()
        cls.reprojector = FisheyeRigReprojector(cls.cameras, cls.extrinsics64)
        cls.poses64 = [
            inverse_transform(torch.from_numpy(dataset.gt_T_world_from_rig[index]).double())
            for index in (0, 1)
        ]

    def sample_pixels(self, camera_index: int, count: int = 2048) -> torch.Tensor:
        camera = self.cameras[camera_index]
        generator = torch.Generator().manual_seed(41 + camera_index)
        pixels = torch.rand((count, 2), generator=generator, dtype=torch.float64)
        pixels *= torch.tensor((camera.width, camera.height), dtype=torch.float64)
        special = torch.tensor(
            [
                [camera.cx, camera.cy],
                [camera.cx, 0.0],
                [0.0, camera.cy],
                [camera.width - 1.0, camera.cy],
                [camera.cx, camera.height - 1.0],
            ],
            dtype=torch.float64,
        )
        return torch.cat((pixels, special), dim=0)

    def test_lietorch_twist_order_and_point_action_jacobian(self):
        points = torch.tensor(
            [[0.4, -0.7, 1.3], [-1.0, 0.2, -0.3]], dtype=torch.float64
        )
        analytic = se3_point_action_jacobian(points)
        step = 1e-6
        columns = []
        for axis in range(6):
            delta = torch.zeros(6, dtype=torch.float64)
            delta[axis] = step
            plus = SE3.exp(delta.expand(len(points), 6)) * points
            minus = SE3.exp((-delta).expand(len(points), 6)) * points
            columns.append((plus - minus) / (2.0 * step))
        finite_difference = torch.stack(columns, dim=-1)
        self.assertLess(float(torch.abs(analytic - finite_difference).max()), 1e-9)
        self.assertTrue(
            torch.allclose(
                analytic[:, :, :3],
                torch.eye(3, dtype=torch.float64).expand(2, 3, 3),
            )
        )

    def test_identity_reprojection_includes_negative_z(self):
        pose = self.poses64[0]
        for camera_index, camera in enumerate(self.cameras):
            pixels = self.sample_pixels(camera_index)
            inverse_range = torch.full((len(pixels),), 0.5, dtype=torch.float64)
            result = self.reprojector.reproject(
                pixels,
                inverse_range,
                pose,
                pose,
                camera_index,
                camera_index,
            )
            valid = result.validity.geometric_valid
            rays, ray_valid = camera.unproject(pixels)
            self.assertTrue(bool((valid & ray_valid & (rays[:, 2] < 0.0)).any()))
            error = torch.linalg.vector_norm(result.pixels[valid] - pixels[valid], dim=-1)
            self.assertLess(float(torch.quantile(error, 0.99)), 1e-9)
            self.assertTrue(
                torch.allclose(
                    result.target_range[valid],
                    torch.full_like(result.target_range[valid], 2.0),
                    atol=1e-12,
                )
            )

    def _pose_finite_difference(
        self,
        source_camera_index: int,
        target_camera_index: int,
        pose_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.sample_pixels(source_camera_index, count=1024)
        inverse_range = torch.linspace(0.1, 1.0, len(pixels), dtype=torch.float64)
        source_pose, target_pose = self.poses64
        base = self.reprojector.reproject(
            pixels,
            inverse_range,
            source_pose,
            target_pose,
            source_camera_index,
            target_camera_index,
            compute_jacobians=True,
        )
        common_valid = base.validity.geometric_valid.clone()
        columns = []
        step = 1e-6
        for axis in range(6):
            delta = torch.zeros(6, dtype=torch.float64)
            delta[axis] = step
            if pose_name == "source":
                plus_source = left_perturb(source_pose, delta)
                minus_source = left_perturb(source_pose, -delta)
                plus_target = minus_target = target_pose
            else:
                plus_source = minus_source = source_pose
                plus_target = left_perturb(target_pose, delta)
                minus_target = left_perturb(target_pose, -delta)
            plus = self.reprojector.reproject(
                pixels,
                inverse_range,
                plus_source,
                plus_target,
                source_camera_index,
                target_camera_index,
            )
            minus = self.reprojector.reproject(
                pixels,
                inverse_range,
                minus_source,
                minus_target,
                source_camera_index,
                target_camera_index,
            )
            common_valid &= (
                plus.validity.geometric_valid & minus.validity.geometric_valid
            )
            columns.append((plus.pixels - minus.pixels) / (2.0 * step))
        finite_difference = torch.stack(columns, dim=-1)
        analytic = getattr(base.jacobians, f"{pose_name}_pose")
        return analytic[common_valid], finite_difference[common_valid]

    def test_source_and_target_pose_jacobians_all_camera_pairs(self):
        for source_camera_index, target_camera_index in (
            (0, 0),
            (1, 1),
            (0, 1),
            (1, 0),
        ):
            for pose_name in ("source", "target"):
                analytic, finite_difference = self._pose_finite_difference(
                    source_camera_index, target_camera_index, pose_name
                )
                self.assertGreater(len(analytic), 20)
                error = relative_jacobian_error(analytic, finite_difference)
                self.assertLess(float(torch.quantile(error, 0.99)), 1e-5)

    def test_inverse_range_jacobian_all_camera_pairs(self):
        source_pose, target_pose = self.poses64
        step = 1e-6
        for source_camera_index, target_camera_index in (
            (0, 0),
            (1, 1),
            (0, 1),
            (1, 0),
        ):
            pixels = self.sample_pixels(source_camera_index, count=1024)
            inverse_range = torch.linspace(0.1, 1.0, len(pixels), dtype=torch.float64)
            base = self.reprojector.reproject(
                pixels,
                inverse_range,
                source_pose,
                target_pose,
                source_camera_index,
                target_camera_index,
                compute_jacobians=True,
            )
            plus = self.reprojector.reproject(
                pixels,
                inverse_range + step,
                source_pose,
                target_pose,
                source_camera_index,
                target_camera_index,
            )
            minus = self.reprojector.reproject(
                pixels,
                inverse_range - step,
                source_pose,
                target_pose,
                source_camera_index,
                target_camera_index,
            )
            common_valid = (
                base.validity.geometric_valid
                & plus.validity.geometric_valid
                & minus.validity.geometric_valid
            )
            finite_difference = (
                (plus.pixels - minus.pixels) / (2.0 * step)
            )[..., None]
            analytic = base.jacobians.inverse_range
            self.assertGreater(int(common_valid.sum()), 20)
            error = relative_jacobian_error(
                analytic[common_valid], finite_difference[common_valid]
            )
            self.assertLess(float(torch.quantile(error, 0.99)), 1e-5)

    def test_validity_components_remain_separate(self):
        camera = self.cameras[0]
        pixels = torch.tensor(
            [
                [camera.cx, camera.cy],
                [-1.0, camera.cy],
                [camera.cx, camera.cy],
                [camera.width + 5000.0, camera.height + 5000.0],
            ],
            dtype=torch.float64,
        )
        inverse_range = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=torch.float64)
        pose = self.poses64[0]
        result = self.reprojector.reproject(
            pixels, inverse_range, pose, pose, 0, 0, compute_jacobians=True
        )
        validity = result.validity
        self.assertTrue(bool(validity.geometric_valid[0]))
        self.assertTrue(bool(validity.source_model_valid[1]))
        self.assertFalse(bool(validity.source_image_valid[1]))
        self.assertFalse(bool(validity.range_valid[2]))
        self.assertFalse(bool(validity.source_model_valid[3]))
        self.assertTrue(bool((result.pixels[~validity.geometric_valid] == 0.0).all()))
        self.assertTrue(
            bool((result.jacobians.source_pose[~validity.geometric_valid] == 0.0).all())
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_float32_matches_cpu(self):
        pixels = self.sample_pixels(0, count=2048).float()
        inverse_range = torch.full((len(pixels),), 0.5, dtype=torch.float32)
        poses = [pose.float() for pose in self.poses64]
        cpu_reprojector = FisheyeRigReprojector(
            self.cameras, self.extrinsics64.float()
        )
        cpu = cpu_reprojector.reproject(
            pixels, inverse_range, poses[0], poses[1], 0, 0, True
        )
        gpu_reprojector = FisheyeRigReprojector(
            self.cameras, self.extrinsics64.float().cuda()
        )
        gpu = gpu_reprojector.reproject(
            pixels.cuda(),
            inverse_range.cuda(),
            poses[0].cuda(),
            poses[1].cuda(),
            0,
            0,
            True,
        )
        common_valid = cpu.validity.geometric_valid & gpu.validity.geometric_valid.cpu()
        self.assertTrue(
            torch.equal(
                cpu.validity.geometric_valid, gpu.validity.geometric_valid.cpu()
            )
        )
        self.assertLess(
            float(torch.abs(cpu.pixels[common_valid] - gpu.pixels.cpu()[common_valid]).max()),
            1e-3,
        )
        for name in ("source_pose", "target_pose", "inverse_range"):
            cpu_jacobian = getattr(cpu.jacobians, name)[common_valid]
            gpu_jacobian = getattr(gpu.jacobians, name).cpu()[common_valid]
            error = relative_jacobian_error(cpu_jacobian, gpu_jacobian)
            self.assertLess(float(torch.quantile(error, 0.99)), 1e-4)


if __name__ == "__main__":
    unittest.main()
