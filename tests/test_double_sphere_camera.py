import unittest
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hislam2.camera import DoubleSphereCamera
from hislam2.data import StereoFisheyeDataset


class DoubleSphereCameraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = StereoFisheyeDataset()
        cls.cameras = [
            DoubleSphereCamera(
                *dataset.camera_params[index],
                *dataset.image_size[index],
            )
            for index in range(2)
        ]

    def test_pixel_ray_pixel_roundtrip_float64(self):
        generator = torch.Generator().manual_seed(7)
        for camera in self.cameras:
            pixels = torch.stack(
                (
                    torch.rand(100_000, generator=generator, dtype=torch.float64)
                    * camera.width,
                    torch.rand(100_000, generator=generator, dtype=torch.float64)
                    * camera.height,
                ),
                dim=-1,
            )
            rays, unproject_valid = camera.unproject(pixels)
            reconstructed, project_valid = camera.project(rays)
            valid = unproject_valid & project_valid
            error = torch.linalg.vector_norm(reconstructed[valid] - pixels[valid], dim=-1)
            self.assertLess(float(torch.quantile(error, 0.99)), 1e-3)

    def test_point_range_ray_reconstruction_includes_negative_z(self):
        for camera in self.cameras:
            rays, model_valid = camera.get_ray_lut(dtype=torch.float64)
            negative = model_valid & (rays[..., 2] < 0.0)
            self.assertTrue(bool(negative.any()))
            selected = rays[negative][:: max(1, int(negative.sum()) // 20_000)]
            ranges = torch.linspace(0.5, 10.0, len(selected), dtype=torch.float64)
            points = selected * ranges[:, None]
            pixels, project_valid = camera.project(points)
            rebuilt_rays, unproject_valid = camera.unproject(pixels)
            rebuilt = rebuilt_rays * torch.linalg.vector_norm(points, dim=-1, keepdim=True)
            valid = project_valid & unproject_valid
            relative_error = torch.linalg.vector_norm(
                rebuilt[valid] - points[valid], dim=-1
            ) / torch.linalg.vector_norm(points[valid], dim=-1)
            self.assertLess(float(torch.quantile(relative_error, 0.99)), 1e-5)
            camera.clear_ray_lut_cache()

    def test_unit_ray_norm(self):
        for camera in self.cameras:
            pixels = torch.tensor(
                [
                    [camera.cx, camera.cy],
                    [0.0, 0.0],
                    [camera.width - 1.0, camera.height - 1.0],
                    [camera.cx, 0.0],
                    [0.0, camera.cy],
                ],
                dtype=torch.float64,
            )
            rays, valid = camera.unproject(pixels)
            norm_error = torch.abs(torch.linalg.vector_norm(rays[valid], dim=-1) - 1.0)
            self.assertLess(float(norm_error.max()), 1e-6)

    def test_model_and_image_valid_are_separate(self):
        camera = self.cameras[0]
        pixels = torch.tensor(
            [
                [camera.cx, camera.cy],
                [-1.0, camera.cy],
                [camera.width + 5000.0, camera.height + 5000.0],
            ],
            dtype=torch.float64,
        )
        model_valid = camera.valid_mask(pixels)
        image_valid = camera.image_valid(pixels)
        self.assertTrue(bool(model_valid[0] & image_valid[0]))
        self.assertTrue(bool(model_valid[1]))
        self.assertFalse(bool(image_valid[1]))
        self.assertFalse(bool(model_valid[2]))
        self.assertFalse(bool(image_valid[2]))

    def test_invalid_projection_is_not_defined_by_positive_z(self):
        camera = self.cameras[0]
        pixels = torch.tensor([[camera.cx, 0.0]], dtype=torch.float64)
        ray, unproject_valid = camera.unproject(pixels)
        self.assertTrue(bool(unproject_valid[0]))
        self.assertLess(float(ray[0, 2]), 0.0)
        _, project_valid = camera.project(ray)
        self.assertTrue(bool(project_valid[0]))

    def test_lut_cache_key_and_clear(self):
        camera = DoubleSphereCamera(-0.12, 0.56, 20.0, 20.0, 16.0, 12.0, 32, 24)
        rays_a, valid_a = camera.get_ray_lut(dtype=torch.float32)
        rays_b, valid_b = camera.get_ray_lut(dtype=torch.float32)
        self.assertEqual(tuple(rays_a.shape), (24, 32, 3))
        self.assertEqual(tuple(valid_a.shape), (24, 32))
        self.assertEqual(rays_a.data_ptr(), rays_b.data_ptr())
        self.assertEqual(valid_a.data_ptr(), valid_b.data_ptr())
        with self.assertRaisesRegex(ValueError, "does not implicitly rescale"):
            camera.get_ray_lut(12, 16)
        camera.clear_ray_lut_cache()
        rays_c, _ = camera.get_ray_lut(dtype=torch.float32)
        self.assertNotEqual(rays_a.data_ptr(), rays_c.data_ptr())

    def test_float32_matches_float64(self):
        camera = self.cameras[1]
        rng = np.random.default_rng(13)
        pixels64 = torch.from_numpy(
            np.column_stack(
                (
                    rng.uniform(0, camera.width, 20_000),
                    rng.uniform(0, camera.height, 20_000),
                )
            )
        ).to(torch.float64)
        rays64, valid64 = camera.unproject(pixels64)
        rays32, valid32 = camera.unproject(pixels64.float())
        self.assertTrue(torch.equal(valid64, valid32))
        maximum_difference = torch.max(torch.abs(rays64.float()[valid64] - rays32[valid32]))
        self.assertLess(float(maximum_difference), 1e-5)

    def test_projection_jacobian_matches_project_and_autograd(self):
        generator = torch.Generator().manual_seed(23)
        for camera in self.cameras:
            pixels = torch.rand((2048, 2), generator=generator, dtype=torch.float64)
            pixels = pixels * torch.tensor(
                (camera.width, camera.height), dtype=torch.float64
            )
            rays, unproject_valid = camera.unproject(pixels)
            points = rays[unproject_valid] * torch.linspace(
                0.5, 10.0, int(unproject_valid.sum()), dtype=torch.float64
            )[:, None]
            points.requires_grad_(True)

            projected, model_valid, analytic = camera.project_jacobian(points)
            plain_projected, plain_valid = camera.project(points)
            self.assertTrue(torch.equal(model_valid, plain_valid))
            self.assertTrue(torch.equal(projected, plain_projected))
            self.assertEqual(tuple(analytic.shape), (len(points), 2, 3))

            autograd_rows = []
            for output_index in range(2):
                autograd_rows.append(
                    torch.autograd.grad(
                        projected[:, output_index].sum(),
                        points,
                        retain_graph=output_index == 0,
                    )[0]
                )
            autograd_jacobian = torch.stack(autograd_rows, dim=-2)
            relative_error = torch.linalg.vector_norm(
                analytic[model_valid] - autograd_jacobian[model_valid], dim=(-2, -1)
            ) / torch.linalg.vector_norm(
                autograd_jacobian[model_valid], dim=(-2, -1)
            )
            self.assertLess(float(torch.quantile(relative_error, 0.99)), 1e-10)

    def test_projection_jacobian_matches_finite_difference_with_negative_z(self):
        camera = self.cameras[0]
        pixels = torch.tensor(
            [
                [camera.cx, camera.cy],
                [camera.cx, 0.0],
                [0.0, camera.cy],
                [camera.width - 1.0, camera.cy],
                [camera.cx, camera.height - 1.0],
                [400.0, 400.0],
                [2480.0, 400.0],
                [400.0, 2480.0],
                [2480.0, 2480.0],
            ],
            dtype=torch.float64,
        )
        rays, unproject_valid = camera.unproject(pixels)
        points = rays[unproject_valid] * 3.0
        self.assertTrue(bool((points[:, 2] < 0.0).any()))
        _, valid, analytic = camera.project_jacobian(points)

        step = 1e-6 * torch.maximum(
            torch.ones(len(points), dtype=points.dtype),
            torch.linalg.vector_norm(points, dim=-1),
        )
        columns = []
        perturbations_valid = valid.clone()
        for axis in range(3):
            delta = torch.zeros_like(points)
            delta[:, axis] = step
            plus, plus_valid = camera.project(points + delta)
            minus, minus_valid = camera.project(points - delta)
            perturbations_valid &= plus_valid & minus_valid
            columns.append((plus - minus) / (2.0 * step[:, None]))
        finite_difference = torch.stack(columns, dim=-1)
        relative_error = torch.linalg.vector_norm(
            analytic[perturbations_valid] - finite_difference[perturbations_valid],
            dim=(-2, -1),
        ) / torch.linalg.vector_norm(
            finite_difference[perturbations_valid], dim=(-2, -1)
        )
        self.assertLess(float(relative_error.max()), 1e-6)

    def test_projection_jacobian_invalid_values_are_placeholders(self):
        camera = self.cameras[0]
        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [float("nan"), 0.0, 1.0]],
            dtype=torch.float64,
        )
        pixels, valid, jacobian = camera.project_jacobian(points)
        self.assertFalse(bool(valid.any()))
        self.assertTrue(bool((pixels == 0.0).all()))
        self.assertTrue(bool((jacobian == 0.0).all()))

    def test_projection_validity_is_scale_invariant(self):
        camera = self.cameras[0]
        ray, unproject_valid = camera.unproject(
            torch.tensor([[camera.cx, 0.0]], dtype=torch.float64)
        )
        self.assertTrue(bool(unproject_valid[0]))
        points = ray * torch.tensor((1e-3, 1.0, 1e3), dtype=torch.float64)[:, None]
        pixels, project_valid = camera.project(points)
        self.assertTrue(bool(project_valid.all()))
        self.assertLess(
            float(torch.max(torch.abs(pixels - pixels[0]))),
            1e-9,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_float32_matches_cpu(self):
        camera = self.cameras[0]
        pixels = torch.tensor(
            [[camera.cx, camera.cy], [0.0, camera.cy], [camera.cx, 0.0]],
            dtype=torch.float32,
        )
        cpu_rays, cpu_valid = camera.unproject(pixels)
        gpu_rays, gpu_valid = camera.unproject(pixels.cuda())
        self.assertTrue(torch.equal(cpu_valid, gpu_valid.cpu()))
        self.assertLess(float(torch.max(torch.abs(cpu_rays - gpu_rays.cpu()))), 1e-5)

        points = cpu_rays[cpu_valid] * 2.0
        cpu_pixels, cpu_project_valid, cpu_jacobian = camera.project_jacobian(points)
        gpu_pixels, gpu_project_valid, gpu_jacobian = camera.project_jacobian(points.cuda())
        self.assertTrue(torch.equal(cpu_project_valid, gpu_project_valid.cpu()))
        self.assertLess(
            float(torch.max(torch.abs(cpu_pixels - gpu_pixels.cpu()))), 1e-3
        )
        self.assertLess(
            float(torch.max(torch.abs(cpu_jacobian - gpu_jacobian.cpu()))), 1e-3
        )


if __name__ == "__main__":
    unittest.main()
