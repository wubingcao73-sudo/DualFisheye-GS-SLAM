import unittest

import numpy as np
import torch

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


if __name__ == "__main__":
    unittest.main()
