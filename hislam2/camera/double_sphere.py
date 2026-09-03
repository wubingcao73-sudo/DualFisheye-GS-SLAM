import math
from typing import Sequence, Tuple

import torch

from .base import CameraModel


class DoubleSphereCamera(CameraModel):
    """Double Sphere camera model from Usenko, Demmel and Cremers (3DV 2018).

    Camera coordinates are x-right, y-down and z-forward. Pixel coordinates
    are u-right and v-down, with integer coordinates denoting pixel centers.
    """

    parameter_order = ("xi", "alpha", "fx", "fy", "cx", "cy")

    def __init__(
        self,
        xi: float,
        alpha: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> None:
        super().__init__()
        values = (xi, alpha, fx, fy, cx, cy)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Double Sphere parameters must be finite")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("fx and fy must be positive")
        if int(width) != width or int(height) != height or width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")
        if not 0.0 <= cx < width or not 0.0 <= cy < height:
            raise ValueError("principal point must lie inside the calibrated image")

        self.xi = float(xi)
        self.alpha = float(alpha)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)

        w1 = alpha / (1.0 - alpha) if alpha <= 0.5 else (1.0 - alpha) / alpha
        w2_radicand = 2.0 * w1 * xi + xi * xi + 1.0
        if w2_radicand <= 0.0:
            raise ValueError("Double Sphere parameters produce an invalid projection domain")
        self._w2 = float((w1 + xi) / math.sqrt(w2_radicand))

    @classmethod
    def from_calibration(cls, calibration: Sequence[float]) -> "DoubleSphereCamera":
        if len(calibration) != 8:
            raise ValueError("calibration must be xi alpha fx fy cx cy width height")
        return cls(*calibration[:6], int(calibration[6]), int(calibration[7]))

    @property
    def parameters(self) -> tuple:
        return (self.xi, self.alpha, self.fx, self.fy, self.cx, self.cy)

    @staticmethod
    def _check_tensor(value: torch.Tensor, final_size: int, name: str) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if value.ndim == 0 or value.shape[-1] != final_size:
            raise ValueError(
                f"{name} must have shape [..., {final_size}], got {tuple(value.shape)}"
            )

    def project(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_tensor(points, 3, "points")
        X, Y, Z = points.unbind(dim=-1)
        d1 = torch.linalg.vector_norm(points, dim=-1)
        z1 = self.xi * d1 + Z
        d2 = torch.sqrt(X * X + Y * Y + z1 * z1)
        denominator = self.alpha * d2 + (1.0 - self.alpha) * z1

        finite = torch.isfinite(points).all(dim=-1)
        scale = torch.maximum(d1, torch.ones_like(d1))
        epsilon = torch.finfo(points.dtype).eps * 16.0 * scale
        model_valid = (
            finite
            & (d1 > epsilon)
            & (Z > -self._w2 * d1)
            & (denominator > epsilon)
        )

        safe_denominator = torch.where(model_valid, denominator, torch.ones_like(denominator))
        u = self.fx * X / safe_denominator + self.cx
        v = self.fy * Y / safe_denominator + self.cy
        pixels = torch.stack((u, v), dim=-1)
        pixels = torch.where(model_valid[..., None], pixels, torch.zeros_like(pixels))
        return pixels, model_valid

    def _unprojection_terms(self, pixels: torch.Tensor):
        self._check_tensor(pixels, 2, "pixels")
        u, v = pixels.unbind(dim=-1)
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        radius_squared = mx * mx + my * my

        first_radicand = 1.0 - (2.0 * self.alpha - 1.0) * radius_squared
        model_valid = torch.isfinite(pixels).all(dim=-1) & (first_radicand >= 0.0)
        safe_first = torch.where(model_valid, first_radicand, torch.ones_like(first_radicand))
        first_root = torch.sqrt(safe_first)
        mz_denominator = self.alpha * first_root + (1.0 - self.alpha)
        scale = torch.maximum(radius_squared, torch.ones_like(radius_squared))
        epsilon = torch.finfo(pixels.dtype).eps * 16.0 * scale
        model_valid &= mz_denominator > epsilon
        safe_mz_denominator = torch.where(
            model_valid, mz_denominator, torch.ones_like(mz_denominator)
        )
        mz = (1.0 - self.alpha * self.alpha * radius_squared) / safe_mz_denominator

        second_radicand = mz * mz + (1.0 - self.xi * self.xi) * radius_squared
        model_valid &= second_radicand >= 0.0
        ray_denominator = mz * mz + radius_squared
        model_valid &= ray_denominator > epsilon
        return mx, my, mz, radius_squared, second_radicand, ray_denominator, model_valid

    def valid_mask(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._unprojection_terms(pixels)[-1]

    def unproject(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            mx,
            my,
            mz,
            _,
            second_radicand,
            ray_denominator,
            model_valid,
        ) = self._unprojection_terms(pixels)

        safe_second = torch.where(
            model_valid, second_radicand, torch.ones_like(second_radicand)
        )
        safe_ray_denominator = torch.where(
            model_valid, ray_denominator, torch.ones_like(ray_denominator)
        )
        coefficient = (
            mz * self.xi + torch.sqrt(safe_second)
        ) / safe_ray_denominator
        rays = torch.stack(
            (coefficient * mx, coefficient * my, coefficient * mz - self.xi),
            dim=-1,
        )
        ray_norm = torch.linalg.vector_norm(rays, dim=-1)
        epsilon = torch.finfo(pixels.dtype).eps * 16.0
        model_valid &= torch.isfinite(rays).all(dim=-1) & (ray_norm > epsilon)
        safe_norm = torch.where(model_valid, ray_norm, torch.ones_like(ray_norm))
        rays = rays / safe_norm[..., None]
        rays = torch.where(model_valid[..., None], rays, torch.zeros_like(rays))
        return rays, model_valid
