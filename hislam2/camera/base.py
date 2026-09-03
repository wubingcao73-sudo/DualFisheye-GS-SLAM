from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class CameraModel(ABC):
    """Geometry-only camera interface.

    Camera models own mathematical validity and image-bound validity. They do
    not know whether a particular frame contains a valid scene observation.
    """

    width: int
    height: int

    def __init__(self) -> None:
        self._ray_lut_cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    @abstractmethod
    def project(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project camera-frame points ``[..., 3]`` to pixels ``[..., 2]``."""

    @abstractmethod
    def unproject(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unproject pixels ``[..., 2]`` to unit rays ``[..., 3]``."""

    @abstractmethod
    def valid_mask(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return only the camera model's mathematical unprojection validity."""

    def image_valid(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return whether pixels lie inside calibrated image bounds."""
        if pixels.shape[-1] != 2:
            raise ValueError(f"pixels must have shape [..., 2], got {tuple(pixels.shape)}")
        u, v = pixels.unbind(dim=-1)
        return (
            torch.isfinite(pixels).all(dim=-1)
            & (u >= 0.0)
            & (u < self.width)
            & (v >= 0.0)
            & (v < self.height)
        )

    def get_ray_lut(
        self,
        height: int | None = None,
        width: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lazily create a native-resolution ray table and model-valid mask."""
        height = self.height if height is None else int(height)
        width = self.width if width is None else int(width)
        if (height, width) != (self.height, self.width):
            raise ValueError(
                "Camera Geometry V0 does not implicitly rescale calibration: "
                f"requested {(width, height)}, calibrated {(self.width, self.height)}"
            )
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(f"ray LUT dtype must be float32 or float64, got {dtype}")

        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (height, width, device.type, device.index, dtype)
        cached = self._ray_lut_cache.get(key)
        if cached is not None:
            return cached

        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        pixels = torch.stack((x, y), dim=-1)
        rays, model_valid = self.unproject(pixels)
        self._ray_lut_cache[key] = (rays, model_valid)
        return rays, model_valid

    def clear_ray_lut_cache(self, device: torch.device | str | None = None) -> None:
        """Drop all cached LUTs, or only LUTs on one device."""
        if device is None:
            self._ray_lut_cache.clear()
            return
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        keys = [
            key
            for key in self._ray_lut_cache
            if key[2] == device.type and key[3] == device.index
        ]
        for key in keys:
            del self._ray_lut_cache[key]
