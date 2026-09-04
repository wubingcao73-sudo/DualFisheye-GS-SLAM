import math
from typing import TYPE_CHECKING

import torch

from .base import RangeObservation, RangeProvider

if TYPE_CHECKING:
    from hislam2.data.frame_types import StereoFisheyeFrame


class GroundTruthRangeProvider(RangeProvider):
    """Convert raw simulated GT Euclidean range into a uniform observation."""

    def __init__(
        self,
        invalid_sentinel: float = 1.0e10,
        sentinel_relative_margin: float = 1.0e-6,
    ) -> None:
        if not math.isfinite(invalid_sentinel) or invalid_sentinel <= 0.0:
            raise ValueError("invalid_sentinel must be finite and positive")
        if not 0.0 <= sentinel_relative_margin < 1.0:
            raise ValueError("sentinel_relative_margin must lie in [0, 1)")
        self.invalid_sentinel = float(invalid_sentinel)
        self.sentinel_relative_margin = float(sentinel_relative_margin)

    def from_range(self, range_m: torch.Tensor) -> RangeObservation:
        """Build an observation from a raw Euclidean range tensor."""
        if not isinstance(range_m, torch.Tensor):
            raise TypeError("range_m must be a torch.Tensor")
        if range_m.dtype not in (torch.float32, torch.float64):
            raise TypeError("range_m must use float32 or float64")
        if range_m.ndim < 2:
            raise ValueError(
                f"range_m must have at least 2 dimensions, got {tuple(range_m.shape)}"
            )

        sentinel_limit = self.invalid_sentinel * (1.0 - self.sentinel_relative_margin)
        observation_valid = (
            torch.isfinite(range_m)
            & (range_m > 0.0)
            & (range_m < sentinel_limit)
        )
        safe_range = torch.where(observation_valid, range_m, torch.ones_like(range_m))
        inverse_range = torch.where(
            observation_valid,
            safe_range.reciprocal(),
            torch.zeros_like(range_m),
        )
        confidence = observation_valid.to(dtype=range_m.dtype)
        return RangeObservation(
            range_m=range_m,
            inverse_range=inverse_range,
            observation_valid=observation_valid,
            confidence=confidence,
        )

    def provide(self, frame: "StereoFisheyeFrame") -> RangeObservation:
        if not hasattr(frame, "gt_range"):
            raise TypeError("frame must provide a gt_range tensor")
        return self.from_range(frame.gt_range)
