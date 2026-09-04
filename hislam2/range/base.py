from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from hislam2.data.frame_types import StereoFisheyeFrame


@dataclass(frozen=True)
class RangeObservation:
    """Range data and frame-dependent validity, independent of camera geometry."""

    range_m: torch.Tensor
    inverse_range: torch.Tensor
    observation_valid: torch.Tensor
    confidence: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.range_m.shape
        if self.range_m.dtype not in (torch.float32, torch.float64):
            raise TypeError("range_m must use float32 or float64")
        if self.range_m.ndim < 2:
            raise ValueError(f"range_m must have at least 2 dimensions, got {shape}")
        for name in ("inverse_range", "observation_valid", "confidence"):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match range_m {shape}")
            if value.device != self.range_m.device:
                raise ValueError(f"{name} and range_m must be on the same device")
        if self.inverse_range.dtype != self.range_m.dtype:
            raise TypeError("inverse_range and range_m must use the same dtype")
        if self.confidence.dtype != self.range_m.dtype:
            raise TypeError("confidence and range_m must use the same dtype")
        if self.observation_valid.dtype != torch.bool:
            raise TypeError("observation_valid must use bool dtype")


class RangeProvider(ABC):
    """Interface for producing Euclidean range observations."""

    @abstractmethod
    def provide(self, frame: "StereoFisheyeFrame") -> RangeObservation:
        """Return range, inverse range, observation validity and confidence."""
