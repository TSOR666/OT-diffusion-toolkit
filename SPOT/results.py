"""Result dataclasses and lightweight context containers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

__all__ = ["StepContext", "SamplingResult"]


class StepContext(NamedTuple):
    """Thread-safe context for current timestep information."""

    current_t: float
    current_dt: float


@dataclass
class SamplingResult:
    """Result from sampling with optional statistics."""

    samples: torch.Tensor
    stats: Optional[Dict[str, Any]] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.samples.shape)

    def to(self, device_or_dtype):
        """Move samples to device or dtype, returning a new instance."""

        return SamplingResult(
            samples=self.samples.to(device_or_dtype),
            stats=None if self.stats is None else dict(self.stats),
        )
