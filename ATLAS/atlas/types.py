from __future__ import annotations

from typing import Protocol, TypeAlias, TypedDict, overload

import torch


class NoiseSchedule(Protocol):
    @overload
    def __call__(self, t: float) -> float: ...

    @overload
    def __call__(self, t: torch.Tensor) -> torch.Tensor: ...


class ConditioningDict(TypedDict, total=False):
    """Typed conditioning payload with optional classifier-free guidance fields."""

    context: torch.Tensor
    context_mask: torch.Tensor
    mask: torch.Tensor
    embedding: torch.Tensor
    negative_context: torch.Tensor
    conditioning: torch.Tensor
    cond: "ConditioningPayload"
    uncond: "ConditioningPayload"
    guidance_scale: float | torch.Tensor
    base_batch: int


ConditioningPayload: TypeAlias = bool | torch.Tensor | ConditioningDict
