from __future__ import annotations

from typing import Any, Protocol, TypeAlias, overload

import torch


class NoiseSchedule(Protocol):
    @overload
    def __call__(self, t: float) -> float: ...

    @overload
    def __call__(self, t: torch.Tensor) -> torch.Tensor: ...


ConditioningPayload: TypeAlias = bool | torch.Tensor | dict[str, Any]
