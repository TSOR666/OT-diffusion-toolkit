"""Noise schedule implementations for the SPOT solver."""
from __future__ import annotations

import math
from typing import Protocol, Tuple, Union

import torch

from .constants import EPSILON_CLAMP

__all__ = ["NoiseScheduleProtocol", "CosineSchedule", "LinearSchedule"]


class NoiseScheduleProtocol(Protocol):
    """Unified protocol for noise schedules with consistent semantics."""

    def alpha_sigma(self, t: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def lambda_(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        ...


class CosineSchedule:
    """Reference cosine schedule with unified semantics."""

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.s = 0.008

    def _ensure_tensor(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(t, (int, float)):
            return torch.tensor([t], device=self.device, dtype=torch.float32)
        return t.to(device=self.device, dtype=torch.float32)

    def alpha_sigma(self, t: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        t32 = self._ensure_tensor(t).clamp(0.0, 1.0)
        ft = torch.cos((t32 + self.s) / (1 + self.s) * math.pi / 2).pow(2)
        alpha = torch.sqrt(ft)
        sigma = torch.sqrt((1 - ft).clamp_min(0))
        return alpha.to(self.dtype), sigma.to(self.dtype)

    def lambda_(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(t)
        a32, s32 = alpha.float(), sigma.float()
        out = torch.log((a32 * a32) / (s32 * s32 + EPSILON_CLAMP))
        if __debug__:
            assert out.dtype == torch.float32, f"(t) must be fp32, got {out.dtype}"
        return out


class LinearSchedule:
    """Linear beta schedule with correct continuous-time integral."""

    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.device = device
        self.dtype = dtype

    def _ensure_tensor(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(t, (int, float)):
            return torch.tensor([t], device=self.device, dtype=torch.float32)
        return t.to(device=self.device, dtype=torch.float32)

    def alpha_sigma(self, t: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        t32 = self._ensure_tensor(t).clamp(0.0, 1.0)
        beta0, beta1 = self.beta_start, self.beta_end
        integral = beta0 * t32 + 0.5 * (beta1 - beta0) * t32 * t32
        alpha_bar = torch.exp(-integral)
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt((1.0 - alpha_bar).clamp_min(0))
        return alpha.to(self.dtype), sigma.to(self.dtype)

    def lambda_(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(t)
        a32, s32 = alpha.float(), sigma.float()
        out = torch.log((a32 * a32) / (s32 * s32 + EPSILON_CLAMP))
        if __debug__:
            assert out.dtype == torch.float32, f"(t) must be fp32, got {out.dtype}"
        return out

