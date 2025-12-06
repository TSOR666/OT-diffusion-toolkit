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

    def beta(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute beta(t) = -d/dt lambda(t) analytically.

        This is the noise rate schedule used in SDE/ODE formulations.
        Should be implemented analytically for numerical stability.
        """
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

    def beta(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Analytical beta(t) = -d/dt lambda(t) for cosine schedule.

        Derivation:
            f(t) = cos²(π/2 * (t+s)/(1+s))
            λ(t) = log(f/(1-f))
            dλ/dt = (df/dt) / (f(1-f))
            df/dt = -sin(2θ) * π/(2(1+s)) where θ = π/2 * (t+s)/(1+s)
            β(t) = -dλ/dt

        Returns:
            beta(t) as float32 tensor
        """
        t32 = self._ensure_tensor(t).clamp(0.0, 1.0)

        # Compute angle θ = π/2 * (t+s)/(1+s)
        theta = (t32 + self.s) / (1 + self.s) * math.pi / 2

        # f(t) = cos²(θ)
        cos_theta = torch.cos(theta)
        f = cos_theta ** 2

        # df/dt = -sin(2θ) * π/(2(1+s))
        sin_2theta = torch.sin(2 * theta)
        df_dt = -sin_2theta * math.pi / (2 * (1 + self.s))

        # β(t) = -dλ/dt = -(df/dt) / (f(1-f))
        # Add small epsilon to denominator for numerical stability
        denominator = (f * (1 - f)).clamp_min(EPSILON_CLAMP)
        beta_t = -df_dt / denominator

        # Ensure beta is non-negative (it should be by construction)
        beta_t = beta_t.clamp_min(0.0)

        if __debug__:
            assert beta_t.dtype == torch.float32, f"beta(t) must be fp32, got {beta_t.dtype}"

        return beta_t


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

    def beta(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Analytical -dλ/dt for linear schedule.

        Returns:
            beta(t) = -(d/dt) lambda(t) where lambda(t) = log(alpha^2 / sigma^2)
        """
        t32 = self._ensure_tensor(t).clamp(0.0, 1.0)
        beta_t = self.beta_start + (self.beta_end - self.beta_start) * t32

        # Relation for VP-style schedules: -d lambda/dt = beta(t) / (1 - alpha_bar(t))
        alpha, _ = self.alpha_sigma(t32)
        alpha_bar = (alpha.float() * alpha.float()).clamp(0.0, 1.0)
        denom = (1.0 - alpha_bar).clamp_min(EPSILON_CLAMP)
        beta_lambda = beta_t / denom

        if __debug__:
            assert beta_lambda.dtype == torch.float32, f"beta(t) must be fp32, got {beta_lambda.dtype}"

        return beta_lambda

