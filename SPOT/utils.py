"""Utility helpers for adapting models to SPOT."""

from __future__ import annotations

from typing import Callable, Protocol, Union

import torch
import torch.nn as nn

from .schedules import NoiseScheduleProtocol

__all__ = [
    "NoisePredictorToScoreWrapper",
    "wrap_noise_predictor",
]


class _AlphaSigmaCallable(Protocol):
    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...


ScheduleLike = Union[NoiseScheduleProtocol, _AlphaSigmaCallable]


class NoisePredictorToScoreWrapper(nn.Module):
    """Wrap a noise-predicting model so SPOT can consume it as a score model.

    Parameters
    ----------
    noise_model:
        Module that predicts epsilon for inputs ``(x, t)``.
    schedule:
        Callable returning ``alpha, sigma`` for ``t`` (matching ``SPOT.schedules``).
    clamp:
        Minimum value used when computing ``sigma`` to avoid division by zero.
    device:
        Optional device hint; if provided, the wrapper is moved to that device.
    """

    predicts_score: bool = True
    predicts_noise: bool = False

    def __init__(
        self,
        noise_model: nn.Module,
        schedule: ScheduleLike,
        *,
        clamp: float = 1e-8,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        super().__init__()
        self.noise_model = noise_model
        self.schedule = schedule
        self.clamp = float(clamp)
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
        """Convert predicted noise to score with shape-preserving broadcast."""
        eps = self.noise_model(x, t)  # (B, *S) -> (B, *S)
        sigma = self._sigma_from_t(t, ref=x)  # (1,) or (B,) -> broadcast

        while sigma.ndim < eps.ndim:
            sigma = sigma.unsqueeze(-1)  # (B, ..., 1)

        return -eps / sigma  # (B, *S)

    # ------------------------------------------------------------------
    def _sigma_from_t(self, t: Union[torch.Tensor, float], ref: torch.Tensor) -> torch.Tensor:
        device = ref.device
        if isinstance(t, torch.Tensor):
            t_tensor = t.detach().to(device=device, dtype=torch.float32)  # (B,) or (1,)
        else:
            t_tensor = torch.tensor([float(t)], device=device, dtype=torch.float32)  # (1,)

        if hasattr(self.schedule, "alpha_sigma"):
            _, sigma = self.schedule.alpha_sigma(t_tensor)  # (N,)
        else:
            _, sigma = self.schedule(t_tensor)  # (N,)

        sigma = sigma.to(device=device, dtype=torch.float32)  # (N,)
        sigma = torch.clamp(sigma, min=self.clamp)  # (N,)

        return sigma.to(ref.dtype if ref.dtype.is_floating_point else torch.float32)


def wrap_noise_predictor(
    noise_model: nn.Module,
    schedule: ScheduleLike,
    *,
    clamp: float = 1e-8,
    device: Union[torch.device, str, None] = None,
) -> NoisePredictorToScoreWrapper:
    """Convenience factory for :class:`NoisePredictorToScoreWrapper`."""

    return NoisePredictorToScoreWrapper(noise_model, schedule, clamp=clamp, device=device)
