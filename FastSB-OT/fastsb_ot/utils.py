"""Utility helpers for adapting models to FastSB-OT."""

from __future__ import annotations

from typing import Callable, Union

import torch
import torch.nn as nn

__all__ = [
    "NoisePredictorToScoreWrapper",
    "wrap_noise_predictor",
]


class NoisePredictorToScoreWrapper(nn.Module):
    """Wrap a noise-predicting model so it returns score estimates.

    Many diffusion models are trained to predict the added noise (``epsilon``).
    FastSB-OT, however, consumes score estimates ``_x log p_t(x)``. This wrapper
    converts epsilon predictions into scores using the provided noise schedule
    ``alpha_bar(t)`` so the solver can be used with either type of model.

    Parameters
    ----------
    noise_model:
        Neural network that predicts noise ``epsilon`` for inputs ``(x, t)``.
    schedule:
        Callable returning ``alpha_bar(t)`` for ``t  [0, 1]``  typically obtained
        through :func:`fastsb_ot.solver.make_schedule` or a custom schedule that
        matches training.
    clamp:
        Minimum value for ``1 - alpha_bar`` when computing ``sigma`` to avoid
        division by zero.
    device:
        Optional device hint; if provided we eagerly move the wrapper to the device.
    """

    predicts_score: bool = True
    predicts_noise: bool = False

    def __init__(
        self,
        noise_model: nn.Module,
        schedule: Callable[[float], float],
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
        eps = self.noise_model(x, t)
        sigma = self._sigma_from_t(t, x)

        # Broadcast sigma to match epsilon shape.
        while sigma.ndim < eps.ndim:
            sigma = sigma.unsqueeze(-1)

        return -eps / sigma

    # ------------------------------------------------------------------
    def _sigma_from_t(self, t: Union[torch.Tensor, float], ref: torch.Tensor) -> torch.Tensor:
        """Compute sigma with improved numerical stability.

        CRITICAL FIX: Compute 1 - alpha in FP64 to avoid catastrophic cancellation
        when alpha_bar ≈ 1 (near t=0, clean samples).
        """
        device = ref.device
        dtype = torch.float32

        if isinstance(t, torch.Tensor):
            t_tensor = t.detach().to(device=device, dtype=torch.float32)
        else:
            t_tensor = torch.tensor([float(t)], device=device, dtype=torch.float32)

        flat = t_tensor.reshape(-1)
        alpha_vals = [float(self.schedule(float(v))) for v in flat]
        alpha_tensor = torch.tensor(alpha_vals, device=device, dtype=torch.float64).reshape(t_tensor.shape)

        # Compute 1 - alpha in FP64 to preserve precision near alpha=1
        one_minus_alpha = torch.clamp(1.0 - alpha_tensor, min=self.clamp)
        sigma = torch.sqrt(one_minus_alpha).to(torch.float32)

        return sigma.to(ref.dtype if ref.dtype.is_floating_point else torch.float32)


def wrap_noise_predictor(
    noise_model: nn.Module,
    schedule: Callable[[float], float],
    *,
    clamp: float = 1e-8,
    device: Union[torch.device, str, None] = None,
) -> NoisePredictorToScoreWrapper:
    """Factory helper mirroring :class:`NoisePredictorToScoreWrapper` construction."""

    return NoisePredictorToScoreWrapper(noise_model, schedule, clamp=clamp, device=device)


