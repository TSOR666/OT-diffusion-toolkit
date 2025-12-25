"""Utility helpers for adapting models to FastSB-OT."""

from __future__ import annotations

from typing import Callable, Protocol, Union

import torch
import torch.nn as nn

from . import common

check_tensor_finite = common.check_tensor_finite
nan_checks_enabled = common.nan_checks_enabled
__all__ = [
    "NoisePredictorToScoreWrapper",
    "wrap_noise_predictor",
]


class NoiseModel(Protocol):
    def __call__(self, x: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
        ...


ScheduleFn = Callable[[Union[torch.Tensor, float]], torch.Tensor]


class NoisePredictorToScoreWrapper(nn.Module):
    """Wrap a noise-predicting model so it returns score estimates."""

    predicts_score: bool = True
    predicts_noise: bool = False

    def __init__(
        self,
        noise_model: NoiseModel,
        schedule: ScheduleFn,
        *,
        clamp: float = 1e-8,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        super().__init__()
        self.noise_model = noise_model
        self.schedule = schedule
        self.clamp = float(clamp)
        self._warned_non_vectorized = False
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
        """Return score estimate from a noise-predicting model.

        Shapes:
            - x: (B, ...) arbitrary spatial/feature shape
            - t: scalar or (B,)
            - return: same shape as x
        """
        if nan_checks_enabled(None):
            check_tensor_finite("x", x, enabled=True)
        eps = self.noise_model(x, t)  # (B, ...) -> (B, ...)
        sigma = self._sigma_from_t(t, x)  # (B,) or scalar -> (B,)

        # Broadcast sigma to match epsilon shape.
        while sigma.ndim < eps.ndim:
            sigma = sigma.unsqueeze(-1)  # (B,) -> (B, 1, ..., 1)

        # Guard against extreme scores near clean data (sigma -> 0).
        sigma_clamped = torch.clamp(sigma, min=1e-4)  # (B, 1, ..., 1) -> (B, 1, ..., 1)
        return -eps / sigma_clamped  # (B, ...) / (B, 1, ..., 1) broadcast -> (B, ...)

    # ------------------------------------------------------------------
    def _sigma_from_t(self, t: Union[torch.Tensor, float], ref: torch.Tensor) -> torch.Tensor:
        """Compute sigma with improved numerical stability.

        Uses vectorized schedule evaluation when available, with FP64 subtraction
        to avoid catastrophic cancellation near alpha_bar ≈ 1.
        """
        device = ref.device

        if isinstance(t, torch.Tensor):
            t_tensor = t.detach().to(device=device, dtype=torch.float32)  # (B,) -> (B,)
        else:
            t_tensor = torch.tensor([float(t)], device=device, dtype=torch.float32)  # () -> (1,)

        try:
            alpha_tensor = self.schedule(t_tensor)
            if not isinstance(alpha_tensor, torch.Tensor):
                alpha_tensor = torch.tensor(alpha_tensor, device=device, dtype=torch.float32)  # -> (B,)
            else:
                alpha_tensor = alpha_tensor.to(device=device, dtype=torch.float32)  # (B,) -> (B,)
        except (TypeError, RuntimeError, AttributeError) as e:
            if not self._warned_non_vectorized:
                import warnings

                warnings.warn(
                    f"Schedule function does not support tensor inputs ({type(e).__name__}). "
                    "Falling back to slow element-wise evaluation. "
                    "For best performance, ensure your schedule uses torch.* functions instead of math.*. "
                    "This warning will only appear once.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_non_vectorized = True

            flat = t_tensor.reshape(-1)  # (B, ...) -> (T,)
            t_cpu = flat.cpu()
            alpha_values = [float(self.schedule(float(t_cpu[i]))) for i in range(t_cpu.shape[0])]
            alpha_tensor = torch.tensor(alpha_values, device=device, dtype=torch.float32).reshape(t_tensor.shape)  # (T,) -> t_tensor.shape

        if torch.any(alpha_tensor < 0) or torch.any(alpha_tensor > 1):
            raise ValueError(
                f"Schedule returned invalid alpha_bar values outside [0, 1]. "
                f"Got min={alpha_tensor.min().item():.6f}, max={alpha_tensor.max().item():.6f}."
            )

        alpha_tensor = alpha_tensor.to(torch.float64)  # (B,) -> (B,)
        one_minus_alpha = torch.clamp(1.0 - alpha_tensor, min=self.clamp)  # scalar - (B,) broadcast -> (B,)
        sigma = torch.sqrt(one_minus_alpha).to(torch.float32)  # (B,) -> (B,)

        # Ensure sigma can broadcast to eps shape
        if sigma.ndim > 0 and sigma.shape != t_tensor.shape:
            sigma = sigma.reshape(t_tensor.shape)  # (...) -> t_tensor.shape
        return sigma.to(ref.dtype if ref.dtype.is_floating_point else torch.float32)  # (B,) -> (B,)


def wrap_noise_predictor(
    noise_model: NoiseModel,
    schedule: ScheduleFn,
    *,
    clamp: float = 1e-8,
    device: Union[torch.device, str, None] = None,
) -> NoisePredictorToScoreWrapper:
    """Factory helper mirroring :class:`NoisePredictorToScoreWrapper` construction."""

    return NoisePredictorToScoreWrapper(noise_model, schedule, clamp=clamp, device=device)
