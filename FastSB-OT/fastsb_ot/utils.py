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
        division by zero. Default 1e-6 provides safer bounds for FP16:
        σ_min = √(1e-6) = 1e-3, so max score scale ≈ 1000.
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
        clamp: float = 1e-6,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        super().__init__()
        self.noise_model = noise_model
        self.schedule = schedule
        self.clamp = float(clamp)
        self._warned_non_vectorized = False  # Track if we've warned about non-vectorized schedule
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, t: Union[torch.Tensor, float]) -> torch.Tensor:
        """Convert noise prediction to score estimate.

        IMPORTANT: This assumes DDPM/variance-preserving forward process:
            x_t = √(α_bar_t) * x_0 + √(1 - α_bar_t) * ε

        The score-noise relationship is: ∇_x log p_t(x_t) = -ε / √(1 - α_bar_t)

        For other parameterizations (EDM, VE, DDIM with η ≠ 1), this conversion
        may be incorrect. Ensure your noise schedule matches your training setup.
        """
        # Handle empty batch edge case
        if x.shape[0] == 0:
            return torch.empty_like(x)

        eps = self.noise_model(x, t)
        sigma = self._sigma_from_t(t, x)

        # Validate shape compatibility before broadcasting
        if sigma.ndim > 1 and sigma.shape[0] != eps.shape[0] and sigma.shape[0] != 1:
            raise ValueError(
                f"Batch size mismatch: sigma has {sigma.shape[0]} elements, "
                f"but eps has batch size {eps.shape[0]}"
            )

        # Broadcast sigma to match epsilon shape by adding trailing dimensions
        while sigma.ndim < eps.ndim:
            sigma = sigma.unsqueeze(-1)

        # Final check: ensure batch dimensions align
        if sigma.shape[0] != eps.shape[0] and sigma.shape[0] != 1:
            raise ValueError(
                f"After broadcasting, batch dimension mismatch: "
                f"sigma.shape={sigma.shape}, eps.shape={eps.shape}"
            )

        # Compute score with overflow protection for reduced precision
        score = -eps / sigma

        # Protect against overflow in FP16/BF16
        if score.dtype == torch.float16:
            score = torch.clamp(score, -65504, 65504)
        elif score.dtype == torch.bfloat16:
            score = torch.clamp(score, -3.38e38, 3.38e38)

        return score

    # ------------------------------------------------------------------
    def _sigma_from_t(self, t: Union[torch.Tensor, float], ref: torch.Tensor) -> torch.Tensor:
        """Compute sigma from timestep with vectorized schedule evaluation.

        CRITICAL FIX: Eliminated Python list comprehension that caused CPU-GPU sync stall.
        The original code forced GPU→CPU transfer for every timestep, breaking torch.compile
        and causing ~10-50ms overhead per batch.

        New implementation:
        1. Try vectorized schedule call first (tensor in → tensor out)
        2. Fallback to element-wise only if schedule doesn't support tensors
        3. torch.compile compatible when schedule uses tensor ops
        """
        device = ref.device
        dtype = torch.float32

        # Ensure t is a tensor on correct device
        if isinstance(t, torch.Tensor):
            t_tensor = t.detach().to(device=device, dtype=torch.float32)
        else:
            t_tensor = torch.tensor([float(t)], device=device, dtype=torch.float32)

        # FIX: Try vectorized schedule call first (GPU-friendly)
        try:
            # Most schedules (cos, linear) can operate on tensors directly
            alpha_tensor = self.schedule(t_tensor)

            # Ensure tensor is on correct device and dtype
            if not isinstance(alpha_tensor, torch.Tensor):
                # Schedule returned scalar, wrap as tensor
                alpha_tensor = torch.tensor(alpha_tensor, device=device, dtype=dtype)
            else:
                alpha_tensor = alpha_tensor.to(device=device, dtype=dtype)

        except (TypeError, RuntimeError, AttributeError) as e:
            # FALLBACK: Schedule doesn't support tensors (e.g., uses math.cos instead of torch.cos)
            # This path forces CPU sync but is only taken for non-vectorized schedules
            # Warn only once per instance to avoid log spam
            if not self._warned_non_vectorized:
                import warnings
                warnings.warn(
                    f"Schedule function does not support tensor inputs ({type(e).__name__}). "
                    f"Falling back to slow element-wise evaluation. "
                    f"For best performance, ensure your schedule uses torch.* functions instead of math.* "
                    f"(e.g., torch.cos instead of math.cos). "
                    f"This warning will only appear once.",
                    UserWarning,
                    stacklevel=3
                )
                self._warned_non_vectorized = True

            # Single GPU→CPU transfer, then apply schedule on CPU
            flat = t_tensor.reshape(-1)
            t_cpu = flat.cpu()  # Single sync: move entire batch to CPU

            # Apply schedule element-wise on CPU (no additional syncs)
            alpha_values = [float(self.schedule(float(t_cpu[i]))) for i in range(t_cpu.shape[0])]

            # Move result back to GPU as single tensor
            alpha_tensor = torch.tensor(alpha_values, device=device, dtype=dtype)
            alpha_tensor = alpha_tensor.reshape(t_tensor.shape)

        # Validate that schedule returned valid α_bar values in [0, 1]
        # This catches bugs in the schedule function that could lead to NaN
        if torch.any(alpha_tensor < 0) or torch.any(alpha_tensor > 1):
            raise ValueError(
                f"Schedule returned invalid α_bar values outside [0, 1]. "
                f"Got min={alpha_tensor.min().item():.6f}, max={alpha_tensor.max().item():.6f}. "
                f"Check your noise schedule implementation."
            )

        sigma = torch.sqrt(torch.clamp(1.0 - alpha_tensor, min=self.clamp))

        return sigma.to(ref.dtype if ref.dtype.is_floating_point else torch.float32)


def wrap_noise_predictor(
    noise_model: nn.Module,
    schedule: Callable[[float], float],
    *,
    clamp: float = 1e-6,
    device: Union[torch.device, str, None] = None,
) -> NoisePredictorToScoreWrapper:
    """Factory helper mirroring :class:`NoisePredictorToScoreWrapper` construction."""

    return NoisePredictorToScoreWrapper(noise_model, schedule, clamp=clamp, device=device)


