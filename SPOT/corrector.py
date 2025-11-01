"""Corrector steps for predictor-corrector sampling in SPOT.

This module implements various corrector schemes that can be applied after
predictor steps to refine samples and reduce discretization errors.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .logger import logger
from .schedules import NoiseScheduleProtocol

__all__ = [
    "LangevinCorrector",
    "TweedieCorrector",
    "AdaptiveCorrector",
]


class LangevinCorrector:
    """Langevin dynamics corrector using score-based MCMC.

    Applies n_steps of Langevin MCMC to refine the sample by following
    the score function more closely. This can reduce discretization errors
    from the predictor step.
    """

    def __init__(
        self,
        schedule: NoiseScheduleProtocol,
        n_steps: int = 1,
        snr: float = 0.16,
        denoise: bool = False,
    ):
        """Initialize Langevin corrector.

        Args:
            schedule: Noise schedule
            n_steps: Number of Langevin steps per correction
            snr: Signal-to-noise ratio for step size (typical: 0.05-0.2)
            denoise: Whether to perform final denoising step
        """
        self.schedule = schedule
        self.n_steps = n_steps
        self.snr = snr
        self.denoise = denoise

    def correct(
        self,
        x: torch.Tensor,
        t: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply Langevin correction to refine sample.

        Args:
            x: Current sample to refine
            t: Current time
            score_fn: Function that computes score
            generator: Optional random generator

        Returns:
            Refined sample
        """
        if self.n_steps == 0:
            return x

        # Get noise level
        t_tensor = torch.full((1,), t, device=x.device, dtype=torch.float32)
        _, sigma_t = self.schedule.alpha_sigma(t_tensor)
        sigma = sigma_t.item()

        # Langevin step size based on SNR
        # step_size = 2 * (snr * sigma)^2
        step_size = 2.0 * (self.snr * sigma) ** 2

        x_current = x

        for i in range(self.n_steps):
            # Compute score
            score = score_fn(x_current, t)

            # Generate noise
            if generator is not None:
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            else:
                noise = torch.randn_like(x)

            # Langevin update: x_{i+1} = x_i + step_size * score + sqrt(2 * step_size) * noise
            if self.denoise and i == self.n_steps - 1:
                # Final step: no noise (denoising)
                x_current = x_current + step_size * score
            else:
                # Regular Langevin step with noise
                x_current = x_current + step_size * score + torch.sqrt(
                    torch.tensor(2.0 * step_size, device=x.device)
                ) * noise

        return x_current


class TweedieCorrector:
    """Tweedie's formula corrector for denoising.

    Uses Tweedie's formula to estimate the clean signal from the noisy observation,
    then mixes it with the original sample. This provides a different trade-off
    between exploration and exploitation compared to Langevin dynamics.
    """

    def __init__(
        self,
        schedule: NoiseScheduleProtocol,
        mixing: float = 0.5,
    ):
        """Initialize Tweedie corrector.

        Args:
            schedule: Noise schedule
            mixing: Mixing coefficient (0=no correction, 1=full denoising)
        """
        self.schedule = schedule
        self.mixing = mixing

    def correct(
        self,
        x: torch.Tensor,
        t: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    ) -> torch.Tensor:
        """Apply Tweedie correction.

        Tweedie's formula states that for x_t ~ N(alpha*x0, sigma^2*I):
        E[x0 | x_t] = (x_t - sigma^2 * score(x_t, t)) / alpha

        Args:
            x: Current sample
            t: Current time
            score_fn: Score function

        Returns:
            Corrected sample
        """
        # Get noise parameters
        t_tensor = torch.full((1,), t, device=x.device, dtype=torch.float32)
        alpha_t, sigma_t = self.schedule.alpha_sigma(t_tensor)

        # Compute score
        score = score_fn(x, t)

        # Tweedie's formula: estimate x0
        with torch.amp.autocast(device_type='cuda' if x.is_cuda else 'cpu', enabled=False):
            sigma_sq = sigma_t.float() ** 2
            alpha_float = alpha_t.float()

            # x0_estimate = (x_t - sigma^2 * score) / alpha
            x0_estimate = (x.float() - sigma_sq * score.float()) / (alpha_float + 1e-8)
            x0_estimate = x0_estimate.to(x.dtype)

        # Mix with original sample
        x_corrected = (1 - self.mixing) * x + self.mixing * (
            alpha_t.to(x.dtype) * x0_estimate
        )

        return x_corrected


class AdaptiveCorrector:
    """Adaptive corrector that chooses correction strategy based on error estimate.

    Monitors the local error and applies correction only when needed, which can
    save computation while maintaining quality.
    """

    def __init__(
        self,
        schedule: NoiseScheduleProtocol,
        error_threshold: float = 0.1,
        langevin_snr: float = 0.16,
        max_corrections: int = 3,
    ):
        """Initialize adaptive corrector.

        Args:
            schedule: Noise schedule
            error_threshold: Threshold for applying correction
            langevin_snr: SNR for Langevin steps
            max_corrections: Maximum number of correction steps
        """
        self.schedule = schedule
        self.error_threshold = error_threshold
        self.langevin_corrector = LangevinCorrector(
            schedule, n_steps=1, snr=langevin_snr
        )
        self.max_corrections = max_corrections
        self.correction_count = 0
        self.total_calls = 0

    def correct(
        self,
        x: torch.Tensor,
        t: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
        x_prev: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply adaptive correction based on estimated error.

        Args:
            x: Current sample
            t: Current time
            score_fn: Score function
            x_prev: Previous sample (for error estimation)
            generator: Optional random generator

        Returns:
            Corrected sample
        """
        self.total_calls += 1

        # Estimate error if we have previous sample
        if x_prev is not None:
            # Simple error estimate based on change magnitude
            change = torch.norm(x - x_prev) / (torch.norm(x) + 1e-8)
            error = change.item()
        else:
            # No previous sample, use moderate error estimate
            error = self.error_threshold * 1.5

        # Apply correction if error exceeds threshold
        if error > self.error_threshold:
            # Determine number of correction steps based on error magnitude
            n_corrections = min(
                int((error / self.error_threshold) ** 0.5),
                self.max_corrections
            )

            x_corrected = x
            for _ in range(n_corrections):
                x_corrected = self.langevin_corrector.correct(
                    x_corrected, t, score_fn, generator
                )

            self.correction_count += 1

            return x_corrected
        else:
            # Error is acceptable, no correction needed
            return x

    def get_stats(self) -> dict:
        """Get statistics about correction usage.

        Returns:
            Dictionary with correction statistics
        """
        if self.total_calls == 0:
            correction_rate = 0.0
        else:
            correction_rate = self.correction_count / self.total_calls

        return {
            "total_calls": self.total_calls,
            "correction_count": self.correction_count,
            "correction_rate": correction_rate,
        }

    def reset_stats(self):
        """Reset correction statistics."""
        self.correction_count = 0
        self.total_calls = 0


class PredictorCorrectorSampler:
    """Unified predictor-corrector sampler.

    Combines any predictor (integrator) with any corrector for flexible
    sampling strategies.
    """

    def __init__(
        self,
        predictor,
        corrector,
        schedule: NoiseScheduleProtocol,
        use_corrector: bool = True,
    ):
        """Initialize predictor-corrector sampler.

        Args:
            predictor: Predictor integrator
            corrector: Corrector object
            schedule: Noise schedule
            use_corrector: Whether to apply corrector steps
        """
        self.predictor = predictor
        self.corrector = corrector
        self.schedule = schedule
        self.use_corrector = use_corrector

    def step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Take one predictor-corrector step.

        Args:
            x: Current state
            t_curr: Current time
            t_next: Next time
            score_fn: Score function
            generator: Optional random generator

        Returns:
            Next state after prediction and correction
        """
        # Predictor step
        if hasattr(self.predictor, 'step'):
            x_pred = self.predictor.step(x, t_curr, t_next, score_fn)
        else:
            # Fallback for simple drift functions
            dt = t_next - t_curr
            drift = score_fn(x, t_curr)
            t_tensor = torch.full((1,), t_curr, device=x.device, dtype=torch.float32)
            _, sigma = self.schedule.alpha_sigma(t_tensor)
            x_pred = x + 0.5 * drift * sigma.to(x.dtype) * dt

        # Corrector step
        if self.use_corrector and self.corrector is not None:
            if isinstance(self.corrector, AdaptiveCorrector):
                x_corrected = self.corrector.correct(
                    x_pred, t_next, score_fn, x_prev=x, generator=generator
                )
            else:
                x_corrected = self.corrector.correct(
                    x_pred, t_next, score_fn, generator=generator
                )
        else:
            x_corrected = x_pred

        return x_corrected
