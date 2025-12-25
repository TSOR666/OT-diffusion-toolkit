"""Corrector steps for predictor-corrector sampling in SPOT.

This module implements various corrector schemes that can be applied after
predictor steps to refine samples and reduce discretization errors.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import torch

from .logger import logger
from .schedules import NoiseScheduleProtocol

__all__ = [
    "LangevinCorrector",
    "TweedieCorrector",
    "AdaptiveCorrector",
    "PredictorCorrectorSampler",
]


class PredictorProtocol(Protocol):
    def step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        ...


class CorrectorProtocol(Protocol):
    def correct(
        self,
        x: torch.Tensor,
        t: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...


class LangevinCorrector:
    """Langevin dynamics corrector using score-based MCMC.

    Applies n_steps of Langevin MCMC to refine the sample by following
    the score function more closely. This can reduce discretization errors
    from the predictor step.

    Mathematical Note:
        The Langevin update is: x_{i+1} = x_i + epsilon * score + sqrt(2*epsilon) * z
        where epsilon = 2 * (snr * sigma)^2 is the step size.

        This formulation assumes:
        1. score_fn returns the UNNORMALIZED score: ∇ log p_t(x)
        2. For Gaussian perturbations with variance sigma^2, the score scales as ~x/sigma^2
        3. The step size epsilon scales with sigma^2 to maintain proper SNR dynamics

        CRITICAL: If your score model returns a NORMALIZED score (e.g., sigma * ∇ log p_t(x)
        as in some latent diffusion models), this step size will be incorrect by a factor
        of sigma^2. Verify your score_fn output convention!

        Reference: Song et al. "Score-Based Generative Modeling through SDEs" (2021)
    """

    def __init__(
        self,
        schedule: NoiseScheduleProtocol,
        n_steps: int = 1,
        snr: float = 0.16,
        denoise: bool = False,
    ) -> None:
        """Initialize Langevin corrector.

        Args:
            schedule: Noise schedule
            n_steps: Number of Langevin steps per correction
            snr: Signal-to-noise ratio for step size (typical: 0.05-0.2)
                Higher SNR = larger steps = faster mixing but less accuracy
            denoise: Whether to perform final denoising step (no noise injection)
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
            Refined sample with same shape as ``x`` (B, *S) -> (B, *S)
        """
        if self.n_steps == 0:
            return x

        # Get noise level
        t_tensor = torch.full((1,), t, device=x.device, dtype=torch.float32)  # (1,)
        _, sigma_t = self.schedule.alpha_sigma(t_tensor)  # (1,)
        sigma = sigma_t.item()

        # Langevin step size based on SNR
        # step_size = 2 * (snr * sigma)^2
        step_size = 2.0 * (self.snr * sigma) ** 2

        x_current = x

        for i in range(self.n_steps):
            # Compute score
            score = score_fn(x_current, t)  # (B, *S)

            # Generate noise
            if generator is not None:
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)  # (B, *S)
            else:
                noise = torch.randn_like(x)  # (B, *S)

            # Langevin update: x_{i+1} = x_i + step_size * score + sqrt(2 * step_size) * noise
            if self.denoise and i == self.n_steps - 1:
                # Final step: no noise (denoising)
                x_current = x_current + step_size * score  # (B, *S)
            else:
                # Regular Langevin step with noise
                noise_scale = torch.sqrt(torch.tensor(2.0 * step_size, device=x.device, dtype=x.dtype))  # ()
                x_current = x_current + step_size * score + noise_scale * noise  # (B, *S)

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
    ) -> None:
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
            Corrected sample with same shape as ``x`` (B, *S) -> (B, *S)
        """
        # Get noise parameters
        t_tensor = torch.full((1,), t, device=x.device, dtype=torch.float32)  # (1,)
        alpha_t, sigma_t = self.schedule.alpha_sigma(t_tensor)  # (1,), (1,)

        # Safety check: alpha should never be too close to 0
        # This can happen at t → T (pure noise regime)
        alpha_min = 1e-3
        if alpha_t.item() < alpha_min:
            logger.warning(
                f"TweedieCorrector: alpha_t={alpha_t.item():.2e} < {alpha_min} at t={t:.3f}. "
                f"Tweedie correction is unstable in high-noise regime. Skipping correction."
            )
            return x

        # Compute score
        score = score_fn(x, t)  # (B, *S)

        # Tweedie's formula: estimate x0
        # Disable autocast to avoid precision issues in division
        with torch.amp.autocast(device_type='cuda' if x.is_cuda else 'cpu', enabled=False):
            sigma_sq = sigma_t.float() ** 2  # (1,)
            alpha_float = alpha_t.float()  # (1,)

            # x0_estimate = (x_t - sigma^2 * score) / alpha
            # Use clamp instead of epsilon addition for more principled stability
            alpha_clamped = torch.clamp(alpha_float, min=alpha_min)  # (1,)
            x0_estimate = (x.float() - sigma_sq * score.float()) / alpha_clamped  # (B, *S)
            x0_estimate = x0_estimate.to(x.dtype)

        # Mix with original sample
        x_corrected = (1 - self.mixing) * x + self.mixing * (
            alpha_t.to(x.dtype) * x0_estimate  # (B, *S)
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
    ) -> None:
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
            Corrected sample with same shape as ``x`` (B, *S) -> (B, *S)
        """
        self.total_calls += 1

        # Estimate error if we have previous sample
        if x_prev is not None:
            # Simple error estimate based on change magnitude
            change = torch.norm(x - x_prev) / (torch.norm(x) + 1e-8)  # scalar
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

            # More efficient: call Langevin corrector once with multiple steps
            # instead of looping (avoids redundant overhead)
            original_n_steps = self.langevin_corrector.n_steps
            self.langevin_corrector.n_steps = n_corrections

            x_corrected = self.langevin_corrector.correct(
                x, t, score_fn, generator
            )

            # Restore original n_steps
            self.langevin_corrector.n_steps = original_n_steps

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

    def reset_stats(self) -> None:
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
        predictor: PredictorProtocol,
        corrector: Optional[CorrectorProtocol],
        schedule: NoiseScheduleProtocol,
        use_corrector: bool = True,
    ) -> None:
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
            Next state after prediction and correction with same shape as ``x`` (B, *S) -> (B, *S)

        Raises:
            TypeError: If predictor does not have a step() method
        """
        # Predictor step
        if not hasattr(self.predictor, 'step'):
            raise TypeError(
                f"Predictor must have a 'step(x, t_curr, t_next, score_fn)' method. "
                f"Got type: {type(self.predictor).__name__}. "
                f"Use an integrator from SPOT.integrators (HeunIntegrator, DDIMIntegrator, etc.) "
                f"or ensure your custom predictor implements the step() interface."
            )

        x_pred = self.predictor.step(x, t_curr, t_next, score_fn)  # (B, *S)

        # Corrector step
        if self.use_corrector and self.corrector is not None:
            if isinstance(self.corrector, AdaptiveCorrector):
                x_corrected = self.corrector.correct(
                    x_pred, t_next, score_fn, x_prev=x, generator=generator
                )  # (B, *S)
            else:
                x_corrected = self.corrector.correct(
                    x_pred, t_next, score_fn, generator=generator
                )  # (B, *S)
        else:
            x_corrected = x_pred

        return x_corrected
