"""Advanced numerical integrators for SPOT solver.

This module provides multiple integration schemes:
- Heun's method (2nd order Runge-Kutta)
- DDIM-style deterministic sampling
- Adaptive step size control
- Exponential integrators
- Stochastic SDE samplers
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch

from .constants import EPSILON_CLAMP
from .logger import logger
from .schedules import NoiseScheduleProtocol

def _beta_from_schedule(schedule: NoiseScheduleProtocol, t: float) -> float:
    """Get beta(t) from schedule using analytical method.

    This replaces the previous finite-difference approximation with direct
    analytical computation for numerical stability and efficiency.

    Args:
        schedule: Noise schedule with beta() method
        t: Time point

    Returns:
        beta(t) as a Python float
    """
    # Use analytical beta() method for stability and efficiency
    t_clamped = max(0.0, min(1.0, float(t)))
    beta_tensor = schedule.beta(t_clamped)
    beta_val = float(beta_tensor.item() if hasattr(beta_tensor, 'item') else beta_tensor)

    # Sanity check
    if not math.isfinite(beta_val) or beta_val < 0.0:
        logger.warning(f"Invalid beta(t={t:.3f})={beta_val}, clamping to 0")
        return 0.0

    return beta_val

__all__ = [
    "HeunIntegrator",
    "DDIMIntegrator",
    "AdaptiveIntegrator",
    "ExponentialIntegrator",  # DEPRECATED: Use EulerIntegrator
    "EulerIntegrator",
]


class HeunIntegrator:
    """Heun's method (improved Euler) - 2nd order Runge-Kutta.

    Provides better accuracy than Euler with only one additional function evaluation.
    Uses predictor-corrector structure: predict with Euler, correct with trapezoidal rule.
    """

    def __init__(self, schedule: NoiseScheduleProtocol):
        self.schedule = schedule

    def step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    ) -> torch.Tensor:
        """Take one integration step using Heun's method.

        Args:
            x: Current state
            t_curr: Current time
            t_next: Next time
            score_fn: Function that computes score given (x, t)

        Returns:
            Next state x_{t+1}
        """
        dt = t_next - t_curr

        # Compute score at current time
        score_curr = score_fn(x, t_curr)

        # Get noise parameters and beta
        t_curr_tensor = torch.full((1,), t_curr, device=x.device, dtype=torch.float32)
        t_next_tensor = torch.full((1,), t_next, device=x.device, dtype=torch.float32)

        beta_curr = _beta_from_schedule(self.schedule, t_curr)
        beta_next = _beta_from_schedule(self.schedule, t_next)

        # Probability-flow ODE drift: -0.5*beta*x - beta*score
        drift_curr = -0.5 * beta_curr * x - beta_curr * score_curr
        x_pred = x + drift_curr * dt

        # Corrector step (evaluate score at predicted point)
        score_next = score_fn(x_pred, t_next)
        drift_next = -0.5 * beta_next * x_pred - beta_next * score_next

        # Trapezoidal rule (average of drifts)
        x_next = x + 0.5 * (drift_curr + drift_next) * dt

        return x_next


class DDIMIntegrator:
    """DDIM-style deterministic sampling with configurable eta parameter.

    DDIM (Denoising Diffusion Implicit Models) provides deterministic sampling
    by removing the stochastic component. Can interpolate between DDPM (eta=1)
    and fully deterministic (eta=0).

    IMPORTANT: This implementation assumes score_fn returns the SCORE ∇ log p_t(x),
    NOT the noise prediction ε. The relationship is: ε = -σ * score.
    """

    def __init__(
        self,
        schedule: NoiseScheduleProtocol,
        eta: float = 0.0,
    ):
        """Initialize DDIM integrator.

        Args:
            schedule: Noise schedule
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
        """
        self.schedule = schedule
        self.eta = eta

    def step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Take one DDIM step.

        Mathematical derivation:
            Given x_t = α_t * x_0 + σ_t * ε, where ε = -σ_t * score
            We have: x_t = α_t * x_0 - σ_t² * score
            Therefore: x_0 = (x_t + σ_t² * score) / α_t

        Args:
            x: Current state
            t_curr: Current time
            t_next: Next time
            score_fn: Function that computes score ∇ log p_t(x)
            generator: Optional random generator for stochastic component

        Returns:
            Next state
        """
        # Get noise parameters
        t_curr_tensor = torch.full((1,), t_curr, device=x.device, dtype=torch.float32)
        t_next_tensor = torch.full((1,), t_next, device=x.device, dtype=torch.float32)

        alpha_curr, sigma_curr = self.schedule.alpha_sigma(t_curr_tensor)
        alpha_next, sigma_next = self.schedule.alpha_sigma(t_next_tensor)

        # Compute score
        score = score_fn(x, t_curr)

        # Predict x0 (denoised sample)
        # FIXED: Correct formula for score (not noise)
        # x_0 = (x_t + σ_t² * score) / α_t
        with torch.amp.autocast(device_type='cuda' if x.is_cuda else 'cpu', enabled=False):
            sigma_curr_32 = sigma_curr.float()
            sigma_curr_sq = sigma_curr_32 ** 2
            alpha_curr_32 = alpha_curr.float()

            # Correct formula: x0 = (x + sigma^2 * score) / alpha
            pred_x0 = (x.float() + sigma_curr_sq.to(x.device) * score.float()) / (alpha_curr_32.to(x.device) + EPSILON_CLAMP)
            pred_x0 = pred_x0.to(x.dtype)

        # Compute direction pointing to x_t
        # The noise component is ε = -σ * score
        sigma_next_sq = (sigma_next.to(x.dtype) ** 2)
        direction = -sigma_next_sq * score

        # DDIM sampling formula: x_{t+1} = α_{t+1} * x_0 + σ_{t+1} * ε
        x_next = alpha_next.to(x.dtype) * pred_x0 + direction

        # Add stochastic component if eta > 0
        if self.eta > 0 and t_next > 0:
            sigma_eta = (
                self.eta
                * ((sigma_next ** 2 - sigma_curr ** 2 * (sigma_next / sigma_curr) ** 2).sqrt())
            ).to(x.dtype)

            if generator is not None:
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            else:
                noise = torch.randn_like(x)

            x_next = x_next + sigma_eta * noise

        return x_next


class AdaptiveIntegrator:
    """Adaptive step size integrator with error estimation.

    Uses embedded Runge-Kutta method (Dormand-Prince style) to estimate
    local truncation error and adjust step sizes accordingly.
    """

    def __init__(
        self,
        schedule: NoiseScheduleProtocol,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        max_steps: int = 1000,
    ):
        """Initialize adaptive integrator.

        Args:
            schedule: Noise schedule
            atol: Absolute tolerance for error control
            rtol: Relative tolerance for error control
            max_steps: Maximum number of steps
        """
        self.schedule = schedule
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps

    def step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    ) -> torch.Tensor:
        """Take a single adaptive step from t_curr to t_next.

        This method provides a single-step interface compatible with other integrators.
        It uses adaptive step size internally but ensures we end at exactly t_next.
        Handles both forward (t_curr < t_next) and backward (t_curr > t_next) integration.

        Args:
            x: Current state
            t_curr: Current time
            t_next: Next time
            score_fn: Score function

        Returns:
            Next state x_{t+1}
        """
        # Use integrate with the full step
        x_next, _ = self.integrate(x, t_curr, t_next, score_fn)
        return x_next

    def integrate(
        self,
        x0: torch.Tensor,
        t_start: float,
        t_end: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Integrate from t_start to t_end with adaptive step size.

        Handles both forward (t_start < t_end) and backward (t_start > t_end) integration.
        Backward integration is typical for diffusion sampling (t: 1 -> 0).

        Args:
            x0: Initial state
            t_start: Start time
            t_end: End time
            score_fn: Score function

        Returns:
            Tuple of (final state, number of steps taken)
        """
        x = x0
        t = t_start
        steps = 0

        # Determine direction of integration
        direction = 1.0 if t_end > t_start else -1.0
        t_distance = abs(t_end - t_start)

        # Initial step size (always positive, direction handled separately)
        dt = t_distance / 10.0

        # Loop until we reach t_end
        while abs(t - t_end) > 1e-9 and steps < self.max_steps:
            # Don't overshoot - ensure we stop exactly at t_end
            dt = min(dt, abs(t_end - t))

            # Signed step size for actual integration
            dt_signed = dt * direction

            # Take step with error estimation
            x_next, error, dt_next = self._step_with_error(x, t, dt_signed, score_fn)

            # Check if error is acceptable
            if error < self.atol or error < self.rtol * torch.norm(x).item():
                # Accept step
                x = x_next
                t = t + dt_signed
                steps += 1

                # Adjust step size for next iteration (keep unsigned)
                if error > 0:
                    # Standard step size controller with safety factors
                    dt = abs(dt_next)
                else:
                    # If error is very small, increase step size
                    dt = dt * 1.5
            else:
                # Reject step and reduce step size
                dt = abs(dt_next)

            # Safety bounds on step size (unsigned)
            dt = max(1e-6, min(dt, t_distance / 2.0))

        if steps >= self.max_steps:
            logger.warning(f"Adaptive integrator reached max_steps={self.max_steps}")

        return x, steps

    def _step_with_error(
        self,
        x: torch.Tensor,
        t: float,
        dt: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    ) -> Tuple[torch.Tensor, float, float]:
        """Take one step with error estimation using embedded RK method.

        Returns:
            Tuple of (next state, error estimate, suggested next dt)
        """
        # Probability-flow drift helper
        def drift(x_val, t_val):
            beta = _beta_from_schedule(self.schedule, float(t_val))
            score = score_fn(x_val, t_val)
            return -0.5 * beta * x_val - beta * score

        # Simple embedded RK method (Heun with Euler comparison)
        k1 = drift(x, t)

        # Heun prediction
        x_heun = x + k1 * dt
        k2 = drift(x_heun, t + dt)
        x_next = x + 0.5 * (k1 + k2) * dt

        # Euler prediction
        x_euler = x + k1 * dt

        # Error estimate (difference between 2nd and 1st order methods)
        error = torch.norm(x_next - x_euler).item()

        # Suggest next step size using standard controller
        if error > EPSILON_CLAMP:
            safety = 0.9
            dt_next = dt * safety * (self.atol / error) ** 0.5
        else:
            dt_next = dt * 1.5

        return x_next, error, dt_next


class EulerIntegrator:
    """Explicit Euler method (1st order) for probability-flow ODE.

    This is the simplest numerical integrator: x_{i+1} = x_i + f(x_i, t_i) * dt
    where f is the drift function of the probability-flow ODE.

    Note: This is a 1st order method with O(dt²) local error. For better accuracy,
    consider HeunIntegrator (2nd order) or AdaptiveIntegrator.
    """

    def __init__(self, schedule: NoiseScheduleProtocol):
        self.schedule = schedule

    def step(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        score_fn: Callable[[torch.Tensor, float], torch.Tensor],
    ) -> torch.Tensor:
        """Take one explicit Euler step.

        Probability-flow ODE drift: f(x,t) = -0.5*beta(t)*x - beta(t)*score(x,t)

        Args:
            x: Current state
            t_curr: Current time
            t_next: Next time
            score_fn: Score function ∇ log p_t(x)

        Returns:
            Next state x_{t+1}
        """
        beta_curr = _beta_from_schedule(self.schedule, t_curr)

        # Compute score once (explicit Euler for PF-ODE)
        score = score_fn(x, t_curr)

        # Explicit Euler update
        dt = t_next - t_curr
        drift = -0.5 * beta_curr * x - beta_curr * score
        x_next = x + drift * dt

        return x_next


class ExponentialIntegrator(EulerIntegrator):
    """DEPRECATED: This is actually Euler method, not an exponential integrator.

    Use EulerIntegrator instead. This class is kept for backward compatibility
    but will be removed in a future version.

    An actual exponential integrator would use exact integration of the linear
    part: exp(-0.5 * beta * dt) * x, which this implementation does NOT do.
    """

    def __init__(self, schedule: NoiseScheduleProtocol):
        logger.warning(
            "ExponentialIntegrator is deprecated and misnamed (it's actually Euler method). "
            "Use EulerIntegrator instead. This class will be removed in a future version."
        )
        super().__init__(schedule)
