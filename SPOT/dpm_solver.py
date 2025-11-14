"""DPM-Solver++ implementation used by the SPOT solver."""
from __future__ import annotations

from collections import deque
from typing import List, Union

import torch

from .constants import EPSILON_CLAMP
from .logger import logger
from .schedules import NoiseScheduleProtocol

__all__ = ["DPMSolverPP"]


class DPMSolverPP:
    """DPM-Solver++ with unified schedule semantics."""

    def __init__(self, order: int = 3, schedule: NoiseScheduleProtocol | None = None):
        if order not in [1, 2, 3]:
            raise ValueError(f"DPM-Solver++ order must be 1-3, got {order}")
        self.order = order
        self.schedule = schedule
        logger.debug("Initialized DPM-Solver++ with order %s", order)

    def _ensure_schedule(self) -> NoiseScheduleProtocol:
        if self.schedule is None:
            from .schedules import CosineSchedule  # Local import to avoid cycle

            self.schedule = CosineSchedule()
        return self.schedule

    def get_timesteps(self, num_steps: int, device: torch.device | None = None) -> List[float]:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        if device is None:
            device = torch.device("cpu")

        t = torch.linspace(1, 0, num_steps + 1, device=device)
        if self.order >= 3:
            t = t ** 1.5
        else:
            t = t ** 2

        t[0] = 1.0
        t[-1] = 0.0

        return t.cpu().tolist()

    def multistep_update(
        self,
        x: torch.Tensor,
        model_outputs: Union[list[torch.Tensor], deque],
        timesteps: List[float],
        current_idx: int,
    ) -> torch.Tensor:
        if isinstance(model_outputs, deque):
            model_outputs = list(model_outputs)

        schedule = self._ensure_schedule()

        try:
            if len(model_outputs) == 1 or current_idx == 0:
                return self._first_order_update(x, model_outputs[-1], timesteps, current_idx, schedule)
            if len(model_outputs) == 2 or self.order < 3:
                return self._second_order_update(x, model_outputs[-2:], timesteps, current_idx, schedule)
            return self._third_order_update(x, model_outputs[-3:], timesteps, current_idx, schedule)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("DPM-Solver++ failed: %s, falling back to first order", exc)
            return self._first_order_update(x, model_outputs[-1], timesteps, current_idx, schedule)

    def _first_order_update(
        self,
        x: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: List[float],
        idx: int,
        schedule: NoiseScheduleProtocol,
    ) -> torch.Tensor:
        t_curr, t_next = timesteps[idx], timesteps[idx + 1]

        dev, dtp = x.device, x.dtype
        t_curr_tensor = torch.full((1,), float(t_curr), device=dev, dtype=torch.float32)
        t_next_tensor = torch.full((1,), float(t_next), device=dev, dtype=torch.float32)

        lambda_curr = schedule.lambda_(t_curr_tensor)
        lambda_next = schedule.lambda_(t_next_tensor)
        h = lambda_next - lambda_curr

        alpha_curr, sigma_curr = schedule.alpha_sigma(t_curr_tensor)
        alpha_next, _ = schedule.alpha_sigma(t_next_tensor)

        alpha_ratio = (alpha_next / alpha_curr).to(dtp)
        sigma_term = (sigma_curr * torch.expm1(h)).to(dtp)

        return alpha_ratio * x - sigma_term * model_output

    def _second_order_update(
        self,
        x: torch.Tensor,
        model_outputs: list[torch.Tensor],
        timesteps: List[float],
        idx: int,
        schedule: NoiseScheduleProtocol,
    ) -> torch.Tensor:
        t_prev, t_curr, t_next = timesteps[idx - 1], timesteps[idx], timesteps[idx + 1]

        dev, dtp = x.device, x.dtype
        t_prev_tensor = torch.full((1,), float(t_prev), device=dev, dtype=torch.float32)
        t_curr_tensor = torch.full((1,), float(t_curr), device=dev, dtype=torch.float32)
        t_next_tensor = torch.full((1,), float(t_next), device=dev, dtype=torch.float32)

        lambda_prev = schedule.lambda_(t_prev_tensor)
        lambda_curr = schedule.lambda_(t_curr_tensor)
        lambda_next = schedule.lambda_(t_next_tensor)

        h = lambda_next - lambda_curr
        h_prev = lambda_curr - lambda_prev

        # Guard against numerical instability in timestep ratios
        if abs(h) < EPSILON_CLAMP:
            logger.debug(
                f"DPM-Solver timestep too small (|h|={abs(h):.2e} < {EPSILON_CLAMP:.2e}), "
                f"falling back to first-order update"
            )
            return self._first_order_update(x, model_outputs[-1], timesteps, idx, schedule)

        r = h_prev / h
        if not torch.isfinite(torch.tensor(r)):
            logger.debug(f"DPM-Solver produced non-finite ratio r={r}, falling back to first-order")
            return self._first_order_update(x, model_outputs[-1], timesteps, idx, schedule)

        alpha_curr, sigma_curr = schedule.alpha_sigma(t_curr_tensor)
        alpha_next, _ = schedule.alpha_sigma(t_next_tensor)

        denom = 2 * r
        if abs(denom) < EPSILON_CLAMP:
            logger.debug(f"DPM-Solver denominator too small, falling back to first-order")
            return self._first_order_update(x, model_outputs[-1], timesteps, idx, schedule)

        D1 = (1 + 1 / denom) * model_outputs[-1] - 1 / denom * model_outputs[-2]

        alpha_ratio = (alpha_next / alpha_curr).to(dtp)
        sigma_term = (sigma_curr * torch.expm1(h)).to(dtp)

        return alpha_ratio * x - sigma_term * D1

    def _third_order_update(
        self,
        x: torch.Tensor,
        model_outputs: list[torch.Tensor],
        timesteps: List[float],
        idx: int,
        schedule: NoiseScheduleProtocol,
    ) -> torch.Tensor:
        t_prev2, t_prev, t_curr, t_next = (
            timesteps[idx - 2],
            timesteps[idx - 1],
            timesteps[idx],
            timesteps[idx + 1],
        )

        dev, dtp = x.device, x.dtype
        tensors = [
            torch.full((1,), float(t), device=dev, dtype=torch.float32)
            for t in (t_prev2, t_prev, t_curr, t_next)
        ]
        t_prev2_tensor, t_prev_tensor, t_curr_tensor, t_next_tensor = tensors

        lambda_prev2 = schedule.lambda_(t_prev2_tensor)
        lambda_prev = schedule.lambda_(t_prev_tensor)
        lambda_curr = schedule.lambda_(t_curr_tensor)
        lambda_next = schedule.lambda_(t_next_tensor)

        h = lambda_next - lambda_curr
        h_prev = lambda_curr - lambda_prev
        h_prev2 = lambda_prev - lambda_prev2

        # Guard against numerical instability in timestep ratios
        if abs(h) < EPSILON_CLAMP:
            logger.debug(
                f"DPM-Solver timestep too small (|h|={abs(h):.2e}), "
                f"falling back to second-order update"
            )
            return self._second_order_update(x, model_outputs[-2:], timesteps, idx, schedule)

        r1 = h_prev / h
        r2 = h_prev2 / h

        if not (torch.isfinite(torch.tensor(r1)) and torch.isfinite(torch.tensor(r2))):
            logger.debug(f"DPM-Solver produced non-finite ratios, falling back to second-order")
            return self._second_order_update(x, model_outputs[-2:], timesteps, idx, schedule)

        alpha_curr, sigma_curr = schedule.alpha_sigma(t_curr_tensor)
        alpha_next, _ = schedule.alpha_sigma(t_next_tensor)

        denom1 = r1 * (r1 + r2)
        denom2 = r2 * (r1 + r2)

        if abs(denom1) < EPSILON_CLAMP or abs(denom2) < EPSILON_CLAMP:
            logger.debug(f"DPM-Solver denominators too small, falling back to second-order")
            return self._second_order_update(x, model_outputs[-2:], timesteps, idx, schedule)

        D1 = (1 + 1 / denom1) * model_outputs[-1] - (1 / denom1) * model_outputs[-2]
        D2 = (
            (1 + 1 / denom2) * model_outputs[-1]
            - (1 + 1 / denom2 + 1 / denom1) * model_outputs[-2]
            + (1 / denom1) * model_outputs[-3]
        )

        alpha_ratio = (alpha_next / alpha_curr).to(dtp)
        sigma_term = (sigma_curr * torch.expm1(h)).to(dtp)

        return alpha_ratio * x - sigma_term * (1.5 * D1 - 0.5 * D2)
