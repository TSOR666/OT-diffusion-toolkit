"""High level sampling interface built on top of the SB solver."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from ..conditioning import CLIPConditioningInterface, expand_condition_dict
from ..config.kernel_config import KernelConfig
from ..config.sampler_config import SamplerConfig
from ..utils.memory import get_peak_memory_mb, reset_peak_memory, warn_on_high_memory
from ..utils.random import set_seed
from .schrodinger_bridge import SchroedingerBridgeSolver


class AdvancedHierarchicalDiffusionSampler:
    """User facing sampler that wraps :class:`SchroedingerBridgeSolver`."""

    def __init__(
        self,
        score_model: nn.Module,
        noise_schedule: Callable[[float], float],
        device: Optional[torch.device] = None,
        kernel_config: Optional[KernelConfig] = None,
        sampler_config: Optional[SamplerConfig] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.score_model = score_model.to(device)
        self.noise_schedule = noise_schedule
        self.kernel_config = kernel_config or KernelConfig()
        self.sampler_config = sampler_config or SamplerConfig()

        if self.sampler_config.seed is not None:
            set_seed(self.sampler_config.seed)

        self.logger = logging.getLogger(self.__class__.__name__)
        level = logging.INFO if self.sampler_config.verbose_logging else logging.WARNING
        self.logger.setLevel(level)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

        self.sb_solver = SchroedingerBridgeSolver(
            score_model=self.score_model,
            noise_schedule=self.noise_schedule,
            device=self.device,
            kernel_config=self.kernel_config,
            sampler_config=self.sampler_config,
        )

        self.conditioner: Optional[CLIPConditioningInterface] = None
        self.current_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_base_batch: Optional[int] = None

    # ------------------------------------------------------------------
    def set_conditioner(self, conditioner: CLIPConditioningInterface) -> None:
        self.conditioner = conditioner

    def prepare_conditioning_from_prompts(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.conditioner is None:
            raise ValueError("No CLIP conditioner attached to sampler.")
        payload = self.conditioner.build_conditioning_payload(
            prompts,
            negative_prompts=negative_prompts,
            guidance_scale=guidance_scale,
        )
        self.current_conditioning = payload
        self._conditioning_base_batch = payload.get("base_batch", len(prompts))
        return payload

    # ------------------------------------------------------------------
    def _resolve_conditioning(
        self,
        conditioning: Optional[Dict[str, Any]],
        batch_size: int,
    ) -> Optional[Dict[str, Any]]:
        payload = conditioning if conditioning is not None else self.current_conditioning
        if payload is None:
            return None
        base_batch = payload.get("base_batch", self._conditioning_base_batch or batch_size)
        cond = payload.get("cond", payload)
        return expand_condition_dict(cond, batch_size, base_batch, self.device)

    def _prepare_conditioning(
        self,
        batch_size: int,
        conditioning: Optional[Dict[str, Any]],
        prompts: Optional[List[str]],
        negative_prompts: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        if prompts is not None:
            payload = self.prepare_conditioning_from_prompts(
                prompts,
                negative_prompts=negative_prompts,
            )
            return self._resolve_conditioning(payload, batch_size)
        if conditioning is not None:
            self.current_conditioning = conditioning
            self._conditioning_base_batch = conditioning.get("base_batch", batch_size)
        return self._resolve_conditioning(conditioning, batch_size)

    # ------------------------------------------------------------------
    def sample(
        self,
        shape: Sequence[int],
        timesteps: Sequence[float],
        verbose: bool = True,
        callback: Optional[Callable[[torch.Tensor, float, float], None]] = None,
        conditioning: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
        negative_prompts: Optional[List[str]] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(shape) < 2:
            raise ValueError("Shape must include batch and channel dimensions.")
        if len(timesteps) < 2:
            raise ValueError("Timesteps must contain at least two values.")

        schedule = sorted(list(timesteps), reverse=True)
        batch_size = int(shape[0])

        if initial_state is None:
            x_t = torch.randn(tuple(shape), device=self.device)
        else:
            x_t = initial_state.to(self.device)

        active_conditioning = self._prepare_conditioning(
            batch_size, conditioning, prompts, negative_prompts
        )

        iterator: Iterable[int]
        iterator = range(len(schedule) - 1)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling", leave=False)

        if self.sampler_config.memory_efficient:
            reset_peak_memory()

        for idx in iterator:
            t_curr = float(schedule[idx])
            t_next = float(schedule[idx + 1])
            x_t = self.sb_solver.solve_once(
                x_t,
                t_curr,
                t_next,
                conditioning=active_conditioning,
            )
            if callback is not None:
                callback(x_t, t_curr, t_next)

        if self.sampler_config.memory_efficient:
            peak = get_peak_memory_mb()
            warn_on_high_memory(peak, threshold_mb=self.sampler_config.memory_threshold_mb)

        return x_t

    # ------------------------------------------------------------------
    def get_performance_stats(self) -> Dict[str, Any]:
        return self.sb_solver.get_performance_stats()

    def auto_tune_parameters(
        self,
        x: torch.Tensor,
        error_tolerance: float = 1e-3,
    ) -> Dict[str, Any]:
        return self.sb_solver.auto_tune_parameters(x, error_tolerance)
