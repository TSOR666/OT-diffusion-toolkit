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
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Sample from the diffusion model.

        Args:
            shape: Output shape (batch, channels, height, width)
            timesteps: Diffusion timesteps or number of steps
            verbose: Show progress bar
            callback: Optional callback(x_t, t_curr, t_next) called each step
            conditioning: Pre-computed conditioning dict
            prompts: Text prompts for CLIP conditioning
            negative_prompts: Negative prompts for CFG
            initial_state: Optional initial noise (if None, uses random)
            return_intermediates: Return all intermediate states

        Returns:
            Final samples, or (samples, intermediates) if return_intermediates=True

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If sampling fails (e.g., OOM, numerical issues)
        """
        # Validate inputs
        if len(shape) < 2:
            raise ValueError(
                f"Shape must include batch and channel dimensions, got shape={shape}.\n"
                f"Expected format: (batch_size, channels, height, width)"
            )

        if shape[0] <= 0:
            raise ValueError(
                f"Batch size must be positive, got batch_size={shape[0]}.\n"
                f"Use shape=(1, ...) for single sample."
            )

        # Validate timesteps
        try:
            schedule = self.sb_solver.validate_timesteps(timesteps)
        except Exception as e:
            raise ValueError(
                f"Invalid timesteps: {e}\n"
                f"Provide either:\n"
                f"  - Integer number of steps (e.g., 50)\n"
                f"  - List of timestep values in [0, 1]"
            )

        batch_size = int(shape[0])

        # Check for potential OOM before starting
        if self.device.type == "cuda":
            available_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            # Rough estimate: ~500 MB per sample for 1024x1024
            estimated_memory = batch_size * 500
            if estimated_memory > available_memory * 0.9:
                self.logger.warning(
                    f"Potential OOM risk: batch_size={batch_size} may need ~{estimated_memory:.0f} MB, "
                    f"but only {available_memory:.0f} MB available.\n"
                    f"Consider reducing batch_size to {int(batch_size * 0.5)} or lower."
                )

        # Initialize state
        if initial_state is None:
            x_t = torch.randn(tuple(shape), device=self.device)
        else:
            if initial_state.shape != tuple(shape):
                raise ValueError(
                    f"initial_state.shape={initial_state.shape} doesn't match "
                    f"target shape={shape}"
                )
            x_t = initial_state.to(self.device)

        # Prepare conditioning
        try:
            active_conditioning = self._prepare_conditioning(
                batch_size, conditioning, prompts, negative_prompts
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to prepare conditioning: {e}\n"
                f"If using prompts, ensure CLIP is installed: pip install open-clip-torch"
            )

        # Setup iteration
        iterator: Iterable[int]
        iterator = range(len(schedule) - 1)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling", leave=False)

        if self.sampler_config.memory_efficient:
            reset_peak_memory()

        # Store intermediates if requested
        intermediates = [] if return_intermediates else None

        # Main sampling loop with error handling
        try:
            for idx in iterator:
                t_curr = float(schedule[idx])
                t_next = float(schedule[idx + 1])

                x_t = self.sb_solver.solve_once(
                    x_t,
                    t_curr,
                    t_next,
                    conditioning=active_conditioning,
                )

                if return_intermediates:
                    intermediates.append(x_t.clone().cpu())

                if callback is not None:
                    callback(x_t, t_curr, t_next)

        except RuntimeError as e:
            if "out of memory" in str(e):
                # Provide helpful OOM suggestions
                current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

                raise RuntimeError(
                    f"Out of GPU memory during sampling.\n"
                    f"Memory usage: {current_memory:.1f} MB allocated, {max_memory:.1f} MB peak\n"
                    f"Current settings:\n"
                    f"  - batch_size: {batch_size}\n"
                    f"  - shape: {shape}\n"
                    f"  - mixed_precision: {self.sampler_config.use_mixed_precision}\n"
                    f"\nSuggestions:\n"
                    f"  1. Reduce batch_size (currently {batch_size})\n"
                    f"  2. Enable mixed_precision if not already enabled\n"
                    f"  3. Use smaller resolution\n"
                    f"  4. Clear GPU cache: torch.cuda.empty_cache()\n"
                    f"  5. Reduce kernel cache size in KernelConfig"
                ) from e
            else:
                # Re-raise other runtime errors with context
                raise RuntimeError(
                    f"Sampling failed at timestep {idx}/{len(schedule)-1}: {e}"
                ) from e

        if self.sampler_config.memory_efficient:
            peak = get_peak_memory_mb()
            warn_on_high_memory(peak, threshold_mb=self.sampler_config.memory_threshold_mb)

        if return_intermediates:
            return x_t, intermediates
        return x_t

    # ------------------------------------------------------------------
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from the SB solver."""
        return self.sb_solver.get_performance_stats()

    def auto_tune_parameters(
        self,
        x: torch.Tensor,
        error_tolerance: float = 1e-3,
    ) -> Dict[str, Any]:
        """Auto-tune solver parameters for given data."""
        return self.sb_solver.auto_tune_parameters(x, error_tolerance)

    def clear_kernel_cache(self) -> None:
        """
        Clear the kernel operator cache to free GPU memory.

        This is useful when running multiple sampling tasks or when memory is tight.
        The cache will be rebuilt automatically on next use.
        """
        if hasattr(self.sb_solver, 'clear_kernel_cache'):
            self.sb_solver.clear_kernel_cache()
        torch.cuda.empty_cache()
        self.logger.info("Cleared kernel cache and GPU memory")

    def estimate_memory_usage(
        self,
        batch_size: int,
        resolution: int = 1024,
    ) -> Dict[str, float]:
        """
        Estimate GPU memory usage for given batch size and resolution.

        Args:
            batch_size: Number of samples per batch
            resolution: Image resolution (assumes square images)

        Returns:
            Dictionary with memory estimates in MB:
                - model_params: Model parameter memory
                - activations: Activation memory
                - kernel_cache: Kernel operator cache
                - total: Total estimated memory

        Example:
            >>> sampler = create_sampler()
            >>> mem = sampler.estimate_memory_usage(batch_size=4, resolution=1024)
            >>> print(f"Estimated memory: {mem['total']:.1f} MB")
        """
        # Model parameters (constant)
        model_params = 150 if self.sampler_config.use_mixed_precision else 300

        # Activations scale with batch size and resolution
        # Rough estimate: 250 MB per sample at 1024x1024 with FP16
        base_activation = 250 if self.sampler_config.use_mixed_precision else 500
        resolution_scale = (resolution / 1024) ** 2
        activations = batch_size * base_activation * resolution_scale

        # Kernel cache (relatively constant)
        kernel_cache = 200

        # CLIP conditioning if enabled
        clip_memory = 0
        if self.conditioner is not None:
            clip_memory = 150 if self.sampler_config.use_mixed_precision else 300

        total = model_params + activations + kernel_cache + clip_memory

        return {
            "model_params_mb": model_params,
            "activations_mb": activations,
            "kernel_cache_mb": kernel_cache,
            "clip_mb": clip_memory,
            "total_mb": total,
        }
