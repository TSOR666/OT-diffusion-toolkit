"""
Common configuration presets used across the ATLAS package.
"""

from __future__ import annotations

from .conditioning_config import ConditioningConfig, LoRAConfig
from .kernel_config import KernelConfig
from .model_config import HighResModelConfig
from .sampler_config import SamplerConfig


def highres_latent_score_default() -> HighResModelConfig:
    """Baseline configuration targeting 1024x1024 latent diffusion."""

    conditioning_cfg = ConditioningConfig()
    lora_cfg = LoRAConfig()
    return HighResModelConfig(conditioning=conditioning_cfg, lora=lora_cfg)


def gaussian_multiscale_kernel() -> KernelConfig:
    """Balanced kernel configuration that works well for large images."""

    return KernelConfig()


def hierarchical_sampler_default() -> SamplerConfig:
    """Preset tuned for hierarchical high-resolution sampling."""

    return SamplerConfig()


PRESETS = {
    "model:highres_default": highres_latent_score_default,
    "kernel:gaussian_multiscale": gaussian_multiscale_kernel,
    "sampler:hierarchical_default": hierarchical_sampler_default,
}
