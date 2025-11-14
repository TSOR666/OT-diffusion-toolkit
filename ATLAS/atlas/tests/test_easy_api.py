import pytest

from atlas.config import ConditioningConfig, HighResModelConfig, KernelConfig, SamplerConfig
from atlas.easy_api import GPUProfile, validate_configs


def test_validate_configs_raises_on_context_mismatch() -> None:
    model_config = HighResModelConfig(context_dim=512)
    kernel_config = KernelConfig()
    sampler_config = SamplerConfig()
    conditioning_config = ConditioningConfig(use_clip=True, clip_model="ViT-L-14")

    with pytest.raises(ValueError):
        validate_configs(
            model_config=model_config,
            kernel_config=kernel_config,
            sampler_config=sampler_config,
            conditioning_config=conditioning_config,
        )


def test_validate_configs_raises_on_memory_overflow() -> None:
    model_config = HighResModelConfig()
    kernel_config = KernelConfig()
    sampler_config = SamplerConfig(use_mixed_precision=False)
    conditioning_config = ConditioningConfig(use_clip=False)
    profile = GPUProfile(
        name="test",
        memory_mb=500,
        batch_size=16,
        resolution=512,
        use_mixed_precision=False,
        kernel_solver="rff",
        kernel_cache_size=4,
        enable_clip=False,
        gradient_checkpointing=False,
        description="test profile",
    )

    with pytest.raises(ValueError):
        validate_configs(
            model_config=model_config,
            kernel_config=kernel_config,
            sampler_config=sampler_config,
            conditioning_config=conditioning_config,
            gpu_profile=profile,
        )
