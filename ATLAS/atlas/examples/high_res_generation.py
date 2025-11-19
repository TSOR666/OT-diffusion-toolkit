"""High-resolution generation example using hierarchical sampling."""

import torch

from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def main(resolution: int = 256, steps: int = 16) -> None:
    model_config = HighResModelConfig(
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
    )
    score_model = HighResLatentScoreModel(model_config)

    kernel_config = KernelConfig(epsilon=0.05, solver_type="auto")
    sampler_config = SamplerConfig(
        sb_iterations=2,
        hierarchical_sampling=True,
        memory_efficient=True,
    )

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=score_model,
        noise_schedule=karras_noise_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    min_resolution = 2 ** len(model_config.channel_mult)
    if resolution < min_resolution or resolution % min_resolution != 0:
        raise ValueError(
            f"Resolution must be divisible by {min_resolution} (got {resolution})."
        )

    timesteps = torch.linspace(1.0, 0.01, steps).tolist()
    samples = sampler.sample((1, model_config.in_channels, resolution, resolution), timesteps, verbose=True)

    print(f"Generated {resolution}x{resolution} sample with shape {samples.shape}")


if __name__ == "__main__":
    main()
