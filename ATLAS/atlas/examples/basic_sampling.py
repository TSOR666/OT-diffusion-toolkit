"""Minimal example of sampling with the ATLAS sampler."""

import torch

from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def main() -> None:
    model_config = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_levels=(1,),
    )
    score_model = HighResLatentScoreModel(model_config)

    kernel_config = KernelConfig(epsilon=0.1)
    sampler_config = SamplerConfig(sb_iterations=1, hierarchical_sampling=False)

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=score_model,
        noise_schedule=karras_noise_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    timesteps = torch.linspace(1.0, 0.05, 8).tolist()
    samples = sampler.sample((1, model_config.in_channels, 32, 32), timesteps, verbose=False)

    print(f"Generated samples with shape: {samples.shape}")


if __name__ == "__main__":
    main()
