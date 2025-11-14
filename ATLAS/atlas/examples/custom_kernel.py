"""Example demonstrating custom kernel configuration."""

import torch

from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def main() -> None:
    model_config = HighResModelConfig()
    score_model = HighResLatentScoreModel(model_config)

    kernel_config = KernelConfig(
        kernel_type="gaussian",  # Supported kernel family
        solver_type="nystrom",
        n_landmarks=32,
        epsilon=0.2,
    )
    sampler_config = SamplerConfig(sb_iterations=2, hierarchical_sampling=False)

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=score_model,
        noise_schedule=karras_noise_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    timesteps = torch.linspace(1.0, 0.02, 6).tolist()
    samples = sampler.sample((1, model_config.in_channels, 32, 32), timesteps, verbose=False)

    print(f"Generated sample with custom kernel, shape: {samples.shape}")


if __name__ == "__main__":
    main()
