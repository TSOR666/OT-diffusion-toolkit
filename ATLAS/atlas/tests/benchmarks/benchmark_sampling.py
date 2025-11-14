import time

import torch

from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def benchmark_sampling(resolution: int = 32, steps: int = 10) -> None:
    config = HighResModelConfig(
        in_channels=2,
        out_channels=2,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_levels=(1,),
        cross_attention_levels=(1,),
    )
    score_model = HighResLatentScoreModel(config)

    kernel_config = KernelConfig(kernel_type="gaussian", epsilon=0.1, solver_type="direct")
    sampler_config = SamplerConfig(
        sb_iterations=1,
        hierarchical_sampling=False,
        use_linear_solver=False,
    )

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=score_model,
        noise_schedule=karras_noise_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    timesteps = torch.linspace(1.0, 0.01, steps).tolist()

    start = time.time()
    _ = sampler.sample((1, config.in_channels, resolution, resolution), timesteps, verbose=False)
    duration = time.time() - start

    print(f"Sampled {resolution}x{resolution} image in {duration:.2f} s over {steps} steps")


if __name__ == "__main__":
    benchmark_sampling()
