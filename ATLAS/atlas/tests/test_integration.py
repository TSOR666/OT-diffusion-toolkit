import torch

from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def test_sampler_generates_tensor() -> None:
    model_config = HighResModelConfig(
        in_channels=2,
        out_channels=2,
        base_channels=32,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(),
        cross_attention_levels=(),
        conditioning_dim=0,
        conditional=False,
    )
    score_model = HighResLatentScoreModel(model_config)

    kernel_config = KernelConfig(
        kernel_type="gaussian",
        epsilon=0.1,
        solver_type="direct",
        orthogonal=False,
    )
    sampler_config = SamplerConfig(
        sb_iterations=1,
        hierarchical_sampling=False,
        use_linear_solver=False,
        memory_efficient=False,
        verbose_logging=False,
    )

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=score_model,
        noise_schedule=karras_noise_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    timesteps = [1.0, 0.0]
    output = sampler.sample((1, model_config.in_channels, 16, 16), timesteps, show_progress=False)

    assert output.shape == (1, model_config.out_channels, 16, 16)
