"""Text-to-image example leveraging CLIP conditioning."""

import torch

from atlas.config import ConditioningConfig, HighResModelConfig, KernelConfig, SamplerConfig
from atlas.conditioning import CLIPConditioningInterface
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def main(prompt: str = "a serene mountain landscape") -> None:
    if not prompt:
        raise ValueError("Prompt must be a non-empty string.")
    conditioning_cfg = ConditioningConfig(use_clip=True)
    clip_interface = CLIPConditioningInterface(conditioning_cfg)

    model_config = HighResModelConfig()
    score_model = HighResLatentScoreModel(model_config)

    kernel_config = KernelConfig()
    sampler_config = SamplerConfig(hierarchical_sampling=True)

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=score_model,
        noise_schedule=karras_noise_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )
    sampler.set_conditioner(clip_interface)
    if sampler.conditioner is None:
        raise RuntimeError("Failed to attach CLIP conditioner to sampler.")

    timesteps = torch.linspace(1.0, 0.01, 12).tolist()
    samples = sampler.sample((1, model_config.in_channels, 64, 64), timesteps, verbose=False, prompts=[prompt])

    print(f"Generated conditioned sample for prompt '{prompt}' with shape {samples.shape}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Skipping text-to-image example: {exc}")
