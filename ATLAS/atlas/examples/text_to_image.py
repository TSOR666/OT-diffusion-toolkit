"""Text-to-image example leveraging CLIP conditioning."""

from __future__ import annotations

import sys
import torch

from typing import cast

from atlas.config import ConditioningConfig, HighResModelConfig, KernelConfig, SamplerConfig
from atlas.conditioning import CLIPConditioningInterface
from atlas.types import ConditioningDict, ConditioningPayload
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def main(prompt: str = "a serene mountain landscape") -> None:
    """Generate an image from a text prompt using CLIP conditioning."""
    if not prompt:
        raise ValueError("Prompt must be a non-empty string.")

    print("=" * 60)
    print("Text-to-Image Generation Example")
    print("=" * 60)
    print(f"Prompt: '{prompt}'")

    # Use a smaller CLIP for demonstration
    conditioning_cfg = ConditioningConfig(
        use_clip=True,
        clip_model="ViT-B-32",
        context_dim=512,
    )
    clip_interface = CLIPConditioningInterface(conditioning_cfg)

    # Small conditional model for CPU friendliness
    model_config = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=64,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_levels=(1,),
        conditional=True,
        cross_attention_levels=(),
        conditioning=conditioning_cfg,
    )
    score_model = HighResLatentScoreModel(model_config).to(torch.device("cpu"))
    score_model.eval()

    kernel_config = KernelConfig()
    sampler_config = SamplerConfig(hierarchical_sampling=True, guidance_scale=7.5, sb_iterations=2)

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

    # Encode prompt explicitly if available
    conditioning: ConditioningPayload | None = None
    try:
        encoded = clip_interface.encode_text([prompt])
        if isinstance(encoded, torch.Tensor):
            conditioning = {"context": encoded}
        elif isinstance(encoded, dict):
            conditioning = cast(ConditioningDict, encoded)
    except Exception:
        conditioning = None  # Fall back to sampler handling prompts directly

    timesteps = torch.linspace(1.0, 0.01, 12).tolist()

    if conditioning is not None:
        samples = sampler.sample(
            (1, model_config.in_channels, 64, 64),
            timesteps,
            conditioning=conditioning,
            show_progress=False,
        )
    else:
        samples = sampler.sample(
            (1, model_config.in_channels, 64, 64),
            timesteps,
            show_progress=False,
            prompts=[prompt],
        )

    if not torch.isfinite(samples).all():
        print("✗ Samples contain NaN or Inf.")
        return

    print(f"✓ Generated conditioned sample with shape {samples.shape}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Skipping text-to-image example: {exc}")
    except Exception as exc:
        print(f"✗ Error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
