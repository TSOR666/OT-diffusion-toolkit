"""High-resolution generation example using hierarchical sampling."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import torch

from atlas.config import ConditioningConfig, HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def validate_resolution(resolution: int, depth: int) -> None:
    """Validate that resolution is compatible with model depth."""
    downsampling_factor = 2 ** depth
    if resolution < downsampling_factor or resolution % downsampling_factor != 0:
        raise ValueError(
            f"Resolution {resolution} must be a multiple of {downsampling_factor} "
            f"(2^{depth}). Valid examples: {downsampling_factor}, {downsampling_factor*2}, "
            f"{downsampling_factor*4}, ..."
        )


def create_model() -> Tuple[HighResLatentScoreModel, HighResModelConfig]:
    """Create a small unconditional model for demonstration."""
    model_config = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=64,
        channel_mult=(1, 2, 4),  # depth=3 → downsampling factor 8
        num_res_blocks=2,
        attention_levels=(1, 2),
        num_heads=4,
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False),
    )
    score_model = HighResLatentScoreModel(model_config)
    score_model.eval()
    return score_model, model_config


def main(resolution: int = 128, steps: int = 16, device: str = "cpu") -> None:
    """Generate a high-resolution sample using hierarchical sampling."""
    print("=" * 70)
    print("Hierarchical High-Resolution Sampling")
    print("=" * 70)

    try:
        score_model, model_config = create_model()
        device_obj = torch.device(device)
        score_model = score_model.to(device_obj)

        depth = len(model_config.channel_mult)
        print(f"Model depth: {depth} levels, max channels: {model_config.max_channels}")

        # Validate resolution compatibility
        validate_resolution(resolution, depth)

        kernel_config = KernelConfig(epsilon=0.05, solver_type="auto")
        sampler_config = SamplerConfig(
            sb_iterations=2,
            hierarchical_sampling=True,
            memory_efficient=True,
        )

        sampler = AdvancedHierarchicalDiffusionSampler(
            score_model=score_model,
            noise_schedule=karras_noise_schedule,
            device=device_obj,
            kernel_config=kernel_config,
            sampler_config=sampler_config,
        )

        timesteps = torch.linspace(1.0, 0.01, steps).tolist()
        shape = (1, model_config.in_channels, resolution, resolution)

        print(f"\nGenerating {resolution}x{resolution} sample ({steps} steps) on {device} ...")
        start = time.time()
        samples = sampler.sample(
            shape,
            timesteps,
            conditioning=None,  # Explicitly unconditional
            verbose=True,
        )
        elapsed = time.time() - start

        if not torch.isfinite(samples).all():
            print("✗ Samples contain NaN or Inf.")
            return

        print(f"\n✓ Generated sample in {elapsed:.2f}s ({elapsed/steps:.3f}s per step)")
        print(f"  Shape: {samples.shape}")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")

        torch.save(samples, f"hierarchical_{resolution}x{resolution}.pt")
        print(f"✓ Saved tensor to hierarchical_{resolution}x{resolution}.pt")

    except ValueError as exc:
        print(f"✗ Configuration error: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"✗ Runtime error: {exc}")
        if "CUDA out of memory" in str(exc):
            print("  Try reducing resolution or steps.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(130)
    except Exception as exc:  # pragma: no cover
        print(f"✗ Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical high-resolution sampling demonstration"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Output resolution (must be multiple of 2^depth, e.g., 128, 256, 512)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Number of diffusion steps (16=fast, 50=quality, 100=best)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on, e.g., 'cpu' or 'cuda'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resolution=args.resolution, steps=args.steps, device=args.device)
