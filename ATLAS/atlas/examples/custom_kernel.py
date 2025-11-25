"""Example demonstrating custom kernel operator configuration."""

from __future__ import annotations

import time
from typing import Tuple

import torch

from atlas.config import ConditioningConfig, HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler


def create_small_model() -> Tuple[HighResLatentScoreModel, HighResModelConfig]:
    """Create a small unconditional model suitable for CPU demonstration."""
    model_config = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_levels=(1,),
        num_heads=4,
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False),
    )
    score_model = HighResLatentScoreModel(model_config)
    score_model.eval()
    return score_model, model_config


def main() -> None:
    """Demonstrate custom kernel configuration for sampling."""
    print("=" * 60)
    print("Custom Kernel Configuration Example")
    print("=" * 60)

    try:
        device = torch.device("cpu")
        score_model, model_config = create_small_model()
        score_model = score_model.to(device)

        print("\n1. Model configuration:")
        print(f"   Levels: {len(model_config.channel_mult)}")
        print(f"   Max channels: {model_config.max_channels}")
        print(f"   Parameters: {sum(p.numel() for p in score_model.parameters()):,}")

        # Configure a Nyström kernel for speed over accuracy
        kernel_config = KernelConfig(
            kernel_type="gaussian",
            solver_type="nystrom",
            n_landmarks=32,
            epsilon=0.2,
        )

        print("\n2. Kernel configuration:")
        print(f"   Type: {kernel_config.kernel_type}")
        print(f"   Solver: {kernel_config.solver_type}")
        print(f"   Landmarks: {kernel_config.n_landmarks}")
        print(f"   Epsilon: {kernel_config.epsilon}")

        sampler_config = SamplerConfig(
            sb_iterations=2,
            hierarchical_sampling=False,
            memory_efficient=True,
        )

        sampler = AdvancedHierarchicalDiffusionSampler(
            score_model=score_model,
            noise_schedule=karras_noise_schedule,
            device=device,
            kernel_config=kernel_config,
            sampler_config=sampler_config,
        )

        # Timesteps and sampling
        timesteps = torch.linspace(1.0, 0.02, 6).tolist()
        shape = (1, model_config.in_channels, 32, 32)

        print("\n3. Sampling...")
        start = time.time()
        samples = sampler.sample(shape, timesteps, conditioning=None, verbose=False)
        elapsed = time.time() - start

        if not torch.isfinite(samples).all():
            print("✗ Samples contain NaN or Inf.")
            return

        print("\n" + "=" * 60)
        print("✓ Sampling completed successfully!")
        print("=" * 60)
        print(f"  Shape: {samples.shape}")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"  Time:  {elapsed:.2f}s ({elapsed/len(timesteps):.3f}s per step)")
        print("=" * 60)

        torch.save(samples, "custom_kernel_sample.pt")
        print("✓ Saved sample to custom_kernel_sample.pt")

    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        import traceback

        traceback.print_exc()


def demo_kernel_comparison() -> None:
    """Compare a few kernel configurations."""
    device = torch.device("cpu")
    score_model, model_config = create_small_model()
    score_model = score_model.to(device)

    configs = {
        "Fast (Nyström)": KernelConfig(solver_type="nystrom", n_landmarks=32, epsilon=0.2),
        "Balanced (RFF)": KernelConfig(solver_type="rff", rff_features=512, epsilon=0.1),
        "Quality (RFF)": KernelConfig(solver_type="rff", rff_features=2048, epsilon=0.05),
    }

    results = []
    for name, kernel_config in configs.items():
        try:
            sampler = AdvancedHierarchicalDiffusionSampler(
                score_model=score_model,
                noise_schedule=karras_noise_schedule,
                device=device,
                kernel_config=kernel_config,
                sampler_config=SamplerConfig(sb_iterations=1, hierarchical_sampling=False),
            )
            timesteps = torch.linspace(1.0, 0.02, 5).tolist()
            shape = (1, model_config.in_channels, 32, 32)
            start = time.time()
            sampler.sample(shape, timesteps, conditioning=None, verbose=False)
            elapsed = time.time() - start
            results.append((name, elapsed))
        except Exception as exc:  # pragma: no cover
            print(f"{name}: failed with {exc}")

    print("\nComparison results:")
    for name, elapsed in results:
        print(f"  {name:20s}: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
    # Uncomment to run comparison:
    # demo_kernel_comparison()
