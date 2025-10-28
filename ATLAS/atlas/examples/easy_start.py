"""
ATLAS Easy Start Example
========================

This example demonstrates the simplified ATLAS API for non-experts.
Perfect for getting started quickly with minimal configuration.

Features demonstrated:
- Automatic GPU detection
- Simple sampler creation
- Basic image generation
- Memory management
- Error handling

Usage:
    python easy_start.py --checkpoint model.pt --num_samples 4
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import easy_api as atlas


def save_images(samples: torch.Tensor, output_dir: Path, prefix: str = "sample"):
    """
    Save generated samples as PNG images.

    Args:
        samples: Tensor of shape (N, C, H, W) in range [-1, 1]
        output_dir: Directory to save images
        prefix: Filename prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        # Convert from [-1, 1] to [0, 255]
        img = (sample.clamp(-1, 1) + 1) * 127.5
        img = img.permute(1, 2, 0).cpu().numpy().astype("uint8")

        # Handle different channel counts
        if img.shape[2] == 1:
            img = img.squeeze(2)  # Grayscale
        elif img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel

        # Save
        save_path = output_dir / f"{prefix}_{i:04d}.png"
        Image.fromarray(img).save(save_path)
        print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Easy Start - Simple image generation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional for testing)",
    )
    parser.add_argument(
        "--gpu_memory",
        type=str,
        default=None,
        choices=["6GB", "8GB", "12GB", "16GB", "24GB"],
        help="GPU memory profile (auto-detects if not specified)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50,
        help="Number of diffusion steps (25=fast, 50=balanced, 100=quality)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Text prompts for conditional generation (requires CLIP)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (only for text-to-image)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--list_profiles",
        action="store_true",
        help="List all available GPU profiles and exit",
    )
    args = parser.parse_args()

    # List profiles if requested
    if args.list_profiles:
        atlas.list_profiles()
        return

    print("=" * 80)
    print("ATLAS Easy Start")
    print("=" * 80)

    # Step 1: Create sampler
    print("\n[1/3] Creating sampler...")
    try:
        sampler = atlas.create_sampler(
            checkpoint=args.checkpoint,
            gpu_memory=args.gpu_memory,
        )
    except RuntimeError as e:
        print(f"\nError creating sampler: {e}")
        print("\nTips:")
        print("  - Make sure you have a CUDA-capable GPU")
        print("  - Check NVIDIA drivers: nvidia-smi")
        print("  - Try smaller GPU profile: --gpu_memory 6GB")
        sys.exit(1)

    # Step 2: Estimate memory
    print("\n[2/3] Estimating memory usage...")
    try:
        mem_estimate = sampler.sampler.estimate_memory_usage(
            batch_size=min(args.num_samples, sampler.profile.batch_size),
            resolution=sampler.profile.resolution,
        )
        print(f"  Model parameters: {mem_estimate['model_params_mb']:.1f} MB")
        print(f"  Activations: {mem_estimate['activations_mb']:.1f} MB")
        print(f"  Kernel cache: {mem_estimate['kernel_cache_mb']:.1f} MB")
        if mem_estimate['clip_mb'] > 0:
            print(f"  CLIP: {mem_estimate['clip_mb']:.1f} MB")
        print(f"  Total estimated: {mem_estimate['total_mb']:.1f} MB")
    except Exception as e:
        print(f"  Warning: Could not estimate memory: {e}")

    # Step 3: Generate samples
    print(f"\n[3/3] Generating {args.num_samples} samples...")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Seed: {args.seed}")
    if args.prompts:
        print(f"  Prompts: {args.prompts}")
        print(f"  Guidance scale: {args.guidance_scale}")

    try:
        samples = sampler.generate(
            prompts=args.prompts,
            num_samples=args.num_samples,
            timesteps=args.timesteps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTips:")
        print("  - For text prompts, use --gpu_memory 8GB or higher")
        print("  - Install CLIP: pip install open-clip-torch")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\nError during generation: {e}")
        print("\nTips:")
        print("  - Reduce --num_samples")
        print("  - Reduce --timesteps")
        print("  - Use smaller --gpu_memory profile")
        print("  - Clear GPU cache: sampler.clear_cache()")
        sys.exit(1)

    # Step 4: Save results
    print(f"\nSaving images to {args.output_dir}...")
    output_dir = Path(args.output_dir)
    save_images(samples, output_dir, prefix="atlas_sample")

    # Clear cache
    sampler.clear_cache()

    print("\n" + "=" * 80)
    print(f"✓ Successfully generated {len(samples)} images!")
    print(f"✓ Saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
