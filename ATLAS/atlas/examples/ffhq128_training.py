"""Training script for FFHQ 128x128 using the ATLAS preset."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Prefer absolute import for direct execution; fallback to relative when packaged
try:  # pragma: no cover
    from atlas.examples.training_pipeline import run_training
except ImportError:  # pragma: no cover
    from .training_pipeline import run_training


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an FFHQ 128x128 diffusion model using preset defaults."
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="PATH",
        help="Path to the FFHQ dataset root (default: preset value)",
    )
    parser.add_argument(
        "--checkpoints",
        default=None,
        metavar="DIR",
        help="Directory where checkpoints will be saved (default: preset value)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Device override such as 'cuda:1' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Optional cap on the number of optimizer steps to run",
    )
    parser.add_argument(
        "--preset",
        default="experiment:ffhq128",
        metavar="NAME",
        help="Preset configuration to use (default: experiment:ffhq128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting training.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments and exit with a helpful message on failure."""
    errors = []
    if args.data_root:
        data_path = Path(args.data_root)
        if not data_path.exists():
            errors.append(f"Dataset path does not exist: {args.data_root}")
        elif not data_path.is_dir():
            errors.append(f"Dataset path is not a directory: {args.data_root}")

    if args.checkpoints:
        ckpt_path = Path(args.checkpoints)
        try:
            ckpt_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"Cannot create checkpoint directory: {exc}")

    if args.device and not re.match(r"^(cpu|cuda(:\d+)?)$", args.device):
        errors.append(f"Invalid device format: {args.device}. Expected 'cpu', 'cuda', or 'cuda:N'.")

    if args.max_steps is not None and args.max_steps <= 0:
        errors.append(f"max_steps must be positive, got {args.max_steps}")

    if errors:
        print("✗ Validation errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


def _print_header(args: argparse.Namespace) -> None:
    """Print basic configuration and expectations."""
    print("=" * 70)
    print("FFHQ 128x128 Training")
    print("=" * 70)
    print("ℹ This model is small and fast compared to 1024x1024 variants.")
    print("   Great for prototyping, quick experiments, and learning.")
    print("=" * 70)
    print(f"Preset:      {args.preset}")
    print(f"Dataset dir: {args.data_root or 'preset default'}")
    print(f"Checkpoints: {args.checkpoints or 'preset default'}")
    print(f"Device:      {args.device or 'auto-detect'}")
    if args.max_steps:
        print(f"Max steps:   {args.max_steps}")
    print("-" * 70)
    print("Estimated requirements:")
    print("  VRAM: ~2-3 GB (mixed precision)")
    print("  Disk: ~20 GB for checkpoints")
    print("  Time: ~12-24 hours on RTX 3090 (faster than 1024px models)")
    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _print_header(args)

    if args.dry_run:
        print("✓ Configuration validated (dry run). Exiting.")
        return

    training_kwargs = {
        "dataset_root": args.data_root,
        "checkpoint_dir": args.checkpoints,
        "device": args.device,
        "max_steps": args.max_steps,
    }

    try:
        run_training(args.preset, **training_kwargs)
    except KeyError as exc:
        print(f"✗ Preset not found: {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"✗ File not found: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"✗ Runtime error: {exc}")
        if "CUDA out of memory" in str(exc):
            print("  Try reducing batch size or using a smaller profile.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user.")
        sys.exit(130)
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"✗ Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n✓ Training completed successfully!")
    print("Next steps: run inference with atlas.examples.ffhq128_inference.py")


if __name__ == "__main__":
    main()
