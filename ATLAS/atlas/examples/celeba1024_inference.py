"""Inference script for CelebA-HQ 1024x1024 checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer absolute import for direct execution; fallback to relative when imported as module
try:  # pragma: no cover
    from atlas.examples.training_pipeline import run_inference
except ImportError:  # pragma: no cover
    from .training_pipeline import run_inference


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CelebA-HQ 1024x1024 samples from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Directory where generated samples will be written (default: preset output)",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="Device override, e.g. 'cuda:1' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--preset",
        default="experiment:celeba1024",
        metavar="NAME",
        help="Preset configuration to use (default: experiment:celeba1024)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists() or not checkpoint.is_file():
        print(f"✗ Checkpoint not found or not a file: {checkpoint}")
        sys.exit(1)

    try:
        manifest_path = run_inference(
            args.preset,
            checkpoint_path=str(checkpoint),
            output_dir=args.output,
            device=args.device,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"✗ Inference failed: {exc}")
        raise

    print(f"✓ Saved CelebA1024 samples manifest to {manifest_path}")


if __name__ == "__main__":
    main()
