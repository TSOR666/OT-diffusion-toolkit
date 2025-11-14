"""Inference script for CelebA-HQ 1024x1024 checkpoints."""

from __future__ import annotations

import argparse

from .training_pipeline import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint file")
    parser.add_argument(
        "--output",
        default=None,
        help="Directory where generated samples will be written",
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. 'cuda:1' or 'cpu'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = run_inference(
        "experiment:celeba1024",
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device=args.device,
    )
    print(f"Saved CelebA1024 samples manifest to {manifest_path}")


if __name__ == "__main__":
    main()
