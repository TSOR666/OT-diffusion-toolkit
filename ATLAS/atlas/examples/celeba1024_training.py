"""Training script for CelebA-HQ 1024x1024 using the ATLAS preset."""

from __future__ import annotations

import argparse

from .training_pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Path to the CelebA-HQ dataset root")
    parser.add_argument(
        "--checkpoints", default=None, help="Directory where checkpoints will be saved"
    )
    parser.add_argument("--device", default=None, help="Device override such as 'cuda:1' or 'cpu'")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on the number of optimizer steps to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(
        "experiment:celeba1024",
        dataset_root=args.data_root,
        checkpoint_dir=args.checkpoints,
        device=args.device,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
