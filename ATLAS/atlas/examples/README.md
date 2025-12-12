# ATLAS Examples

This directory contains runnable examples and helper scripts to get started with the ATLAS diffusion toolkit.

## Inference
- `cifar10_inference.py` - Generate 32x32 CIFAR-10 samples.
- `celeba1024_inference.py` - Generate 1024x1024 CelebA-HQ samples from a checkpoint.
- `ffhq128_inference.py` - Generate 128x128 FFHQ samples.
- `imagenet64_inference.py` - Generate 64x64 ImageNet samples.
- `lsun256_inference.py` - Generate 256x256 LSUN Bedroom samples.

Common flags:
```
--checkpoint PATH   # Required: model checkpoint (.pt)
--output DIR        # Optional: output directory for PNGs + manifest
--device DEVICE     # Optional: e.g. cuda:0 or cpu (default: auto)
--preset NAME       # Optional: preset name (defaults to dataset-specific preset)
```
Run with `python -m atlas.examples.<script> --checkpoint <path> ...`.

## Training
- `cifar10_training.py` - Train on CIFAR-10 32x32.
- `celeba1024_training.py` - Train on CelebA-HQ 1024x1024.
- `ffhq128_training.py` - Train on FFHQ 128x128.
- `imagenet64_training.py` - Train on ImageNet 64x64.
- `lsun256_training.py` - Train on LSUN Bedroom 256x256.

Common flags:
```
--data-root PATH    # Dataset root directory
--checkpoints DIR   # Checkpoint output directory
--device DEVICE     # Device override (cuda:N or cpu)
--max-steps N       # Optional max training steps
```

## Other Examples
- `basic_sampling.py` - Minimal unconditional sampling demo.
- `custom_kernel.py` - Configure alternate kernel operators.
- `high_res_generation.py` - Hierarchical high-resolution sampling.
- `text_to_image.py` - Text-to-image with CLIP conditioning.
- `easy_start.py` - Beginner-friendly "easy API" walkthrough.
- `training_pipeline.py` - Shared training/inference helpers.

All scripts include argument validation and friendly error handling. Use `--help` on any script for details.
