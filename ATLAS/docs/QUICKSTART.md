# ATLAS Quick Start Guide

Quick reference for users who have basic familiarity with diffusion models and Python.

**New to ATLAS?** See the [Complete Beginner's Guide](GETTING_STARTED.md) first.

---

## Prerequisites

Before starting, ensure you have:
- ✅ Python 3.10 or 3.11 installed
- ✅ Basic command line familiarity
- ✅ 16GB+ RAM (8GB minimum)
- ✅ GPU recommended (8GB+ VRAM) but optional
- ✅ A **trained checkpoint** or plan to train one (see below)

**Important:** You cannot generate images without a trained model checkpoint. See [Understanding Checkpoints](#understanding-checkpoints).

---

## Understanding Checkpoints

A **checkpoint** is a trained model file (`.pt`) that ATLAS uses to generate images. You have three options:

1. **Train your own** (2-4 hours for small models, days for production models)
   - See [Training Section](#5-training-your-own-model) below
   - Most flexible but requires time and GPU resources

2. **Download community checkpoints** (if available)
   - Check ATLAS discussions/forums
   - Verify compatibility with your ATLAS version

3. **Quick test model** (for learning)
   - Train a tiny model on MNIST/CIFAR-10 (1-2 hours)
   - Good for understanding the workflow

**This guide assumes you have or will obtain a checkpoint.**

---

## 1. Installation

```bash
# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install ATLAS with all features
cd OT-diffusion-toolkit/ATLAS
pip install -e .[vision,clip]
```

**Platform-specific installation:** See [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md#3-install-pytorch-for-your-platform) for Windows/macOS/AMD GPU instructions.

## 2. Check Your Hardware

```bash
python -m atlas.check_hardware
```

This will show:
- Your GPU/CPU capabilities
- Recommended batch sizes
- Supported precision modes (FP16/BF16/TF32)
- Expected performance

## 3. Generate Your First Images (Simple API)

**Save this as `generate.py`:**

```python
from atlas.easy_api import create_sampler
from torchvision.utils import save_image

# Create sampler (auto-detects GPU)
sampler = create_sampler(
    checkpoint="path/to/your/checkpoint.pt",  # ⚠️ Replace with your actual checkpoint path
    gpu_memory="auto",  # Auto-detect, or specify "8GB", "16GB", "24GB"
)

# Generate images (unconditional)
images = sampler.generate(
    num_samples=4,        # Generate 4 images
    timesteps=50,         # Quality/speed tradeoff (20-100)
)

# Save results
for i, img in enumerate(images):
    save_image(img, f"output_{i}.png")

print("✅ Images saved as output_0.png to output_3.png")
```

**Run it:**
```bash
python generate.py
```

**Parameters explained:**
- `checkpoint`: Path to your trained model file (required)
- `gpu_memory`: "auto" detects automatically, or specify your VRAM size
- `num_samples`: How many images to generate
- `timesteps`: Higher = better quality but slower (recommended: 50)

## 4. Generate Images with Text Prompts (CLIP)

If your model supports text conditioning:

```python
from atlas.easy_api import create_sampler
from torchvision.utils import save_image

sampler = create_sampler(
    checkpoint="path/to/checkpoint.pt",
    enable_clip=True,  # Enable text conditioning
)

images = sampler.generate(
    prompts=["a serene mountain landscape", "a futuristic city at night"],
    negative_prompts=["blurry, low quality"],  # Optional: what to avoid
    num_samples=2,
    timesteps=50,
    guidance_scale=7.5,  # How strongly to follow the prompt (1.0-20.0)
)

for i, img in enumerate(images):
    save_image(img, f"text_to_image_{i}.png")
```

**Note:** CLIP conditioning requires the `[clip]` extra: `pip install -e .[vision,clip]`

## 5. Advanced API (Full Control)

```python
import torch
from atlas.models import HighResLatentScoreModel
from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.solvers import AdvancedHierarchicalDiffusionSampler
from atlas.schedules import karras_noise_schedule

# Configure model
model_config = HighResModelConfig(
    in_channels=4,
    out_channels=4,
    base_channels=192,
    channel_mult=(1, 2, 4, 4),
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HighResLatentScoreModel(model_config).to(device).eval()

# Load checkpoint
checkpoint = torch.load("model.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])

# Configure sampler
kernel_config = KernelConfig(
    kernel_type="gaussian",
    solver_type="auto",  # Automatically selects best kernel
    epsilon=0.01,
)

sampler_config = SamplerConfig(
    hierarchical_sampling=True,
    sb_iterations=3,
    use_mixed_precision=True,
)

# Create sampler
sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model,
    noise_schedule=karras_noise_schedule,
    device=device,
    kernel_config=kernel_config,
    sampler_config=sampler_config,
)

# Sample
timesteps = 50
samples = sampler.sample(
    shape=(4, 4, 64, 64),  # (batch, channels, height, width)
    timesteps=timesteps,
)
```

## 6. Training Your Own Model

### Quick Test Training (1-2 hours)

Train a small model for testing/learning:

```bash
# CIFAR-10 (fastest, auto-downloads)
python -m atlas.examples.cifar10_training \
    --data-root ./data/cifar10 \
    --checkpoints ./my_first_model \
    --device cuda \
    --max-steps 10000

# ImageNet 64×64 (requires manual download)
python -m atlas.examples.imagenet64_training \
    --data-root /path/to/imagenet64 \
    --checkpoints ./my_first_model \
    --device cuda \
    --max-steps 10000
```

**What this does:**
- CIFAR-10: Auto-downloads 32×32 images, fastest for testing
- ImageNet: Uses 64×64 images, requires [manual download](GETTING_STARTED.md#dataset-downloads)
- Trains for 10K steps (~1-2 hours on RTX 4090)
- Saves checkpoints to `./my_first_model/`
- You can start seeing results after ~5K steps

### Full Production Training

For real production models (days to weeks):

```bash
# CelebA-HQ 1024×1024 (takes ~1 week on RTX 4090)
python -m atlas.examples.celeba1024_training \
    --data-root /datasets/celeba_hq \
    --checkpoints ./checkpoints/celeba

# LSUN Bedroom 256×256 (takes ~3-4 days on RTX 4090)
python -m atlas.examples.lsun256_training \
    --data-root /datasets/lsun/bedroom \
    --checkpoints ./checkpoints/lsun
```

**Training tips:**
- Start with small resolutions (CIFAR-10 32×32 or ImageNet 64×64)
- Monitor training: checkpoints saved every 5K steps
- Training times in [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md#55-estimated-training-times-default-presets)

**Dataset downloads:** See [Getting Started § Dataset Downloads](GETTING_STARTED.md#dataset-downloads) for all dataset links.

**Custom datasets:** See [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md#5-training-atlas-presets) for using your own image folders.

## 8. Troubleshooting

### "CUDA out of memory"

**Solutions:**
```python
# 1. Reduce batch size
sampler = create_sampler(checkpoint="model.pt", batch_size=1)

# 2. Lower resolution (if supported by your checkpoint)
sampler = create_sampler(checkpoint="model.pt", resolution=512)

# 3. Enable tiling for ultra-high-res
sampler = create_sampler(checkpoint="model.pt", tile_size=512, tile_overlap=0.125)

# 4. Clear cache between runs
sampler.clear_cache()
```

### "checkpoint file not found"

**Solutions:**
1. Check the path is correct: `ls path/to/checkpoint.pt`
2. Use absolute path: `/full/path/to/checkpoint.pt`
3. Verify file extension is `.pt`

### Slow generation (taking forever)

**Solutions:**
```python
# 1. Enable CUDA graphs (10-30% speedup for fixed shapes)
sampler = create_sampler(checkpoint="model.pt", enable_cuda_graphs=True)

# 2. Reduce timesteps (faster, slightly lower quality)
images = sampler.generate(num_samples=4, timesteps=30)  # Instead of 50

# 3. Check GPU is being used
python -m atlas.check_hardware  # Should show GPU, not CPU

# 4. Verify mixed precision is enabled (automatic on compatible GPUs)
```

### "No module named 'open_clip'" or CLIP errors

**Solutions:**
```bash
# Install CLIP support
pip install -e .[clip]

# Or disable CLIP if not needed
sampler = create_sampler(checkpoint="model.pt", enable_clip=False)
```

### Images are all noise/garbage

**Common causes:**
1. **Model not trained enough**: Train for more steps (check loss curve)
2. **Wrong checkpoint**: Verify you're loading the correct file
3. **Incompatible checkpoint**: Checkpoint might be from different version/architecture
4. **Dataset too small**: Need 1000+ diverse images for good results

**Try:**
- Check training logs/loss values
- Verify checkpoint size (should be 100MB+)
- Test with a known-good checkpoint

### "torch.cuda.is_available() returns False"

**Solutions:**
1. Check NVIDIA drivers: Run `nvidia-smi` in terminal
2. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. Verify GPU is detected: `python -m atlas.check_hardware`
4. Try CPU mode (slower): `sampler = create_sampler(..., device="cpu")`

### Training crashes or stalls

**Solutions:**
1. Reduce batch size in preset configuration
2. Enable gradient checkpointing (see [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md))
3. Check dataset path is correct
4. Monitor GPU temperature (may be thermal throttling)

## 9. Quick Reference

| Task | Command/Configuration |
|------|----------------------|
| Check hardware | `python -m atlas.check_hardware` |
| Fast generation | `timesteps=30, enable_cuda_graphs=True` |
| High quality | `timesteps=100` |
| Ultra-high-res (2K) | `tile_size=512, tile_overlap=0.125` |
| Batch processing | `batch_size=8, enable_cuda_graphs=True` |
| Memory constrained | `batch_size=1, resolution=512` |
| Text conditioning | `enable_clip=True, guidance_scale=7.5` |

## 10. Next Steps

### Learn More
- **Absolute beginner?** → [Complete Beginner's Guide](GETTING_STARTED.md)
- **Cross-platform setup** → [How to Train & Run Anywhere](HOW_TO_TRAIN_AND_INFER.md)
- **Optimize performance** → [GPU/CPU Behavior Guide](GPU_CPU_BEHAVIOR.md)
- **Advanced features** → [CUDA Graphs & Tiling](CUDA_GRAPHS_TILING.md)
- **Extend ATLAS** → [Extending Guide](EXTENDING.md)
- **All docs** → [Documentation Index](README.md)

### Explore Examples
```bash
# List available examples
ls atlas/examples/*.py

# Run basic sampling demo
python -m atlas.examples.basic_sampling

# Get help on any example
python -m atlas.examples.lsun256_training --help
```

### Get Help
- **Documentation**: [docs/README.md](README.md)
- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions, share results
- **Hardware diagnostics**: `python -m atlas.check_hardware`

---

**Ready to dive deeper?** Check out the [complete documentation index](README.md) for specialized guides on advanced topics.
