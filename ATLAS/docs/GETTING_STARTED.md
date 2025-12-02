# Getting Started with ATLAS - Complete Beginner's Guide

Welcome to ATLAS! This guide assumes you're new to diffusion models and will walk you through everything step-by-step.

## What is ATLAS?

ATLAS is a toolkit for creating AI-generated images using **diffusion models** - the same technology behind tools like Stable Diffusion and DALL-E. With ATLAS, you can:

- **Generate images** from trained models (called "inference")
- **Train your own models** on custom datasets
- **Fine-tune existing models** for specific styles or subjects

## Important: Understanding the Workflow

Before starting, understand these key concepts:

### 1. What is a "Checkpoint"?
A checkpoint is a **saved, trained model** - think of it as the "brain" that knows how to generate images. Without a checkpoint, you cannot generate images.

### 2. Two Main Workflows

#### **Workflow A: Using Existing Checkpoints (Faster)**
- Download a pre-trained checkpoint
- Generate images immediately
- **Time**: 5-10 minutes to set up, seconds to minutes per image

#### **Workflow B: Training Your Own Model (Slower)**
- Prepare a dataset (thousands of images)
- Train a model (takes days/weeks on GPU)
- Generate images with your custom model
- **Time**: Days to weeks for training, then same as Workflow A

**For beginners, we recommend starting with Workflow A.**

## Prerequisites

### What You Need

1. **Python 3.10 or 3.11**
   - Check: `python --version`
   - Install from [python.org](https://www.python.org/downloads/)

2. **16GB+ RAM recommended** (8GB minimum)

3. **GPU (Optional but Recommended)**
   - NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3060 or better)
   - AMD/Apple Silicon GPUs work but are slower
   - CPU-only works but is 5-10x slower

4. **Disk Space**
   - 5GB for installation
   - 10-50GB for checkpoints and generated images
   - 100GB+ if training your own models

### What You'll Learn

- How to install ATLAS
- How to check your hardware
- How to generate your first images
- Where to find checkpoints
- (Optional) How to train your own models

## Step 1: Installation

### 1.1 Create a Virtual Environment

This keeps ATLAS separate from other Python projects:

```bash
# Create environment
python -m venv atlas_env

# Activate it
# On Windows:
atlas_env\Scripts\activate
# On macOS/Linux:
source atlas_env/bin/activate
```

You should see `(atlas_env)` in your terminal prompt.

### 1.2 Install PyTorch

Choose the command for your hardware:

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**NVIDIA GPU (CUDA 11.8 - older systems):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**AMD GPU (Linux only):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
```

**CPU Only or Apple Silicon (Mac):**
```bash
pip install torch torchvision
```

### 1.3 Install ATLAS

```bash
# Clone the repository
git clone https://github.com/tsoreze/OT-diffusion-toolkit.git
cd OT-diffusion-toolkit/ATLAS

# Install ATLAS with all features
pip install -e .[vision,clip]
```

This will take 2-5 minutes.

## Step 2: Verify Installation

Check that everything works:

```bash
python -m atlas.check_hardware
```

You should see output like:
```
=== ATLAS Hardware Detection ===
Device: CUDA (NVIDIA GeForce RTX 4090)
GPU Memory: 24.0 GB
BF16 Supported: Yes
Recommended batch size: 4-8
```

If you see errors, check the [Troubleshooting](#troubleshooting) section.

## Step 3: Get a Checkpoint

You have three options:

### Option A: Download a Pre-trained Checkpoint (Recommended for Beginners)

Unfortunately, ATLAS doesn't come with pre-trained checkpoints. You have these choices:

1. **Train a small test model** (see Step 4)
2. **Use community checkpoints** (if available - check ATLAS discussions/issues)
3. **Adapt checkpoints from compatible projects** (advanced)

**Why no pre-trained models?** Training diffusion models requires significant computational resources (days on high-end GPUs). Most projects don't distribute pre-trained models due to size and licensing concerns.

### Option B: Train a Quick Test Model

For learning purposes, train a tiny model on a small dataset:

```bash
# This will train a small model for testing (takes 2-4 hours on GPU)
python -m atlas.examples.basic_training \
    --dataset mnist \
    --epochs 10 \
    --resolution 64 \
    --batch-size 32 \
    --checkpoint-dir ./my_first_model
```

### Option C: Train a Full Model

See [How to Train & Run Anywhere](HOW_TO_TRAIN_AND_INFER.md) for complete training instructions.

## Step 4: Generate Your First Images

Once you have a checkpoint, generate images:

### Simple Method (Easy API)

```python
from atlas.easy_api import create_sampler

# Create sampler - automatically detects your GPU
sampler = create_sampler(
    checkpoint="path/to/your/checkpoint.pt",
    gpu_memory="auto",  # or "8GB", "16GB", "24GB"
)

# Generate images
images = sampler.generate(
    num_samples=4,      # Generate 4 images
    timesteps=50,       # More steps = better quality (slower)
)

# Save images
from torchvision.utils import save_image
for i, img in enumerate(images):
    save_image(img, f"output_{i}.png")

print("Images saved!")
```

Save this as `generate.py` and run:
```bash
python generate.py
```

### Understanding the Parameters

- **`checkpoint`**: Path to your trained model file (.pt)
- **`gpu_memory`**: "auto" lets ATLAS detect your GPU, or specify like "8GB"
- **`num_samples`**: How many images to generate
- **`timesteps`**: Quality/speed trade-off:
  - 20-30: Fast, lower quality
  - 50: Balanced (recommended)
  - 100+: Slow, highest quality

## Step 5: Experiment and Learn

### Generate More Images

Try different settings:

```python
# Higher quality (slower)
images = sampler.generate(num_samples=1, timesteps=100)

# Batch generation (faster than one-by-one)
images = sampler.generate(num_samples=16, timesteps=50)
```

### Text-to-Image (if your model supports it)

```python
sampler = create_sampler(
    checkpoint="path/to/checkpoint.pt",
    enable_clip=True,  # Enable text conditioning
)

images = sampler.generate(
    prompts=["a sunset over mountains", "a futuristic city"],
    num_samples=2,
    guidance_scale=7.5,  # How strongly to follow the prompt
)
```

## Common Beginner Questions

### Q: Where do I get checkpoints?

**A:** You need to either:
1. Train your own (see [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md))
2. Check ATLAS community forums/discussions for shared checkpoints
3. Adapt models from similar projects (advanced)

### Q: How long does training take?

**A:** Depends on your hardware and dataset:
- **Small model (64x64, simple dataset)**: 2-4 hours on RTX 3090
- **Medium model (256x256)**: 2-4 days on RTX 4090
- **Large model (1024x1024)**: 1-2 weeks on RTX 4090

### Q: Can I use CPU only?

**A:** Yes, but it's 5-10x slower. Recommended for learning/testing only.

### Q: What datasets can I use?

**A:** ATLAS supports:
- **Built-in**: MNIST, CIFAR-10, CelebA, FFHQ, ImageNet, LSUN
- **Custom**: Any folder of images organized by class

See [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md#preparing-datasets) for details.

### Q: My GPU runs out of memory, what do I do?

**A:** Reduce memory usage:
```python
sampler = create_sampler(
    checkpoint="checkpoint.pt",
    gpu_memory="8GB",  # Lower than actual memory
    batch_size=1,      # Reduce batch size
    resolution=512,    # Lower resolution
)
```

### Q: How do I know if my images are good?

**A:** Training takes time. Early images will be blurry/noisy. After thousands of training steps, they'll improve. Check:
- After 1K steps: Basic shapes/colors
- After 10K steps: Recognizable objects
- After 50K+ steps: High quality

### Q: Can I fine-tune an existing model?

**A:** Yes! Load a checkpoint and continue training:
```bash
python -m atlas.examples.lsun256_training \
    --checkpoint ./base_model.pt \
    --data-root ./my_custom_data \
    --epochs 50
```

## Troubleshooting

### "No module named 'atlas'"

**Solution:** Make sure you're in the ATLAS directory and installed with `pip install -e .[vision,clip]`

### "CUDA out of memory"

**Solution:** Reduce batch size, resolution, or use CPU:
```python
sampler = create_sampler(..., batch_size=1, resolution=256)
```

### "checkpoint file not found"

**Solution:** Check the path is correct and file exists:
```bash
ls path/to/checkpoint.pt
```

### "torch.cuda.is_available() returns False"

**Solution:**
1. Check NVIDIA drivers: `nvidia-smi` (should show your GPU)
2. Reinstall PyTorch with CUDA support
3. Verify GPU is detected: `python -m atlas.check_hardware`

### Images are all noise/garbage

**Solution:** This is normal if:
1. Model isn't trained enough (train more epochs)
2. Wrong checkpoint loaded
3. Dataset was too small/varied

## Next Steps

Now that you understand the basics:

1. **Learn training**: [How to Train & Run Anywhere](HOW_TO_TRAIN_AND_INFER.md)
2. **Quick reference**: [Quick Start Guide](QUICKSTART.md)
3. **Optimize performance**: [GPU/CPU Behavior](GPU_CPU_BEHAVIOR.md)
4. **Advanced features**: [CUDA Graphs & Tiling](CUDA_GRAPHS_TILING.md)
5. **Customize**: [Extending ATLAS](EXTENDING.md)

## Getting Help

- **Documentation index**: [docs/README.md](README.md)
- **Hardware issues**: `python -m atlas.check_hardware`
- **Bug reports**: [GitHub Issues](https://github.com/tsoreze/OT-diffusion-toolkit/issues)
- **Questions**: [GitHub Discussions](https://github.com/tsoreze/OT-diffusion-toolkit/discussions)

## Quick Command Reference

```bash
# Check hardware
python -m atlas.check_hardware

# Train a test model
python -m atlas.examples.basic_training --help

# Generate images
python generate.py  # (using code from Step 4)

# Get help
python -m atlas.examples.lsun256_training --help
```

---

**Welcome to ATLAS!** Take your time, experiment, and don't hesitate to ask questions in the community forums.
