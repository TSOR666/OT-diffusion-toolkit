# ATLAS Quick Start Guide

**ATLAS** (Advanced Transport Learning via Adaptive Schrodinger) is a state-of-the-art diffusion model for high-resolution image generation. This guide will help you get started quickly, even if you're not a machine learning expert.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start (3 Lines of Code)](#quick-start-3-lines-of-code)
3. [GPU Memory Requirements](#gpu-memory-requirements)
4. [Basic Usage Examples](#basic-usage-examples)
5. [Text-to-Image Generation](#text-to-image-generation)
6. [Training Your Own Model](#training-your-own-model)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Features](#advanced-features)

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (6GB+ VRAM recommended)
- PyTorch 2.0+

### Install ATLAS

```bash
# Clone the repository
git clone https://github.com/yourusername/atlas.git
cd atlas

# Install dependencies
pip install -r requirements.txt

# Optional: Install CLIP for text-to-image generation
pip install open-clip-torch
```

---

## Quick Start (3 Lines of Code)

The easiest way to use ATLAS:

```python
import atlas.easy_api as atlas

# Automatically detects your GPU and configures everything
sampler = atlas.create_sampler(checkpoint="model.pt")

# Generate images
images = sampler.generate(num_samples=4)
```

That's it! ATLAS will:
- ✅ Detect your GPU memory
- ✅ Configure optimal settings
- ✅ Handle all the complexity for you

---

## GPU Memory Requirements

ATLAS automatically detects your GPU and adjusts settings. Here's what to expect:

| GPU Memory | Example GPUs | Resolution | Batch Size | CLIP Support |
|------------|--------------|------------|------------|--------------|
| 6GB | GTX 1660, RTX 3050 | 512×512 | 1 | ❌ No |
| 8GB | RTX 3060, RTX 4060 | 512×512 | 2 | ✅ Yes |
| 12GB | RTX 3080, RTX 4070 Ti | 1024×1024 | 4 | ✅ Yes |
| 16GB | RTX 4080, RTX 4090 | 1024×1024 | 8 | ✅ Yes |
| 24GB | RTX 4090, A5000 | 1024×1024 | 16 | ✅ Yes |

**Note:** If you get "Out of Memory" errors, see [Troubleshooting](#troubleshooting).

---

## Basic Usage Examples

### Example 1: Generate Unconditional Samples

```python
import atlas.easy_api as atlas
import torch
from PIL import Image

# Create sampler (auto-detects GPU)
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    gpu_memory="8GB",  # or let it auto-detect
)

# Generate samples
samples = sampler.generate(
    num_samples=4,
    timesteps=50,  # Higher = better quality, slower (25-100)
    seed=42,       # For reproducibility
)

# Save images
for i, sample in enumerate(samples):
    # Convert from [-1, 1] to [0, 255]
    img = (sample.clamp(-1, 1) + 1) * 127.5
    img = img.permute(1, 2, 0).cpu().numpy().astype("uint8")
    Image.fromarray(img).save(f"output_{i}.png")
```

### Example 2: Specify Your GPU Manually

```python
import atlas.easy_api as atlas

# For 6GB GPU (memory-optimized)
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    gpu_memory="6GB",
    resolution=512,
    batch_size=1,
)

# For 16GB GPU (high quality)
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    gpu_memory="16GB",
    resolution=1024,
    batch_size=8,
)
```

### Example 3: Check Memory Usage

```python
# Estimate memory before generating
mem_estimate = sampler.estimate_memory_usage(
    batch_size=4,
    resolution=1024
)

print(f"Estimated memory usage:")
print(f"  Model parameters: {mem_estimate['model_params_mb']:.1f} MB")
print(f"  Activations: {mem_estimate['activations_mb']:.1f} MB")
print(f"  Total: {mem_estimate['total_mb']:.1f} MB")

# Clear cache if needed
sampler.clear_cache()
```

---

## Text-to-Image Generation

**Requires:** 8GB+ GPU and CLIP installed (`pip install open-clip-torch`)

```python
import atlas.easy_api as atlas

# Create sampler with CLIP enabled
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    gpu_memory="8GB",
    enable_clip=True,  # Enable text conditioning
)

# Generate from text prompts
images = sampler.generate(
    prompts=[
        "a red sports car on a highway",
        "a blue house in the mountains",
        "a cat sitting on a couch",
        "a futuristic cityscape at sunset"
    ],
    num_samples=2,  # 2 images per prompt = 8 total
    guidance_scale=7.5,  # Higher = more faithful to prompt (1.0-15.0)
    timesteps=50,
    seed=42,
)

print(f"Generated {len(images)} images")
```

### Guidance Scale Tips

- **1.0**: No guidance (unconditional)
- **3.0-5.0**: Subtle guidance
- **7.5**: Recommended default
- **10.0-15.0**: Strong guidance (may over-saturate)

---

## Training Your Own Model

### Quick Training on Custom Dataset

```python
from atlas.config.presets import load_preset
from atlas.examples.training_pipeline import run_training

# Load preset for your GPU
config = load_preset("gpu:8gb")  # or "gpu:12gb", "gpu:16gb", etc.

# Customize for your dataset
config["dataset"].root = "./my_dataset"
config["dataset"].resolution = 512
config["training"].epochs = 100
config["training"].checkpoint_dir = "./checkpoints/my_model"

# Start training
run_training(config)
```

### Training with All Settings Automated

```python
import atlas.easy_api as atlas
from atlas.examples.training_pipeline import run_training

# For 8GB GPU users
config = atlas.load_preset("gpu:8gb")

# Point to your data
config["dataset"].root = "./data/my_images"

# Train!
run_training(config)
```

**Your dataset should be organized as:**
```
my_images/
├── img_001.jpg
├── img_002.jpg
├── img_003.jpg
└── ...
```

---

## Troubleshooting

### Problem: "Out of GPU Memory"

**Solution 1: Use a smaller GPU profile**
```python
# Instead of auto-detect
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    gpu_memory="6GB",  # Force smaller profile
)
```

**Solution 2: Reduce batch size**
```python
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    batch_size=1,  # Minimum
)
```

**Solution 3: Clear GPU cache**
```python
import torch
sampler.clear_cache()
torch.cuda.empty_cache()
```

**Solution 4: Use lower resolution**
```python
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    resolution=512,  # Instead of 1024
)
```

### Problem: "CLIP not found" (Text-to-Image)

**Solution:**
```bash
pip install open-clip-torch
```

### Problem: "Slow generation"

**Tips to speed up:**
1. Reduce timesteps: `timesteps=25` instead of `timesteps=100`
2. Use larger batch size (if you have GPU memory)
3. Enable compilation (first run will be slow):
   ```python
   # In config
   config["training"].compile = True
   ```

### Problem: "Poor image quality"

**Solutions:**
1. Increase timesteps: `timesteps=100` (slower but better)
2. Increase guidance scale for text-to-image: `guidance_scale=10.0`
3. Train for more epochs
4. Use larger model (if GPU allows)

---

## Advanced Features

### Custom Configuration

For advanced users who want full control:

```python
from atlas.config.model_config import HighResModelConfig
from atlas.config.kernel_config import KernelConfig
from atlas.config.sampler_config import SamplerConfig
from atlas.solvers.hierarchical_sampler import AdvancedHierarchicalDiffusionSampler
from atlas.models.score_network import HighResLatentScoreModel
from atlas.schedules.noise import karras_noise_schedule
import torch

# Configure model
model_config = HighResModelConfig(
    in_channels=4,
    out_channels=4,
    base_channels=192,
    channel_mult=(1, 2, 4, 4),
    time_emb_dim=768,
)

# Configure kernel solver
kernel_config = KernelConfig(
    solver_type="auto",  # "direct", "fft", "rff", "nystrom", or "auto"
    epsilon=0.01,
    rff_features=2048,
)

# Configure sampler
sampler_config = SamplerConfig(
    sb_iterations=3,
    use_mixed_precision=True,
    memory_efficient=True,
    memory_threshold_mb=8192,
)

# Create components
device = torch.device("cuda")
model = HighResLatentScoreModel(config=model_config).to(device)

# Create sampler
sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model,
    noise_schedule=karras_noise_schedule,
    device=device,
    kernel_config=kernel_config,
    sampler_config=sampler_config,
)

# Sample
samples = sampler.sample(
    shape=(4, 4, 128, 128),  # (batch, channels, height, width)
    timesteps=50,
)
```

### Using Pre-configured Presets

```python
from atlas.config.presets import load_preset

# Available presets
presets = [
    "gpu:6gb",      # 6GB consumer GPU
    "gpu:8gb",      # 8GB consumer GPU
    "gpu:12gb",     # 12GB high-end GPU
    "gpu:16gb",     # 16GB prosumer GPU
    "gpu:24gb",     # 24GB professional GPU
    "experiment:lsun256",    # LSUN Bedroom 256x256
    "experiment:celeba1024", # CelebA-HQ 1024x1024
]

# Load and customize
config = load_preset("gpu:12gb")
print(config["description"])
print(f"Resolution: {config['resolution']}")
print(f"Max batch size: {config['max_batch_size']}")
```

### List All GPU Profiles

```python
import atlas.easy_api as atlas

# See all available profiles
atlas.list_profiles()
```

---

## Performance Tips

### For Best Quality
- Use `timesteps=100`
- Use `guidance_scale=7.5-10.0` for text-to-image
- Train for more epochs (400+)
- Use larger GPU profile if available

### For Best Speed
- Use `timesteps=25`
- Reduce batch size
- Use FFT kernel: `kernel_config.solver_type="fft"`
- Enable model compilation: `config["training"].compile=True`

### For Minimum Memory
- Use `gpu_memory="6GB"`
- Set `batch_size=1`
- Disable CLIP: `enable_clip=False`
- Use `resolution=512`

---

## Next Steps

1. **Try the examples** in `ATLAS/atlas/examples/`
2. **Read the architecture docs** in `reports/atlas/`
3. **Explore advanced configs** in `ATLAS/atlas/config/`
4. **Train on your data** using the presets
5. **Join the community** (if applicable)

---

## Getting Help

- **Documentation:** See `reports/atlas/ATLAS_MEMORY_ANALYSIS.md`
- **Issues:** Check existing issues on GitHub
- **Examples:** Browse `ATLAS/atlas/examples/`

---

## Common Workflows

### Workflow 1: Quick Experimentation
```python
import atlas.easy_api as atlas

sampler = atlas.create_sampler(checkpoint="model.pt")
images = sampler.generate(num_samples=4, timesteps=25, seed=42)
```

### Workflow 2: High-Quality Generation
```python
import atlas.easy_api as atlas

sampler = atlas.create_sampler(checkpoint="model.pt", gpu_memory="12GB")
images = sampler.generate(
    prompts=["your prompt here"],
    num_samples=4,
    timesteps=100,
    guidance_scale=7.5,
)
```

### Workflow 3: Training from Scratch
```python
from atlas.config.presets import load_preset
from atlas.examples.training_pipeline import run_training

config = load_preset("gpu:8gb")
config["dataset"].root = "./my_data"
config["training"].epochs = 200
run_training(config)
```

---

**Happy generating with ATLAS!** 🚀

For more details, see the full documentation in the `reports/` directory.
