# ATLAS Quick Start Guide

Get started with ATLAS in 5 minutes.

## 1. Installation

```bash
# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install ATLAS with all features
pip install -e .[vision,clip]
```

## 2. Check Your Hardware

```bash
python -m atlas.check_hardware
```

This will show your GPU capabilities and recommended settings.

## 3. Generate Your First Images

### Option A: Simple API (Recommended for Beginners)

```python
from atlas.easy_api import create_sampler

# Create sampler (auto-detects GPU)
sampler = create_sampler(
    checkpoint="path/to/model.pt",
    gpu_memory="auto",  # Or "8GB", "16GB", "24GB"
)

# Generate images
images = sampler.generate(
    prompts=["a serene mountain landscape", "a futuristic city"],
    num_samples=4,
    timesteps=50,
    guidance_scale=7.5,
)

# Save results
from torchvision.utils import save_image
for i, img in enumerate(images):
    save_image(img, f"output_{i}.png")
```

### Option B: Advanced API (Full Control)

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
checkpoint = torch.load("model.pt", map_location=device)
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

## 4. Common Configurations

### Text-to-Image with CLIP

```python
sampler = create_sampler(
    checkpoint="model.pt",
    gpu_memory="16GB",
    enable_clip=True,  # Enable CLIP conditioning
)

images = sampler.generate(
    prompts=["a sunset over the ocean"],
    negative_prompts=["blurry, low quality"],
    guidance_scale=7.5,  # Classifier-free guidance strength
    timesteps=60,
)
```

### High-Resolution Generation (2K)

```python
sampler = create_sampler(
    checkpoint="model.pt",
    gpu_memory="24GB",
    resolution=2048,
    tile_size=512,          # Enable tiling
    tile_overlap=0.125,
    enable_cuda_graphs=True,  # Speed optimization
)

images = sampler.generate(
    prompts=["an intricate mandala pattern"],
    num_samples=1,
    timesteps=80,
)
```

### Batch Generation (Production)

```python
sampler = create_sampler(
    checkpoint="model.pt",
    gpu_memory="24GB",
    batch_size=8,
    enable_cuda_graphs=True,  # Reuses graphs for same shape
)

# Generate many variations
prompts = ["a cat"] * 100  # Generate 100 cat images
all_images = []

for i in range(0, len(prompts), 8):  # Process in batches of 8
    batch_prompts = prompts[i:i+8]
    images = sampler.generate(
        prompts=batch_prompts,
        num_samples=1,
        timesteps=50,
    )
    all_images.append(images)
```

## 5. Training Your Own Model

```python
from atlas.examples.training_pipeline import run_training
from atlas.config import TrainingConfig, DatasetConfig

# Configure dataset
dataset_config = DatasetConfig(
    name="celeba",
    root="./data/celeba",
    resolution=256,
    batch_size=32,
)

# Configure training
train_config = TrainingConfig(
    learning_rate=2e-4,
    epochs=100,
    mixed_precision=True,
    checkpoint_interval=5000,
)

# Launch training
run_training(
    preset_name="base",
    dataset_root="./data/celeba",
    checkpoint_dir="./checkpoints",
    device="cuda",
)
```

## 6. Troubleshooting

### Out of Memory
```python
# Reduce batch size
sampler = create_sampler(..., batch_size=1)

# Lower resolution
sampler = create_sampler(..., resolution=512)

# Enable tiling for high-res
sampler = create_sampler(..., tile_size=512)

# Clear cache between runs
sampler.clear_cache()
```

### Slow Performance
```python
# Enable CUDA graphs (fixed shapes)
sampler = create_sampler(..., enable_cuda_graphs=True)

# Use mixed precision
# (automatically enabled on compatible GPUs)

# Reduce timesteps
images = sampler.generate(..., timesteps=30)  # Instead of 50
```

### CLIP Not Working
```bash
# Install CLIP support
pip install open-clip-torch

# Or disable in config
sampler = create_sampler(..., enable_clip=False)
```

## 7. Next Steps

- **Read the full docs**: [docs/](../docs/)
- **Check hardware capabilities**: `python -m atlas.check_hardware`
- **Explore examples**: [examples/](../examples/)
- **Customize components**: [docs/EXTENDING.md](EXTENDING.md)

## 8. Example Gallery

### Unconditional Generation
```python
sampler = create_sampler(checkpoint="unconditional_model.pt")
images = sampler.generate(num_samples=16, timesteps=50)
```

### Class-Conditional
```python
sampler = create_sampler(checkpoint="class_conditional.pt")
images = sampler.generate(
    condition={"class": 5},  # Generate class 5
    num_samples=8,
)
```

### Image-to-Image
```python
import torch
from PIL import Image
from torchvision import transforms

# Load starting image
init_image = Image.open("input.png")
transform = transforms.ToTensor()
x_init = transform(init_image).unsqueeze(0)

# Add noise and denoise
images = sampler.generate(
    num_samples=1,
    timesteps=50,
    x_init=x_init,
    start_timestep=0.5,  # Start from 50% noise
)
```

## Quick Reference

| Task | Configuration |
|------|---------------|
| Fast generation | `timesteps=30, enable_cuda_graphs=True` |
| High quality | `timesteps=100, guidance_scale=7.5` |
| Large images | `resolution=2048, tile_size=512` |
| Batch production | `batch_size=8, enable_cuda_graphs=True` |
| Memory constrained | `batch_size=1, resolution=512, rff_features=1024` |

## Getting Help

- **Documentation**: [docs/](../docs/)
- **Hardware Check**: `python -m atlas.check_hardware`
- **Issues**: Report at GitHub Issues
- **Examples**: See [examples/](../examples/) directory

---

**Need more details?** Check the [full documentation](README.md) or run:
```bash
python -m atlas.check_hardware --help
```
