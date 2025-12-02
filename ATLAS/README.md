# ATLAS Diffusion Toolkit

[![CI](https://github.com/tsoreze/OT-diffusion-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/tsoreze/OT-diffusion-toolkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tsoreze/OT-diffusion-toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/tsoreze/OT-diffusion-toolkit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## What is ATLAS?

ATLAS is a **complete toolkit for creating AI-generated images** using diffusion models (the same technology behind Stable Diffusion). With ATLAS, you can:

- ✨ **Generate high-resolution images** (up to 2048×2048) on consumer GPUs
- 🎨 **Train custom models** on your own datasets
- ⚡ **Optimize for your hardware** (automatic GPU/CPU detection, mixed precision)
- 🔧 **Extend and customize** every component (models, kernels, samplers, schedules)

ATLAS is a **full-stack solution** - it includes everything you need: model architectures, training loops, sampling algorithms, and examples. It's built on mathematically rigorous foundations combining score-based models, Schrödinger-bridge transport, and flexible kernel operators.

**✨ Latest Features:** Auto hardware detection, BF16/TF32 optimization, native 2K generation on RTX 4090/5090, CUDA graphs for 10-30% speedup.

---

## For Beginners

**New to ATLAS or diffusion models?**

👉 **Start here:** [Complete Beginner's Guide](docs/GETTING_STARTED.md)

This guide explains:
- What diffusion models are
- How to install ATLAS on any platform
- Where to get or train model checkpoints
- How to generate your first images
- Common beginner questions and troubleshooting

---

## Quick Start

**Prerequisites:**
- Python 3.10+, PyTorch 2.0+
- A **trained checkpoint** (model file) - see [Getting Started Guide](docs/GETTING_STARTED.md) for how to obtain one
- GPU recommended (8GB+ VRAM) but CPU works (slower)

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install ATLAS with all features
pip install -e .[vision,clip]

# 3. Check your hardware
python -m atlas.check_hardware

# 4. Generate images using a trained checkpoint
python -c "
from atlas.easy_api import create_sampler
# Replace 'model.pt' with a checkpoint you trained/downloaded
sampler = create_sampler(checkpoint='model.pt', gpu_memory='auto')
images = sampler.generate(prompts=['a mountain landscape'], timesteps=50)
"
```

**📚 Guides for different experience levels:**
- **Absolute beginner?** → [Complete Beginner's Guide](docs/GETTING_STARTED.md)
- **Quick reference?** → [Quick Start Guide](docs/QUICKSTART.md)
- **Detailed walkthrough?** → [How to Train & Run Anywhere](docs/HOW_TO_TRAIN_AND_INFER.md)
- **All documentation** → [docs/README.md](docs/README.md)

---

## Installation

### Standard Installation
1. Create a Python 3.10+ environment.
2. Install a CUDA-, ROCm-, or CPU-enabled PyTorch build.
3. Install ATLAS (optionally with extras):
   ```bash
   pip install .             # core (PyTorch, NumPy, tqdm)
   pip install .[vision]     # dataset and image utilities (torchvision, Pillow)
   pip install .[clip]       # CLIP conditioning (open-clip-torch)
   pip install .[dev]        # development tools (pytest, ruff)
   ```

### Check Hardware Capabilities
After installation, verify your setup:
```bash
python -m atlas.check_hardware
```

This will show your GPU capabilities and recommend optimal settings.

See [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) for detailed requirements and compatibility.

---

## Key Features

### For All Users
- **Easy API**: Simple `create_sampler()` interface for quick results
- **Hardware Auto-Detection**: Automatically optimizes for your GPU/CPU
- **Consumer GPU Support**: Runs on 6GB+ GPUs (RTX 3060 and up)
- **Cross-Platform**: Works on Windows, Linux, macOS (including Apple Silicon)
- **Production Ready**: Full test coverage, CI/CD, type hints

### For Advanced Users
- **Multiple Kernel Operators**: Direct, FFT, RFF, Nyström solvers
- **CUDA Graph Acceleration**: 10-30% speedup for repeated sampling
- **Ultra-High Resolution**: Native 2K generation with automatic tiling
- **Mixed Precision**: BF16/FP16/TF32 support with automatic selection
- **CLIP Conditioning**: Text-to-image generation
- **Flexible Architecture**: Customize models, kernels, schedules, samplers

### Technical Strengths
- **Hierarchical Samplers**: Multi-scale generation for high resolution
- **Schrödinger Bridge Transport**: Stable, mathematically grounded sampling
- **Kernel Registry**: Pluggable kernel operators with automatic tier selection
- **LoRA Support**: Memory-efficient fine-tuning

## Training Guide

1. **Configure the score model**
   ```python
   from atlas.config import HighResModelConfig

   model_cfg = HighResModelConfig(
       in_channels=4,
       out_channels=4,
       base_channels=192,
       channel_mult=(1, 2, 4, 4),
       num_res_blocks=2,
       attention_levels=(1, 2),
       conditional=True,
       context_dim=768,
   )
   ```

2. **Build the dataset pipeline**
   ```python
   from atlas.config import DatasetConfig

   dataset_cfg = DatasetConfig(
       name="celeba",
       root="./data/celeba",
       resolution=256,
       channels=3,
       center_crop=True,
       batch_size=16,
       download=True,
   )
   ```

3. **Set training hyperparameters**
   ```python
   from atlas.config import TrainingConfig

   train_cfg = TrainingConfig(
       batch_size=32,
       micro_batch_size=8,
       learning_rate=2e-4,
       gradient_clip_norm=1.0,
       epochs=300,
       mixed_precision=True,
       log_interval=100,
       checkpoint_interval=5000,
   )
   ```

4. **Launch a preset pipeline**
   ```python
   from atlas.examples.training_pipeline import run_training

   run_training(
       preset_name="experiment:celeba1024",
       dataset_root="./data/celeba",
       checkpoint_dir="./checkpoints/celeba",
       device="cuda",
   )
   ```

   Or use the CLI shims that mirror the preset names:

   ```bash
   python -m atlas.examples.cifar10_training --data-root ./data/cifar10  # Auto-downloads
   python -m atlas.examples.imagenet64_training --data-root /datasets/imagenet64
   python -m atlas.examples.ffhq128_training --data-root /datasets/ffhq
   python -m atlas.examples.lsun256_training --data-root /datasets/lsun/bedroom
   python -m atlas.examples.celeba1024_training --data-root /datasets/celeba_hq
   ```

   **Dataset downloads:** CIFAR-10 auto-downloads. Other datasets require manual download - see [Dataset Downloads](docs/GETTING_STARTED.md#dataset-downloads).

**Tips**
- Enable gradient checkpointing for deep UNets (`model_cfg.use_checkpointing=True`).
- Use `micro_batch_size` to trade compute for memory on 12 GB GPUs.
- For text-to-image setups construct a `CLIPConditioningInterface` and supply
  conditioning payloads to the hierarchical sampler.

---

## Inference Guide

### High-Level API
```python
from atlas.easy_api import create_sampler

sampler = create_sampler(
    checkpoint="checkpoints/atlas_latest.pt",
    gpu_memory="16GB",
)

images = sampler.generate(
    prompts=["a marble statue", "a city at dawn"],
    num_samples=4,
    timesteps=60,
    guidance_scale=7.5,
)

sampler.clear_cache()
```

### Advanced Sampler
```python
import torch
from atlas.models import HighResLatentScoreModel
from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.solvers import AdvancedHierarchicalDiffusionSampler
from atlas.schedules import karras_noise_schedule

model = HighResLatentScoreModel(HighResModelConfig()).to("cuda").eval()
state = torch.load("checkpoints/atlas_latest.pt", map_location="cuda")
model.load_state_dict(state["model"])

kernel_cfg = KernelConfig(kernel_type="gaussian", epsilon=0.1, solver_type="rff")
sampler_cfg = SamplerConfig(hierarchical_sampling=True, sb_iterations=2)

sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model,
    noise_schedule=karras_noise_schedule,
    device=torch.device("cuda"),
    kernel_config=kernel_cfg,
    sampler_config=sampler_cfg,
)

timesteps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
images = sampler.sample((4, 4, 64, 64), timesteps, verbose=False)
```

---

## Kernel and Sampler Registry
- `atlas.kernels`: direct, FFT, RFF, and Nystrom operator implementations.
- `atlas.solvers`: Schrodinger bridge solver with Richardson extrapolation and selective updates.
- `atlas.examples`: runnable scripts for training, inference, conditioning, and custom kernels.

---

## Consumer GPU Profiles
- **6-12 GB** (RTX 3050/3060/4070): mixed precision, gradient checkpointing, RFF kernels by default.
- **16 GB** (RTX 4080/4090): BF16 (when supported), auto kernel tier, CLIP conditioning enabled.
- **24 GB** (4090 workstation): larger kernel caches, higher RFF feature counts.
- **32 GB** (next-gen flagship): presets for 1536x1536 generation with adaptive kernels.

`EasySampler.generate` automatically halves batch size if a CUDA OOM is detected.

---

## Documentation

For detailed guides and advanced topics, see the [docs/](docs/) directory:

- **[Quick Start Guide](docs/QUICKSTART.md)** ⚡ - Get started in 5 minutes with examples and common configurations
- **[Dependencies Matrix](docs/DEPENDENCIES.md)** - Detailed dependency requirements, compatibility matrix, installation guides
- **[GPU/CPU Behavior](docs/GPU_CPU_BEHAVIOR.md)** - Hardware detection, precision modes (FP16/BF16/TF32), performance expectations, troubleshooting
- **[CUDA Graphs & Tiling](docs/CUDA_GRAPHS_TILING.md)** - Complete guide to CUDA graph acceleration and ultra-high-resolution tiling
- **[Extending ATLAS](docs/EXTENDING.md)** - How to implement custom score networks, kernels, schedules, conditioning, and samplers

**All documentation**: See [docs/README.md](docs/README.md) for the complete index.

### Quick Reference: Hardware Capabilities

ATLAS automatically detects hardware capabilities and adjusts settings:

```python
from atlas.utils.hardware import print_hardware_info

# Print detailed hardware information
print_hardware_info()

# Get capabilities programmatically
from atlas.utils.hardware import get_hardware_info
info = get_hardware_info()
print(f"Device: {info['device_name']}")
print(f"BF16 Supported: {info['bf16_supported']}")
print(f"Recommended Batch: {info['max_batch_size']}")
```

The toolkit automatically:
- ✅ Enables optimal precision (BF16/TF32 on Ampere+, FP16 on older GPUs)
- ✅ Adjusts RFF features based on available memory
- ✅ Selects best kernel solver for your hardware
- ✅ Disables incompatible features (e.g., CUDA graphs on CPU)
- ✅ Warns about expensive operations and suggests alternatives

---

## Hardware & Resolution Guidance

ATLAS includes presets for common VRAM sizes; below are indicative maxima for typical UNet variants. Actual limits depend on model width, LoRA use, steps, and kernel tier.

- 6–8 GB (RTX 2060/3060/4060): 512x512, batch 1–2; RFF/FFT kernels, mixed precision
- 10–12 GB (RTX 3080/4070): 1024x1024, batch 1–2; auto kernel tier, CLIP optional
- 16 GB (RTX 4080/4090/A4000): 1024x1024, batch 4–8; BF16/TF32 where supported
- 24 GB (RTX 3090/4090/A5000): 1024x1024, batch 8–16; larger kernel caches
- 32 GB+ (A6000/flagship): up to 1536x1536; adaptive kernels and higher RFF counts

Tips:
- Reduce `batch_size` or enable `gradient_checkpointing` for deep UNets.
- Prefer RFF/FFT operators for large images; use Nystrom at low dimensions.
- Use `micro_batch_size` for memory trade‑offs on consumer GPUs.

---

## Project Layout

## High-Resolution Optimisations (4090/5090)

ATLAS now includes native CUDA graph capture and spatial tiling to sustain 1K–2K generation on high-end GPUs without exhausting memory:

- **CUDA graphs** reduce launch overhead. Enable via `SamplerConfig(enable_cuda_graphs=True, cuda_graph_warmup_iters=2)` when running on CUDA.
- **Tiling** evaluates the score model on overlapping windows. Set `tile_size` (e.g., 960 for 1K, 1152–1280 for 2K), adjust `tile_overlap` (default 12.5%) and optional `tile_stride`.
- Both features compose: the score network is wrapped with CUDA graphs first, then tile evaluation is applied, enabling native 1024–2048 sampling on 4090/5090 without shrinking batch size.

Example:
```python
from atlas.config import SamplerConfig

sampler_cfg = SamplerConfig(
    enable_cuda_graphs=True,
    cuda_graph_warmup_iters=3,
        tile_size=960,
    tile_overlap=0.125,
)
```

Windowing uses Hann weights by default to eliminate seams. The sampler automatically adjusts edge tiles so the entire image is covered.

---

```
atlas/
  conditioning/    # CLIP interfaces and guidance helpers
  config/          # Dataclasses (models, kernels, samplers, presets)
  examples/        # Training/inference scripts
  kernels/         # Kernel operator implementations
  models/          # Score networks, attention blocks, LoRA helpers
  schedules/       # Noise schedules and timestep utilities
  solvers/         # Schrodinger bridge solver and hierarchical sampler
  tests/           # Pytest regression suite
  utils/           # Data handling, randomness, memory helpers
```

---

## Testing and Diagnostics
- `python -m compileall atlas`
- `pytest`
- `python -m atlas.examples.basic_sampling`
- Set `ATLAS_DISABLE_MEMORY_WARNINGS=1` to silence optional CUDA warnings.

---

## Mathematical Foundations

**For researchers and advanced users:** ATLAS implements mathematically rigorous algorithms based on recent advances in diffusion models and optimal transport.

### Score-Based Diffusion
ATLAS adopts the variance-preserving SDE:
```
dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dW_t
```
where the score network learns `s_theta(x, t) ≈ grad_x log p_t(x)` via denoising score matching. The probability-flow ODE:
```
dx/dt = -0.5 * beta(t) * x - beta(t) * s_theta(x, t)
```
is integrated with predictor-corrector schemes or higher-order samplers.

### Schrödinger Bridge Updates
To stabilize long trajectories, the sampler inserts an entropic optimal transport solve at each step. Given model samples `x_t` and reference samples `y_t`, the bridge computes dual potentials `(u, v)` satisfying:
```
pi(x, y) ∝ exp(u(x)) * k(x, y) * exp(v(y))
```
where `k` is a kernel operator (Gaussian by default). The barycentric projection of `pi` supplies a transport map that complements the probability-flow update.

### Kernel Operators
ATLAS implements several kernel backends optimized for different scenarios:
- **Direct**: Exact O(n²) Gram matrices for small batches
- **FFT**: Convolutional kernels for grid-structured data (images/volumes)
- **Random Fourier Features (RFF)**: Sub-quadratic approximations for point clouds
- **Nyström**: Low-rank sketches when memory budgets are tight

Bandwidths, feature counts, and approximations are configured via `KernelConfig`.

**Further Reading:**
- Score-based models: Song et al. (2021)
- Schrödinger bridges: De Bortoli et al. (2021)
- Optimal transport: Peyré & Cuturi (2019)

---

## License & Citation
ATLAS is distributed under the Apache License 2.0. Retain attribution to Thierry Silvio Claude Soreze. If ATLAS supports your research, cite the repository and acknowledge the author.
