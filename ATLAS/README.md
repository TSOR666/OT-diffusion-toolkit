# ATLAS Diffusion Toolkit

ATLAS is a modular high-resolution diffusion stack that unifies score-based models,
Schrodinger bridge transport, and a flexible kernel registry. This document covers
the mathematics, training and inference workflows, and the major modules.

---

## Installation
1. Create a Python 3.10+ environment.
2. Install a CUDA-, ROCm-, or CPU-enabled PyTorch build.
3. Install ATLAS (optionally with extras):
   ```bash
   pip install .             # core (PyTorch, NumPy, tqdm)
   pip install .[vision]     # dataset and image utilities (torchvision, Pillow)
   pip install .[clip]       # CLIP conditioning (open-clip-torch)
   pip install .[dev]        # development tools (pytest, ruff)
   ```

---

## Mathematical Foundations

### Score-Based Diffusion
ATLAS adopts the variance-preserving SDE
```
dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dW_t
```
where the score network learns `s_theta(x, t) ~ grad_x log p_t(x)` via denoising score
matching. The probability-flow ODE is
```
dx/dt = -0.5 * beta(t) * x - beta(t) * s_theta(x, t)
```
and is integrated with predictor-corrector schemes or higher-order samplers.

### Schrodinger Bridge Updates
To stabilise long trajectories the sampler inserts an entropic optimal transport solve
at each step. Given model samples `x_t` and reference samples `y_t`, the bridge computes
dual potentials `(u, v)` satisfying
```
pi(x, y) ~ exp(u(x)) * k(x, y) * exp(v(y))
```
where `k` is a kernel operator (Gaussian by default). The barycentric projection of
`pi` supplies a transport map that complements the probability-flow update.

### Kernel Operators
ATLAS implements several kernel backends:
- **Direct**: exact `O(n^2)` Gram matrices for small batches.
- **FFT**: convolutional kernels for grid-structured data (images/volumes).
- **Random Fourier Features (RFF)**: sub-quadratic approximations for point clouds.
- **Nystrom**: low-rank sketches when memory budgets are tight.

Bandwidths, feature counts, and approximations are configured via `KernelConfig`.

---

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

## Project Layout
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

## License & Citation
ATLAS is distributed under the Apache License 2.0. Retain attribution to Thierry Silvio Claude Soreze. If ATLAS supports your research, cite the repository and acknowledge the author.
