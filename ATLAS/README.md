# ATLAS Diffusion Toolkit

## Overview
ATLAS is a modular high-resolution diffusion research toolkit that combines
score-based models with Schrodinger bridge solvers and a flexible kernel
operator stack. The library targets rapid experimentation on CPU or GPU
hardware and exposes the building blocks required to train, evaluate, and
sample from latent or pixel-space diffusion models.


## Installation
ATLAS ships as a standard Python package. Install it inside a virtual environment (Python 3.10+) and add extras as needed:
```bash
pip install .             # Core functionality (PyTorch, NumPy, tqdm)
pip install .[vision]     # + torchvision & Pillow for dataset/image utilities
pip install .[clip]       # + open-clip-torch for text conditioning
pip install .[dev]        # + pytest and ruff for development
```

For GPU acceleration, install a CUDA-enabled PyTorch build before running the commands above (see https://pytorch.org/get-started/locally/).
## Mathematical Foundations
- **Score-based diffusion**: Models learn `grad_x log p_t(x)` under a noise
  schedule `beta(t)` using denoising score matching. The sampler supports
  probability-flow ODE and predictor-corrector sampling variants.
- **Schrodinger bridges**: Bridge solvers reconcile forward diffusion and
  reverse denoising trajectories, yielding improved stability for long
  horizons and high resolutions.
- **Kernel operators**: Direct, FFT, random Fourier feature (RFF), and
  Nystr�m approximations provide efficient transport maps and covariance
  operators with controllable numerical error.
- **Hierarchical samplers**: The hierarchical bridge decomposes the
  trajectory into coarse-to-fine stages, enabling aggressive down-sampling
  and memory-aware scheduling.

## Training Workflow
1. **Configure the model** using the dataclasses in `atlas.config` to set
   architecture, attention, LoRA, and conditioning options.
2. **Select a schedule** from `atlas.schedules` (linear, cosine, Karras) or
   provide a custom callable returning `alpha_bar(t)`.
3. **Prepare conditioning** with `atlas.conditioning` modules for CLIP or
   classifier-free guidance pipelines.
4. **Train** using your preferred framework (PyTorch Lightning, custom loop).
   The toolkit provides utilities for gradient checkpointing, mixed
   precision, and distributed sampling.
5. **Validate transports** with kernel diagnostics in `atlas.tests` to ensure
   numerical stability before scaling to larger resolutions.

## Sampling
```python
import torch
from atlas import HighResModelConfig, HighResLatentScoreModel
from atlas.config import KernelConfig, SamplerConfig
from atlas.solvers import AdvancedHierarchicalDiffusionSampler
from atlas.schedules import karras_noise_schedule

model = HighResLatentScoreModel(HighResModelConfig())
kernel_cfg = KernelConfig()
sampler_cfg = SamplerConfig(hierarchical_sampling=False)

sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model,
    noise_schedule=karras_noise_schedule,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    kernel_config=kernel_cfg,
    sampler_config=sampler_cfg,
)

samples = sampler.sample((1, 4, 32, 32), timesteps=[1.0, 0.5, 0.01], verbose=False)
```

## Project Layout
```
atlas/
  conditioning/           # CLIP interface and guidance helpers
  config/                 # Dataclasses describing model, kernel, sampler config
  examples/               # Ready-to-run sampling scripts
  kernels/                # Direct, FFT, RFF, Nystrom kernel operators
  models/                 # Score networks, attention blocks, LoRA utilities
  schedules/              # Noise schedules and timestep utilities
  solvers/                # Schrodinger bridge solver and hierarchical sampler
  tests/                  # Pytest regression and kernel diagnostics
  utils/                  # Image, randomness, and memory utilities
```

## Consumer GPU Profiles
- **6–12 GB** cards (e.g. RTX 3060, RTX 4070) run 512–1024² sampling with mixed precision and gradient checkpointing.
- **16 GB** cards (e.g. RTX 4080, 4090) enable auto-tuned kernels and CLIP guidance while maintaining FP16/BF16 inference.
- **24 GB** cards (RTX 4090, professional Ada GPUs) default to larger kernel caches and mixed precision for faster sampling.
- **32 GB** flagship cards (anticipated RTX 5090 class) unlock the new `gpu:32gb` preset and 1536² generation via `atlas.easy_api`.
- Dynamic OOM handling in `EasySampler.generate` automatically reduces batch size if the requested workload exceeds available memory.

## Testing and Debugging
- `python -m compileall atlas` ensures the package parses without syntax
  errors.
- `pytest` runs the regression suite (requires PyTorch and pytest).
- `python -m atlas.examples.basic_sampling` executes a minimal sampling demo.
- Set `ATLAS_DISABLE_MEMORY_WARNINGS=1` to silence optional CUDA memory
  warnings during development.

## License
ATLAS is available under the Apache License 2.0 (see `LICENSE`). Please retain
attribution to **Thierry Silvio Claude Soreze** in derivative works.

## Citation
If ATLAS supports your published work, cite the repository and credit
Thierry Silvio Claude Soreze as the original author.
