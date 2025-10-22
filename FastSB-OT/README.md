# FastSB-OT Solver Toolkit

## Overview
FastSB-OT delivers a production-grade implementation of a Fast Schrodinger
Bridge sampler with optimal transport refinements. The solver couples
score-based generative models with regularised optimal transport updates to
stabilise long denoising trajectories and accelerate convergence on modern
GPU hardware. The package now ships as a modular Python subpackage
(`fastsb_ot`) with clearly separated configuration, kernels, transport and
solver layers.

## Mathematical Foundations
- **Schrodinger bridge formulation**: The sampler follows the forward
  SDE/ODE pair
  ```
  dx = f(x, t) dt + g(t) dW,        dy = [f(x, t) - g(t)**2 * grad_x log p_t(x)] dt
  ```
  where the score network approximates `grad_x log p_t`. The bridge solves for
  the most likely path between the prior (pure noise) and data marginals.
- **Entropy-regularised optimal transport**: Each iteration solves a
  Sinkhorn-regularised OT problem with adaptive `epsilon`, optionally switching
  to sliced projections when full kernels exceed memory budgets.
- **Momentum and hierarchical bridges**: The transport module adds a
  discrete-time momentum update and multi-scale residual correction driven by
  Fisher information estimates to maintain stability at high resolutions.
- **Fisher-aware kernels**: Triton-accelerated kernels compute diagonal
  Fisher approximations and spectrum-weighted Gaussian convolutions to curb
  score bias at low signal-to-noise ratios.

## Package Layout
- `fastsb_ot/common.py` - shared utilities, Triton kernels, compile cache.
- `fastsb_ot/config.py` - `FastSBOTConfig` dataclass with quality presets and
  hardware-aware defaults.
- `fastsb_ot/cache.py` - GPU-aware tensor cache for score and kernel reuse.
- `fastsb_ot/kernels.py` - frequency-domain Gaussian kernels and Fisher
  estimators with automatic Triton fall-backs.
- `fastsb_ot/transport.py` - sliced OT, momentum transport and hierarchical
  bridge helpers.
- `fastsb_ot/solver.py` - `FastSBOTSolver` glue logic plus sampling schedules.

## Training Strategy
1. **Score network** - train a score model with denoising score matching or
   diffusion losses on the target data distribution.
2. **Noise schedule** - fit or select a monotone `alpha_bar(t)` schedule
   (cosine, Karras, linear). Use the provided `make_schedule` helper for
   common schedules.
3. **Guidance** - the solver expects raw score outputs. Classifier-free or
   conditional guidance can be injected outside the solver by adjusting the
   score model wrapper.
4. **Optimisation tips**
   - Warm-start with the `balanced` preset; switch to `ultra` only after the
     score network stabilises.
  - Enable `use_fp32_fisher` when training with mixed precision to avoid
     bias in Fisher updates.
   - Cap `sliced_ot_projections` to 64 on consumer GPUs to control memory
     pressure during early experimentation.

## Sampling Workflow
```python
import torch
from fastsb_ot import FastSBOTConfig, FastSBOTSolver, make_schedule

# 1. Load a trained score model (must implement .to(device) and __call__)
score_model = MyScoreNetwork().eval()

# 2. Configure solver and schedule
config = FastSBOTConfig(quality="balanced", use_mixed_precision=True)
schedule = make_schedule("cosine", num_timesteps=config.num_timesteps)

# 3. Construct solver and sample
solver = FastSBOTSolver(score_model, schedule, config=config)
samples = solver.sample((8, 4, 256, 256), verbose=True)
```
The solver automatically handles CUDA initialisation, score caching,
compilation and fallback strategies. Use the `persistent_cache` argument to
reuse kernel/Fisher caches across solver instances in long-running services.

## Hardware and Performance Notes
- Triton kernels provide highest throughput on Ampere+ GPUs; set
  `use_triton_kernels=False` when targeting older architectures.
- The memory cache aggressively adapts to available VRAM; monitor
  `cache_size_mb` and `memory_limit_ot_mb` for large-batch inference.
- Mixed precision sampling is enabled by default with automatic fall-back to
  FP32 for numerically sensitive paths.

## License
FastSB-OT is distributed under the Apache License 2.0 (see `LICENSE`). Please
retain attribution to **Thierry Silvio Claude Soreze** in derivative works.

## Citation
If you build upon FastSB-OT in academic work, please cite the repository and
acknowledge Thierry Silvio Claude Soreze as the original author.
