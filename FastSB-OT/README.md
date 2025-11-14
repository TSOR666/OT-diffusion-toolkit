# FastSB-OT Solver Toolkit

FastSB-OT is a SOLVER that couples your trained score network with fast, entropyâ€‘
### Schrödinger Bridge Formulation
accelerated kernels, and deploymentâ€‘friendly presets.

---

## Installation
```bash
pip install .             # core (PyTorch, NumPy, tqdm)
pip install .[dev]        # development tools (pytest, ruff)
```
Optional extras:
- `pip install triton` for accelerated Gaussian kernels and Fisher estimators.
- `pip install .[env]` if you maintain separate requirements for deployment.

---

## What It Is

- Category: SOLVER â€” requires a trained score network (`s_Î¸(x,t)`) or an
  epsilonâ€‘predicting model wrapped as a score model (see below).
- Strengths: fast OT updates, Fisher momentum, Triton acceleration.
- Use cases: highâ€‘throughput transportâ€‘aware sampling with your existing models.

## Mathematical Foundations

### Schrodinger Bridge Formulation
FastSB-OT solves the dynamic SchrÃ¶dinger bridge problem: given forward SDE
```
dx = f(x, t) dt + g(t) dW_t,
```
find the most likely path measure connecting the prior and data marginals subject to
an entropic constraint. The bridge dynamics are expressed via a pair of dual potentials
`(phi_t, psi_t)` that correct the score-based drift.

### Entropy-Regularised Optimal Transport
Each iteration solves the regularised OT problem
```
Î³* = argmin_Î³ âŸ¨C, Î³âŸ© + Îµ KL(Î³ || a âŠ— b),
```
where `C` is the cost matrix (squared Euclidean distance by default) and `a`, `b`
are marginal weights. Sinkhorn iterations with adaptive `Îµ` provide the transport plan.

### Fisher-Aware Momentum
FastSB-OT augments the bridge update with a momentum term driven by diagonal Fisher
information estimates. This stabilises updates at low signal-to-noise ratios and
improves convergence on large resolutions.

---

## Training Guide

FastSB-OT assumes you already trained a score network `s_theta(x, t)` on your dataset.
Typical training loop:
1. Train a score model with denoising score matching or noise prediction losses.
2. Export checkpoints compatible with PyTorch (`state_dict` or Lightning weights).
3. Record the noise schedule used during training (cosine, Karras, linear, etc.).

The package does not ship a full trainer, but `fastsb_ot.config.FastSBOTConfig`
captures the solver hyperparameters used during sampling.

---

## How To Use

1) Provide a trained score model (or wrap an epsilon model using `NoisePredictorToScoreWrapper`).
2) Choose a continuous noise schedule via `make_schedule(...)` that matches training.
3) Construct `FastSBOTSolver(score_model, schedule, config=...)`.
4) Call `solver.sample(shape=(B,C,H,W), verbose=True)` or the improved APIs.

```python
import torch
from fastsb_ot import FastSBOTConfig, FastSBOTSolver, make_schedule
from fastsb_ot.utils import NoisePredictorToScoreWrapper

# 1. Load a trained score network
score_model = MyScoreNetwork().to("cuda").eval()
state = torch.load("checkpoints/score_model.pt", map_location="cuda")
score_model.load_state_dict(state)

# 2. Configure solver and schedule
config = FastSBOTConfig(
    quality="balanced",
    use_mixed_precision=True,
    use_triton_kernels=True,
)
schedule = make_schedule("cosine", num_timesteps=config.num_timesteps)

# 2b. (Optional) Wrap epsilonâ€‘predicting models so they return scores
# noise_model = MyEpsilonModel().to("cuda").eval()
# score_model = NoisePredictorToScoreWrapper(noise_model, schedule)

# 3. Construct solver and generate samples
solver = FastSBOTSolver(score_model, schedule, config=config)
samples = solver.sample((8, 4, 256, 256), verbose=True)
```

**Quality presets**
- `quality="fast"`: fewer OT iterations, lower transport ranks.
- `quality="balanced"`: default trade-off for high-res synthesis.
- `quality="ultra"`: full bridge with maximal accuracy (requires high-end GPUs).

### Using noiseâ€‘predicting models

If your trained network outputs the added noise (``epsilon``) rather than the score,
wrap it before constructing the solver:

```python
from fastsb_ot.utils import NoisePredictorToScoreWrapper

noise_model = MyEpsilonModel().to(device).eval()
schedule = make_schedule("cosine", num_timesteps=config.num_timesteps)
score_model = NoisePredictorToScoreWrapper(noise_model, schedule, device=device)
solver = FastSBOTSolver(score_model, schedule, config=config, device=device)
```

Alternatively, set ``predicts_noise = True`` on your module to let the solver convert
the outputs automatically, but the wrapper keeps concerns explicit and re-uses the
same schedule for conversion.

---

## Configuration Highlights

`FastSBOTConfig` exposes the following knobs:

| Param | Description |
|-------|-------------|
| `num_timesteps` | Number of diffusion steps used during sampling. |
| `epsilon` | Sinkhorn entropic regularisation parameter. |
| `use_triton_kernels` | Enable Triton kernels for Gaussian convolution and Fisher estimates. |
| `sliced_ot_projections` | Number of random projections used for sliced OT fallback. |
| `use_fp32_fisher` | Force Fisher updates to FP32 when using AMP to reduce bias. |
| `momentum_alpha` | Strength of momentum correction in the transport update. |
| `memory_limit_ot_mb` | VRAM budget for OT buffers (auto-tunes defaults per preset). |

---

## Mathematical Checks

The solver includes diagnostic routines (see `fastsb_ot/tests` for more examples):
- `solver.self_check()` validates Sinkhorn convergence, Fisher estimator stability,
  and transport symmetry on synthetic data.
- Setting `config.profile=True` records FLOPs, memory usage, and adaptive epsilon traces.

---

## Deployment Notes
- Triton kernels provide the best throughput on Ampere+ GPUs. Disable them when targeting
  older architectures or CPU deployments.
- Use the persistent cache (`fastsb_ot.cache.PersistentKernelCache`) for services that
  instantiate multiple solvers; it stores FFT kernels and Fisher buffers on disk.
- For multi-GPU inference, shard batches manually and reuse the same `FastSBOTConfig`
  to ensure deterministic behaviour across workers.

---

## Testing
```bash
pytest
python -m fastsb_ot.selftest
```
Both commands run lightweight transport and Sinkhorn tests to confirm the installation.

---

## License & Citation
FastSB-OT is distributed under the Apache License 2.0. Retain attribution to Thierry Silvio Claude Soreze in derivative works. If the solver supports your research, cite the repository and acknowledge the author.

---

## Hardware & Resolution Guidance

Indicative limits assuming typical latent-space UNets and mixed precision on CUDA. Effective resolution depends on channels, steps, and preset.

- 6–8 GB (RTX 2060/3060/4060):
  - Up to 512x512, batch 1–2
  - Use quality="fast", fewer OT iterations, smaller momentum buffers
- 10–12 GB (RTX 3080/4070):
  - Up to 1024x1024, batch 1–2
  - quality="balanced", 25–50 steps; Triton kernels recommended
- 16 GB (RTX 4080/4090/A4000):
  - 1024x1024, batch 4–8 (or 1536x1536, batch 1)
  - Enable TF32; Fisher momentum on for stability
- 24 GB (RTX 3090/4090/A5000):
  - 1024x1024, batch 8–16; 1536x1536, batch 1–2
- 32 GB+ (A6000/flagship):
  - 1536x1536, batch 4–8; higher OT ranks and iterations feasible

CPU: use small images (≤256×512) and modest OT settings; expect longer runtime.

Memory tips:
- Reduce num_timesteps, sinkhorn_iterations, or switch to sliced/FFT transports.
- Use use_mixed_precision=True and Triton where available.
- Lower batch size before reducing resolution.

---
