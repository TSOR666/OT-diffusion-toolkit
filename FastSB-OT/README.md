# FastSB-OT Solver Toolkit

FastSB-OT delivers a production-grade Schrodinger bridge sampler tailored to fast
optimal transport updates. It complements trained score networks with entropy-regularised
transport, Fisher-aware momentum, and Triton-accelerated kernels.

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

## Mathematical Foundations

### Schrodinger Bridge Formulation
FastSB-OT solves the dynamic Schrödinger bridge problem: given forward SDE
```
dx = f(x, t) dt + g(t) dW_t,
```
find the most likely path measure connecting the prior and data marginals subject to
an entropic constraint. The bridge dynamics are expressed via a pair of dual potentials
`(phi_t, psi_t)` that correct the score-based drift.

### Entropy-Regularised Optimal Transport
Each iteration solves the regularised OT problem
```
γ* = argmin_γ ⟨C, γ⟩ + ε KL(γ || a ⊗ b),
```
where `C` is the cost matrix (squared Euclidean distance by default) and `a`, `b`
are marginal weights. Sinkhorn iterations with adaptive `ε` provide the transport plan.

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

## Sampling Workflow

```python
import torch
from fastsb_ot import FastSBOTConfig, FastSBOTSolver, make_schedule

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

# 3. Construct solver and generate samples
solver = FastSBOTSolver(score_model, schedule, config=config)
samples = solver.sample((8, 4, 256, 256), verbose=True)
```

**Quality presets**
- `quality="fast"`: fewer OT iterations, lower transport ranks.
- `quality="balanced"`: default trade-off for high-res synthesis.
- `quality="ultra"`: full bridge with maximal accuracy (requires high-end GPUs).

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
