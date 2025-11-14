# SBDS — Score-Based Schrödinger Bridge Diffusion (Research Toolkit)

SBDS is a research toolkit focused on Schrödinger bridges and kernel approximations. It augments probability-flow integration with bridge transport and provides multiple computational tiers (full kernels, random Fourier features, Nystrom sketches, FFT transport) and adaptive noise schedules. Use it to explore algorithms and ablations around SB transport and kernel design.

---

## Installation
```bash
pip install .             # core (PyTorch, NumPy)
pip install .[dev]        # tests and linting (pytest, ruff)
```

SBDS assumes PyTorch 2.1+ and runs on CPU or CUDA devices. Triton is optional; the
advanced kernels in SBDS are PyTorch-based.

---

## Mathematical Foundations

### Probability-Flow ODE
The solver integrates the probability-flow ODE associated with the variance-preserving SDE
```
dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dW_t,
```
using the learned score function `s_theta(x, t)`. The deterministic drift is
```
dx/dt = -0.5 * beta(t) * x - beta(t) * s_theta(x, t).
```

### Schrödinger Bridge Transport
A Schrödinger bridge update refines the drift by solving an entropic optimal transport
problem between model samples and reference samples at each timestep. SBDS supports:
- **Full kernel transport** (`O(n^2)`) for small problems.
- **Random Fourier features (RFF)** for large point clouds.
- **Nystrom** low-rank transport for very large batches.
- **FFT transport** for grid-structured data (images/volumes).

### Hilbert Sinkhorn Divergence
To compare distributions, SBDS offers the Hilbert Sinkhorn divergence:
```
- Category: full-stack DIFFUSION RESEARCH TOOLKIT — ships solver, schedules,
- Focus: Schrödinger bridges and kernel approximations (RFF, Nystrom, FFT).
with optional RFF-based cost approximations for high ambient dimensions.

### Adaptive Noise Schedule
`EnhancedAdaptiveNoiseSchedule` provides:
- analytic linear/cosine/quadratic schedules,
- adaptive timestep selection via maximum mean discrepancy (MMD),
- optional signal-to-noise weighting to focus compute on difficult regions.

---

## What It Is

- Category: full-stack DIFFUSION RESEARCH TOOLKIT — ships solver, schedules,
  transport tiers, and diagnostics suitable for standalone experimentation.
- Focus: Schrödinger bridges and kernel approximations (RFF, Nystrom, FFT).

## Training and Preparation

SBDS consumes a trained score network `s_theta(x, t)`; it does not train the network
itself. Typical steps:
1. Train a score network using denoising score matching on your dataset.
2. Save model checkpoints and record the noise schedule used during training.
3. Optionally prepare conditioning modules (e.g., class conditioning) external to SBDS.

---

## Quick Start

```python
import torch
import torch.nn as nn
from sbds import EnhancedScoreBasedSBDiffusionSolver
from sbds.noise_schedule import EnhancedAdaptiveNoiseSchedule
from sbds.utils import create_standard_timesteps

class ScoreNet(nn.Module):
    def __init__(self, data_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, data_dim),
        )

    def forward(self, x, t):
        t = t.reshape(-1, 1)
        return self.net(torch.cat([x, t], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_model = ScoreNet(data_dim=2).to(device).eval()

noise_schedule = EnhancedAdaptiveNoiseSchedule(
    beta_start=1e-4,
    beta_end=0.02,
    schedule_type="cosine",
    num_timesteps=1000,
    device=device,
)

solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,
    device=device,
    computational_tier="auto",
    use_hilbert_sinkhorn=True,
    sb_iterations=3,
)

timesteps = create_standard_timesteps(num_steps=50, schedule_type="linear")
samples = solver.sample(shape=(64, 2), timesteps=timesteps, verbose=True)
print("Samples:", samples.shape)
```

---

## Computational Tiers

SBDS automatically selects a transport tier based on batch size and data dimension.

| Tier        | Complexity      | Suitable for                      |
|-------------|-----------------|----------------------------------|
| `full`      | `O(n^2)`        | Batches up to ~500k elements     |
| `rff`       | `O(nk)`         | Mid-sized point clouds           |
| `nystrom`   | `O(nm)`         | Large batches/high dimensions    |
| `multiscale`| `O(n log n)`    | Extremely large problems         |
| `fft`       | `O(n log n)`    | Grid-structured data (images)    |

Manual override:
```python
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,
    computational_tier="rff",
    rff_features=2048,
)
```

---

## Noise Scheduling

`EnhancedAdaptiveNoiseSchedule` supports analytic schedules and adaptive refinement:
```python
adaptive_steps = noise_schedule.get_adaptive_timesteps(
    n_steps=50,
    score_model=score_model,
    device=device,
    shape=(8, 2),
    snr_weighting=True,
)
samples = solver.sample(shape=(8, 2), timesteps=adaptive_steps, verbose=False)
```

---

## Diagnostics and Testing

SBDS includes a mathematical correctness harness (`test_mathematical_correctness` in
`sbds/solver.py`). It verifies:
- kernel derivative accuracy via finite differences,
- probability-flow drift implementation,
- stability of logarithm/exponential stabilisers,
- MMD estimates in the adaptive schedule,
- Sinkhorn convergence and debiasing.

Run the full suite:
```bash
pytest
python -m sbds.solver  # executes the built-in mathematical tests and demo
```

---

## Best Practices

- 6–8 GB: 256x256, batch 2–4
- 10–12 GB: 512x512, batch 1–2
- 16 GB: 1024x1024, batch 1–2
- 24 GB+: 1024x1024, batch 4–8 (or 1536x1536, batch 1)

- Prefer RFF (1024–4096 features) for mid-to-large sets; Nystrom for low-dimensional large N.
- Full kernels are O(n²) — keep batches small (<50k) or switch tiers.
- Lower resolution or point count before shrinking batch size.
- Reduce `rff_features` / rank, decrease OT iterations, or switch tiers when OOM.
- Use `adaptive_eps` and fewer steps to reduce peak usage.
---

---

## Hardware & Resolution Guidance

Guidance varies with transport tier and data type (images vs. point clouds). The figures below assume mixed precision on CUDA and typical configs.

Images (FFT transport):
- 6–8 GB: 256x256, batch 2–4
- 10–12 GB: 512x512, batch 1–2
- 16 GB: 1024x1024, batch 1–2
- 24 GB+: 1024x1024, batch 4–8 (or 1536x1536, batch 1)

Point clouds / vectors:
- Prefer RFF (1024–4096 features) for mid-to-large sets; Nystrom for low-dimensional large N.
- Full kernels are O(n²) — keep batches small (<50k) or switch tiers.

Memory tips:
- Lower ff_features / rank, reduce OT iterations, or switch tiers when OOM.
- Use daptive_eps and fewer steps to reduce peak usage.

---
