# SPOT – Schrodinger Partial Optimal Transport Solver

SPOT is a Schrodinger bridge sampler specialised for partial optimal transport. It
pairs DPM-Solver++ integrators with patch-based transport maps, deterministic
execution modes, and a command-line validation utility.

---

## Installation
```bash
pip install .             # core (PyTorch, NumPy, tqdm)
pip install .[dev]        # tests, linting
pip install triton        # optional: accelerates Sinkhorn kernels
```

---

## Mathematical Overview

### Partial Optimal Transport
Partial OT seeks a transport plan `π` that moves only a fraction of mass between
two distributions. SPOT implements an entropy-regularised objective:
```
min_π ⟨C, π⟩ + ε KL(π || a ⊗ b) + λ (||π 1 - a||_1 + ||π^T 1 - b||_1),
```
where `λ` controls the unbalanced mass penalty. The solution feeds the drift of the
probability-flow ODE.

### DPM-Solver++ Integrators
Probability-flow integration is handled by DPM-Solver++ variants (order 1/2/3). Each
step applies:
1. Predictor update (Heun, exponential, or adaptive integrators).
2. Optional corrector (Langevin, Tweedie, or adaptive correctors).
3. Partial OT barycentric projection using the current transport plan.

### Patch-Based Transport
For grid-structured data, SPOT tiles the domain into overlapping patches, computes
local transport maps, and blended barycentric updates. This reduces the dimensionality
of each transport solve while preserving global structure.

---

## Quick Start

```python
import torch
from SPOT.builder import SolverBuilder
from SPOT.schedules import CosineSchedule

class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x, t):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_model = TinyUNet().to(device).eval()
noise_schedule = CosineSchedule(device=device)

solver = (
    SolverBuilder(score_model)
    .with_device(device)
    .with_compute_dtype(torch.float32)
    .with_noise_schedule(noise_schedule)
    .with_dpm_solver_order(3)
    .with_patch_based_ot(True, patch_size=32)
    .build()
)

samples, stats = solver.sample_enhanced(
    shape=(4, 3, 64, 64),
    num_steps=30,
    return_stats=True,
    seed=42,
)
print("Samples:", samples.shape)
print("Average score latency:", stats["avg_score_time"])

solver.cleanup()
```

**Convenience helpers**
```python
from SPOT.builder import create_balanced_solver, create_fast_solver, create_repro_solver

balanced_solver = create_balanced_solver(score_model)
fast_solver = create_fast_solver(score_model)
repro_solver = create_repro_solver(score_model)
```

---

## Configuration Notes
- `patch_size` controls the local OT granularity; 32–64 works well for 512x512 images.
- `richardson_threshold` governs when Richardson extrapolation fires; tightening improves accuracy at the cost of time.
- `deterministic=True` enforces bit-exact execution (Sinkhorn CPU fallbacks, TF32 disabled).
- `with_tf32(True)` enables TF32 matrix multiplications on Ampere+ GPUs for speed.

---

## CLI Self-Test
SPOT ships with a validation flow to ensure numerical assumptions hold on the target machine:
```bash
python -m SPOT
```
The command runs kernel checks, transport accuracy tests, and reports deterministic mode readiness.

---

## Mathematical Diagnostics
`solver.test_mathematical_correctness()` (available within `SPOT/solver.py`) performs:
- kernel derivative finite-difference checks,
- transport idempotence tests under partial mass,
- determinism verification (GPU/CPU parity),
- Richardson extrapolation overhead monitoring.

---

## Training Integration
SPOT consumes a trained score network; training is handled outside the package. When
adapting a new model:
1. Train the score model with variance-preserving noise.
2. Export checkpoints and match the compute dtype used during sampling (FP32 or AMP).
3. Wrap the model with any required conditioning (e.g., classifier-free guidance).

---

## Troubleshooting
- **OOM during transport**: reduce `patch_size`, increase `lambda_unbalanced`, or lower batch size.
- **Slow convergence**: increase `richardson_max_overhead` or `corrector_steps`.
- **Determinism mismatches**: ensure `with_deterministic(True, cdist_cpu=True)` and run `python -m SPOT` to validate.

---

## License & Citation
SPOT is distributed under the Apache License 2.0. Retain attribution to Thierry Silvio Claude Soreze in derivative works. Cite the repository if SPOT supports your research.
