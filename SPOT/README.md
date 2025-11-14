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

## What It Is

- Category: SOLVER — SPOT couples your trained diffusion score network with partial OT updates.
- You bring the score model; SPOT provides transport, integrators, and tooling.
- Deterministic modes and a CLI self-test help validate the environment.

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

## How To Use

1) Provide a trained model that returns a score (or wrap an epsilon model).
2) Choose a noise schedule (`SPOT.schedules.CosineSchedule` or `LinearSchedule`).
3) Build via `SolverBuilder`, then call `sample_enhanced` with a shape and step count.

## Quick Start

```python
import torch
from SPOT.builder import SolverBuilder
from SPOT.utils import NoisePredictorToScoreWrapper
from SPOT.schedules import CosineSchedule

class TinyUNet(torch.nn.Module):
    """Minimal example model that returns a score tensor.

    Expected: forward(x, t) -> score with same shape as x.
    See "Model Input/Output" for details and adapters if your model
    predicts noise (epsilon) instead of score.
    """
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x, t):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_model = TinyUNet().to(device).eval()
noise_schedule = CosineSchedule(device=device)

# Optional: wrap epsilon models so SPOT receives scores
# noise_model = MyEpsModel().to(device).eval()
# score_model = NoisePredictorToScoreWrapper(noise_model, noise_schedule, device=device)

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

## Model Input/Output

By default, SPOT expects a score-based model:

- Input: `x` tensor of shape `(B, C, H, W)` and a time tensor `t` (scalar or shape `(B,)`).
- Output: a tensor with the same shape as `x` representing the score
  `∇_x log p_t(x)`.

The solver uses this score directly in its drift computation. If your model predicts
noise (epsilon) instead of score (common in DDPM/DDIM), use the built-in wrapper:

```python
from SPOT.utils import NoisePredictorToScoreWrapper
from SPOT.schedules import CosineSchedule

noise_schedule = CosineSchedule(device=device)
noise_model = MyEpsModel().to(device).eval()
score_model = NoisePredictorToScoreWrapper(noise_model, noise_schedule, device=device)
```

If your model already returns a score, pass it directly to `SolverBuilder(score_model)`.

---

## Hardware & Resolution Guidance

Indicative limits assuming UNet‑style 3–4 level models and mixed precision on CUDA. Actual maxima depend on model width, channels, steps, and patch size.

- 6–8 GB (RTX 2060/3060/4060):
  - Up to 512x512, batch 1–2, patch_size 32–48
  - Prefer `deterministic=False`, `use_corrector=False` for speed
- 10–12 GB (RTX 3080/4070):
  - Up to 1024x1024, batch 1–2, patch_size 32–64
  - 25–50 steps typical with DPM‑Solver++
- 16 GB (RTX 4080/4090/A4000):
  - 1024x1024, batch 4–8 (or 1536x1536, batch 1)
  - Enable TF32 for speed; keep Richardson off unless needed
- 24 GB (RTX 3090/4090/A5000):
  - 1024x1024, batch 8–16; 1536x1536, batch 1–2
- 32 GB+ (A6000/flagship):
  - 1536x1536, batch 4–8; larger patch sizes (64–96)

CPU: feasible for small demos (≤256–512) with batch 1 using FFT/patched transport; expect slow runtime.

Memory tips:
- Reduce `patch_size` or batch size on OOM.
- Disable CLIP/conditioning to save memory.
- Lower `sinkhorn_iterations` or switch to FFT transport for images.

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

