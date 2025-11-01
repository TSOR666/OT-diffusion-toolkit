# SPOT &mdash; Schrodinger Partial Optimal Transport Solver

SPOT is a production-ready Schrodinger Partial Optimal Transport (SPOT) sampler
designed for high-fidelity diffusion pipelines. It combines numerically robust
optimal transport kernels, DPM-Solver++ integrators, and patch-based transport
acceleration to deliver deterministic, reproducible sampling on modern GPUs.

## Highlights
- **Production-focused solver** &mdash; hardened configurations for balanced,
  fast, and reproducible deployments.
- **Deterministic modes** &mdash; bit-exact computation paths (including
  Sinkhorn CPU fallbacks) for auditability.
- **Patch-based OT** &mdash; efficient large-image transport via cached norms
  and adaptive Richardson extrapolation.
- **Pluggable schedules and correctors** &mdash; cosine / linear schedules plus
  Langevin, Tweedie, and adaptive correctors (loaded lazily when available).
- **Self-diagnostics** &mdash; `python -m SPOT` runs the integrated validation
  suite to verify numerical assumptions on the target machine.

## Installation
1. Install PyTorch (2.1 or newer recommended) with CUDA, ROCm, or CPU support.
2. (Optional) Install Triton for the accelerated Sinkhorn kernels:
   `pip install triton`.
3. From the repository root:
   ```bash
   cd SPOT
   pip install -e .
   ```

The solver automatically selects CUDA or CPU execution. Mixed precision is
enabled on CUDA devices, while full precision is used elsewhere.

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
print("Generated batch:", samples.shape)
print("Average score latency:", stats["avg_score_time"])

solver.cleanup()  # Restore TF32 flags and release cached resources

```

The convenience helpers `create_balanced_solver`, `create_fast_solver`, and `create_repro_solver` return preconfigured `ProductionSPOTSolver` instances if you prefer not to use the fluent builder.


### CLI self-test
```bash
python -m SPOT
```
The command runs the bundled validation suite and reports whether the solver is
ready for deployment on the current hardware.

## Configuration Guide
- `SPOT.builder.SolverBuilder` offers a fluent API for configuring devices,
  determinism, patch-based transport, Richardson extrapolation, and schedule
  overrides.
- `SPOT.builder.create_balanced_solver`, `create_fast_solver`, and
  `create_repro_solver` provide pre-tuned profiles.
- `SPOT.config.SolverConfig` exposes the full set of knobs (e.g. Sinkhorn
  iterations, mixed precision, adaptive epsilon scaling) for advanced users.
- `SPOT.solver.ProductionSPOTSolver.sample_enhanced` returns either tensors or a
  `SamplingResult` dataclass with execution statistics.

## Testing
- Run `python -m SPOT` for the integrated validation flow.
- Import `SPOT.selftest.selftest` in unit tests to assert that the solver is
  correctly linked against PyTorch and Triton backends.

## License
SPOT is distributed under the Apache License 2.0 (see `LICENSE`). Please retain
proper attribution to Thierry Silvio Claude Soreze in derivative works.



