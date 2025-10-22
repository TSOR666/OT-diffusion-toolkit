# SBDS (Score-Based Schrodinger Diffusion Solver)

## Overview
SBDS is a research-oriented implementation of an enhanced score-based
Schrodinger bridge sampler. The library blends probability-flow dynamics
with optimal transport corrections, Hilbert Sinkhorn divergences, and
random Fourier feature (RFF) kernel approximations. The codebase has been
refactored into a structured Python package (`sbds`) to simplify reuse and
experimentation across projects.

## Mathematical Foundations
- **Probability flow ODE**: The sampler integrates the deterministic flow
  `dx/dt = f(x, t) - g(t)**2 * grad_x log p_t(x)` derived from the reverse-time
  SDE, using the learned score network to estimate `grad_x log p_t`.
- **Hilbert Sinkhorn divergence**: Transport updates minimise the entropic
  Sinkhorn divergence in a reproducing kernel Hilbert space, allowing
  smooth control of regularisation via `epsilon` and enabling debiased
  divergence estimates.
- **Random Fourier features**: The `KernelDerivativeRFF` module tracks both
  kernel values and their spatial derivatives using orthogonal RFFs with
  provable error bounds, providing accurate score refinements in high
  dimensions.
- **FFT-accelerated OT**: For grid-structured data the solver switches to an
  FFT-based multi-scale Sinkhorn routine, drastically reducing run time on
  large batches.

## Package Layout
- `sbds/common.py` - shared imports, KeOps availability flags, type helpers.
- `sbds/metrics.py` - runtime metrics logger with FLOP and convergence
  tracking support.
- `sbds/schedule.py` - timestep utilities and
  `EnhancedAdaptiveNoiseSchedule` for RFF-guided adaptive grids.
- `sbds/kernels.py` - kernel derivative approximations with RFFs.
- `sbds/transport.py` - FFT optimal transport and Hilbert Sinkhorn solvers.
- `sbds/solver.py` - `EnhancedScoreBasedSBDiffusionSolver` orchestrating the
  bridge iterations.
- `sbds/testing.py` - smoke tests and mathematical correctness checks.

## Training Strategy
1. Train a score model on the target data distribution using score matching
   or diffusion objectives. The solver expects unconditional scores; wrap
   conditional models externally if needed.
2. Build an `EnhancedAdaptiveNoiseSchedule` to precompute `alpha_bar(t)` and
   identify high-curvature regions via RFF-driven importance sampling.
3. Tune the computational tier:
   - `full` for exact kernels,
   - `rff` for large-scale experiments,
   - `nystrom` or `multiscale` when memory is constrained.
4. Enable Hilbert Sinkhorn only after the score stabilises; start with
   entropy-regularised Sinkhorn to avoid overfitting early iterations.

## Sampling Workflow
```python
import torch
from sbds import (
    EnhancedAdaptiveNoiseSchedule,
    EnhancedScoreBasedSBDiffusionSolver,
)

score_model = MyScoreModel().eval()
noise_schedule = EnhancedAdaptiveNoiseSchedule(schedule_type="cosine")
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    computational_tier="auto",
)

samples = solver.sample(shape=(4, 2, 64, 64), timesteps=50, verbose=True)
```
Use the `metrics_logger` argument to collect timing, transport cost, and
memory statistics during generation. The `testing` module exposes helper
functions to validate kernel derivatives and Sinkhorn convergence on toy
problems.

## License
SBDS is released under the MIT License (see `LICENSE`). Please retain
attribution to **Thierry Silvio Claude Soreze** when redistributing or
extending the project.

## Citation
If the solver informs your research, cite the repository and credit
Thierry Silvio Claude Soreze as the original author.
