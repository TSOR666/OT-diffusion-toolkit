# SBDS: Score-Based Schrödinger Bridge Diffusion Solver

A production-grade implementation of Schrödinger Bridge diffusion solvers with advanced kernel methods, optimal transport, and adaptive scheduling.

## Overview

SBDS implements a state-of-the-art diffusion solver that combines:
- **Score-based modeling** for probability flow ODE dynamics
- **Schrödinger Bridge** optimal transport for efficient sample generation
- **Multiple computational tiers** for different problem scales
- **Advanced kernel approximations** (RFF, FFT, Nyström)
- **Adaptive noise scheduling** with importance-weighted timestep selection

## Installation

The package is part of the `solvers` module:

```python
from solvers import sbds
```

## Quick Start

```python
import torch
import torch.nn as nn
from solvers.sbds import (
    EnhancedScoreBasedSBDiffusionSolver,
    EnhancedAdaptiveNoiseSchedule,
    create_standard_timesteps,
)

# 1. Define or load your score model
class MyScoreModel(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim)
        )

    def forward(self, x, t):
        # x: [batch, data_dim], t: [batch]
        t = t.reshape(-1, 1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# 2. Create noise schedule
noise_schedule = EnhancedAdaptiveNoiseSchedule(
    beta_start=1e-4,
    beta_end=0.02,
    schedule_type='cosine',
    num_timesteps=1000
)

# 3. Initialize solver
score_model = MyScoreModel(data_dim=2).cuda()
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,
    device=torch.device('cuda'),
    computational_tier='auto',  # Automatically select best method
    use_hilbert_sinkhorn=True,
    selective_sb=True,
)

# 4. Generate samples
timesteps = create_standard_timesteps(num_steps=50, schedule_type='linear')
samples = solver.sample(
    shape=(64, 2),  # 64 samples of 2D data
    timesteps=timesteps,
    verbose=True
)

print(f"Generated samples shape: {samples.shape}")
print(f"Sample statistics: mean={samples.mean():.4f}, std={samples.std():.4f}")
```

## Features

### 1. Multiple Computational Tiers

SBDS automatically selects the optimal computational method based on problem size:

| Tier | Complexity | Use Case | When Used |
|------|-----------|----------|-----------|
| **full** | O(n²) | Small batches | <500K elements |
| **rff** | O(nk) | Medium batches | 500K-5M elements |
| **nystrom** | O(nm) | Large batches | 5M-50M elements |
| **multiscale** | O(n log n) | Very large batches | >50M elements |
| **fft** | O(n log n) | Grid-structured data | Images/volumes |

```python
# Manual tier selection
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,
    computational_tier='rff',  # Force RFF method
    rff_features=2048,  # Number of random features
)
```

### 2. FFT-Based Optimal Transport for Images

For grid-structured data (images, volumes), SBDS uses efficient FFT-based transport:

```python
# Generate image samples
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=image_score_model,
    noise_schedule=noise_schedule,
    use_fft_ot=True,  # Enable FFT optimal transport
    multiscale=True,
    scale_levels=3,
)

# Sample images
image_samples = solver.sample(
    shape=(16, 3, 32, 32),  # 16 RGB images, 32x32
    timesteps=timesteps,
)
```

### 3. Adaptive Timestep Selection

Select timesteps based on score function magnitude and distributional changes:

```python
# Create adaptive schedule
noise_schedule = EnhancedAdaptiveNoiseSchedule(
    schedule_type='cosine',
    use_mmd=True,  # Use MMD for timestep importance
)

# Get adaptive timesteps
adaptive_timesteps = noise_schedule.get_adaptive_timesteps(
    n_steps=50,
    score_model=score_model,
    device=torch.device('cuda'),
    shape=(64, 2),
    snr_weighting=True,  # Weight by signal-to-noise ratio
)

# Use with solver
samples = solver.sample(
    shape=(64, 2),
    timesteps=adaptive_timesteps,
)
```

### 4. Performance Monitoring

Track performance metrics during sampling:

```python
from solvers.sbds import MetricsLogger

# Create logger
logger = MetricsLogger(log_file='metrics.json')

# Sample with logging
samples = solver.sample(
    shape=(64, 2),
    timesteps=timesteps,
    metrics_logger=logger,
    enable_profiling=True,  # Enable detailed profiling
)

# Get performance summary
summary = logger.get_summary()
print(f"Average step time: {summary['avg_step_time']:.4f}s")
print(f"Peak memory: {summary['peak_memory_gb']:.2f}GB")
print(f"Average transport cost: {summary['avg_transport_cost']:.4f}")
```

### 5. Corrector Steps (Langevin MCMC)

Refine samples using Langevin dynamics:

```python
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,
    corrector_steps=5,  # Number of Langevin steps
    corrector_snr=0.1,  # Signal-to-noise ratio
)
```

### 6. Selective Schrödinger Bridge

Use full SB only at critical timesteps for efficiency:

```python
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,
    selective_sb=True,  # Enable selective SB
    critical_thresholds=[0.9, 0.5, 0.1],  # Critical timesteps
    sb_iterations=5,  # SB iterations at critical points
)
```

## Configuration Options

### Main Solver Parameters

```python
solver = EnhancedScoreBasedSBDiffusionSolver(
    score_model=score_model,
    noise_schedule=noise_schedule,

    # Device configuration
    device=torch.device('cuda'),
    use_mixed_precision=True,

    # Transport configuration
    eps=0.01,  # Entropy regularization
    adaptive_eps=True,  # Adapt epsilon based on noise level
    sb_iterations=3,  # Number of SB iterations

    # Computational tier
    computational_tier='auto',  # 'auto', 'full', 'rff', 'nystrom', 'multiscale'

    # Kernel configuration
    kernel_bandwidth=1.0,
    rff_features=1024,  # For RFF method
    num_landmarks=100,  # For Nyström method
    score_guided_landmarks=True,  # Use score for landmark selection

    # FFT-OT configuration
    use_fft_ot=True,  # Enable FFT optimal transport
    multiscale_levels=3,  # Number of multiscale levels

    # Optimization
    early_stopping=True,
    early_stopping_tol=1e-4,
    selective_sb=True,
    warm_start_potentials=True,

    # Corrector
    corrector_steps=0,  # Number of Langevin MCMC steps
    corrector_snr=0.1,

    # Numerical stability
    chunk_size=128,  # Chunk size for pairwise computations
)
```

## Mathematical Background

### Probability Flow ODE

SBDS implements the corrected probability flow ODE:

```
dX_t = [-(β(t)/2)X_t - β(t)σ²(t)∇log p(X_t|t)] dt
```

where:
- `β(t) = -d/dt log α_bar(t)` is the time-dependent noise schedule
- `σ²(t) = 1 - α_bar(t)` is the variance at time t
- `∇log p(X_t|t)` is the score function

### Schrödinger Bridge

The Schrödinger Bridge formulation finds the optimal transport plan that minimizes:

```
min_{π ∈ Π(μ,ν)} ∫∫ c(x,y) dπ(x,y) + ε·KL(π || k(x,y))
```

where:
- `μ, ν` are source and target distributions
- `c(x,y)` is the transport cost (typically L2 distance)
- `ε` is the entropy regularization parameter
- `k(x,y)` is a reference kernel

### Convergence Guarantees

Theoretical convergence rates:

- **Score estimation**: O(n^{-1/2})
- **RFF approximation**: O(D^{-1/2}) where D is the number of features
- **Sinkhorn convergence**: O(exp(-t/ε)) exponential in iterations
- **Nyström approximation**: O(m^{-1/2}) where m is the number of landmarks

```python
# Estimate convergence rates
convergence = solver.estimate_convergence_rate(
    n_samples=batch_size,
    dim=data_dim
)
print(f"Score estimation error: {convergence['score_estimation']:.6f}")
print(f"RFF approximation error: {convergence['rff_approximation']:.6f}")
print(f"Total error bound: {convergence['total_bound']:.6f}")
```

## Advanced Usage

### Custom Timestep Schedules

```python
# Linear schedule
timesteps = create_standard_timesteps(num_steps=100, schedule_type='linear')

# Quadratic schedule (more steps at high noise)
timesteps = create_standard_timesteps(num_steps=100, schedule_type='quadratic')

# Logarithmic schedule
timesteps = create_standard_timesteps(num_steps=100, schedule_type='log')

# Custom schedule
custom_timesteps = [1.0, 0.9, 0.8, 0.5, 0.3, 0.1, 0.05, 0.01, 0.0]
```

### Callback Functions

Monitor or modify samples during generation:

```python
def my_callback(t, x_t):
    """Called after each timestep."""
    print(f"Timestep {t:.4f}: min={x_t.min():.4f}, max={x_t.max():.4f}")
    # Optionally: save intermediate results, visualize, etc.

samples = solver.sample(
    shape=(64, 2),
    timesteps=timesteps,
    callback=my_callback,
)
```

### Spectral Gradient Computation

For smoother gradients on grid data:

```python
from solvers.sbds import spectral_gradient

# Compute spectral gradient
grad = spectral_gradient(
    u,  # Input tensor
    grid_spacing=[1.0, 1.0],  # Grid spacing
    apply_filter=True,  # Apply spectral filtering
)
```

## Performance Tips

1. **Choose appropriate tier**: Use `computational_tier='auto'` for automatic selection
2. **Batch size**: Larger batches (32-128) are more efficient
3. **Mixed precision**: Enable for ~2x speedup on modern GPUs
4. **Selective SB**: Use `selective_sb=True` to reduce computation
5. **FFT for images**: Always use `use_fft_ot=True` for grid-structured data
6. **RFF features**: More features (1024-4096) improve accuracy but increase memory
7. **Timesteps**: Fewer timesteps (20-50) are often sufficient

## Benchmarks

Performance on various problem sizes (NVIDIA A100 GPU):

| Problem Size | Batch | Dims | Tier | Time/Step | Memory |
|--------------|-------|------|------|-----------|--------|
| Small | 64 | 2 | full | 5ms | 0.1GB |
| Medium | 256 | 32 | rff | 15ms | 0.5GB |
| Large | 128 | 256 | rff | 45ms | 2.0GB |
| Image | 32 | 3×64×64 | fft | 80ms | 1.5GB |
| Very Large | 512 | 256 | nystrom | 120ms | 4.0GB |

## API Reference

### Main Classes

- **`EnhancedScoreBasedSBDiffusionSolver`**: Main solver class
- **`EnhancedAdaptiveNoiseSchedule`**: Adaptive noise schedule with MMD-based timestep selection
- **`KernelDerivativeRFF`**: Random Fourier Features with kernel derivatives
- **`HilbertSinkhornDivergence`**: Hilbert space Sinkhorn divergence
- **`FFTOptimalTransport`**: FFT-based optimal transport for grid data
- **`MetricsLogger`**: Performance metrics logging

### Utility Functions

- **`create_standard_timesteps`**: Create standard timestep schedules
- **`spectral_gradient`**: Compute spectral gradients using FFT

## Troubleshooting

### Out of Memory Errors

```python
# Reduce batch size
samples = solver.sample(shape=(32, data_dim), ...)  # Instead of 128

# Reduce RFF features
solver = EnhancedScoreBasedSBDiffusionSolver(
    rff_features=512,  # Instead of 2048
    ...
)

# Use multiscale tier for very large problems
solver = EnhancedScoreBasedSBDiffusionSolver(
    computational_tier='multiscale',
    ...
)
```

### Numerical Instability

```python
# Increase entropy regularization
solver = EnhancedScoreBasedSBDiffusionSolver(
    eps=0.1,  # Increase from 0.01
    ...
)

# Enable adaptive epsilon
solver = EnhancedScoreBasedSBDiffusionSolver(
    adaptive_eps=True,
    ...
)

# Use more SB iterations
solver = EnhancedScoreBasedSBDiffusionSolver(
    sb_iterations=10,  # Increase from 3
    ...
)
```

### Slow Convergence

```python
# Use adaptive timesteps
adaptive_timesteps = noise_schedule.get_adaptive_timesteps(
    n_steps=50,
    score_model=score_model,
    device=device,
    shape=shape,
)

# Enable warm starting
solver = EnhancedScoreBasedSBDiffusionSolver(
    warm_start_potentials=True,
    ...
)

# Use appropriate computational tier
solver = EnhancedScoreBasedSBDiffusionSolver(
    computational_tier='rff',  # Faster than 'full' for medium problems
    ...
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sbds2025,
  title={SBDS: Score-Based Schrödinger Bridge Diffusion Solver},
  author={Thierry Silvio Claude Soreze},
  year={2025},
  url={https://github.com/TSOR666/OT-diffusion-toolkit}
}
```

## License

See the main repository LICENSE file.

## Contributing

Contributions are welcome! Please see the main repository contributing guidelines.

## References

1. Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
2. De Bortoli, V., et al. "Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling." NeurIPS 2021.
3. Cuturi, M. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NIPS 2013.
4. Rahimi, A. & Recht, B. "Random Features for Large-Scale Kernel Machines." NIPS 2007.
