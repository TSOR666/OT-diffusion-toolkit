# Extending ATLAS

This guide shows how to extend ATLAS with custom components.

## Table of Contents
- [Custom Score Networks](#custom-score-networks)
- [Custom Kernel Operators](#custom-kernel-operators)
- [Custom Noise Schedules](#custom-noise-schedules)
- [Custom Conditioning](#custom-conditioning)
- [Custom Samplers](#custom-samplers)

---

## Custom Score Networks

### Implementing a New Architecture

```python
import torch.nn as nn
from atlas.models.score_network import HighResLatentScoreModel

class CustomScoreNetwork(nn.Module):
    """Custom score network architecture."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Required attributes for compatibility
        self.conditional = config.conditional
        self.use_context = config.use_clip_conditioning

        # Your architecture here
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)

    def forward(self, x, timesteps, condition=None):
        """
        Forward pass must accept:
        - x: [batch, channels, height, width]
        - timesteps: [batch] or scalar
        - condition: Optional conditioning (dict, tensor, or bool)

        Returns:
        - score: [batch, channels, height, width] (same shape as x)
        """
        # Encode timesteps
        t_emb = self.time_embedding(timesteps)

        # Process conditioning
        if condition is not None:
            # Handle dict, tensor, or bool conditioning
            ctx = self._parse_condition(condition, x.shape[0], x.device)
        else:
            ctx = None

        # Forward through architecture
        score = self.encoder(x, t_emb, ctx)
        score = self.decoder(score)

        return score

    def _parse_condition(self, condition, batch, device):
        """Parse different conditioning formats."""
        if isinstance(condition, dict):
            return condition.get("context")
        elif isinstance(condition, torch.Tensor):
            return condition.to(device)
        elif isinstance(condition, bool):
            return self.cond_embedding if condition else self.uncond_embedding
        return None
```

### Using Custom Network

```python
from atlas.config import HighResModelConfig
from atlas.solvers import AdvancedHierarchicalDiffusionSampler

# Your custom config
custom_config = HighResModelConfig(
    in_channels=4,
    out_channels=4,
    # ... your parameters
)

# Instantiate
model = CustomScoreNetwork(custom_config).to(device).eval()

# Use with ATLAS sampler
sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model,
    noise_schedule=karras_noise_schedule,
    device=device,
)
```

---

## Custom Kernel Operators

### Implementing a New Kernel

```python
import torch
from atlas.kernels.base import KernelOperator

class MyCustomKernel(KernelOperator):
    """Custom kernel operator implementation."""

    def __init__(self, epsilon: float, device: torch.device, **kwargs):
        super().__init__(epsilon, device)

        # Your custom parameters
        self.custom_param = kwargs.get("custom_param", 1.0)

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel operator: K[v](x)

        Args:
            x: Input data [batch_size, features]
            v: Vector to apply [batch_size, ...]

        Returns:
            K[v](x): Result [batch_size, ...]
        """
        # Compute kernel matrix
        K = self._compute_kernel_matrix(x, x)

        # Apply to v
        v_flat = v.reshape(v.shape[0], -1)
        result = K @ v_flat

        return result.reshape(v.shape)

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply transpose operator K^T[v](x)."""
        # For symmetric kernels, K^T = K
        return self.apply(x, v)

    def get_error_bound(self, n_samples: int) -> float:
        """Theoretical approximation error bound."""
        # Return error estimate for your kernel
        return 1.0 / n_samples

    def clear_cache(self) -> None:
        """Clear any cached computations."""
        pass

    def _compute_kernel_matrix(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix K[i,j] = k(x1[i], x2[j])."""
        # Example: RBF kernel
        diff = x1[:, None, :] - x2[None, :, :]
        dist_sq = (diff ** 2).sum(-1)
        return torch.exp(-dist_sq / (2 * self.epsilon ** 2))
```

### Registering Custom Kernel

```python
from atlas.solvers.schrodinger_bridge import SchroedingerBridgeSolver

# Monkey-patch into operator selection
original_select = SchroedingerBridgeSolver._select_optimal_kernel_operator

def extended_select(self, x, epsilon):
    if self.solver_type == "custom":
        return MyCustomKernel(
            epsilon=epsilon,
            device=self.device,
            custom_param=self.custom_param,
        )
    return original_select(self, x, epsilon)

SchroedingerBridgeSolver._select_optimal_kernel_operator = extended_select

# Use in config
kernel_config = KernelConfig(
    solver_type="custom",
    epsilon=0.1,
)
kernel_config.custom_param = 2.0  # Add custom parameter
```

---

## Custom Noise Schedules

### Creating a Schedule Function

```python
import numpy as np
import torch

def my_custom_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Custom noise schedule function.

    Args:
        t: Time values in [0, 1], shape [batch] or scalar

    Returns:
        sigma: Noise levels, same shape as t
    """
    # Example: exponential schedule
    t = torch.as_tensor(t, dtype=torch.float32)
    sigma_min, sigma_max = 0.01, 10.0

    # Exponential interpolation
    log_sigma = torch.log(sigma_min) + t * (torch.log(sigma_max) - torch.log(sigma_min))
    return torch.exp(log_sigma)
```

### Using Custom Schedule

```python
from atlas.solvers import AdvancedHierarchicalDiffusionSampler

sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model,
    noise_schedule=my_custom_schedule,  # Your custom schedule
    device=device,
)

# Sample with custom timesteps
timesteps = torch.linspace(1.0, 0.0, 50)  # Or your custom timestep sequence
samples = sampler.sample((4, 4, 64, 64), timesteps)
```

### Common Schedule Patterns

```python
def linear_schedule(t, sigma_min=0.01, sigma_max=10.0):
    """Linear interpolation."""
    return sigma_min + t * (sigma_max - sigma_min)

def cosine_schedule(t, s=0.008):
    """Cosine schedule from improved DDPM."""
    return torch.tan((t + s) / (1 + s) * np.pi / 2)

def karras_schedule(t, sigma_min=0.01, sigma_max=10.0, rho=7.0):
    """Karras et al. schedule (ATLAS default)."""
    t = torch.as_tensor(t)
    return (sigma_max ** (1/rho) + t * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
```

---

## Custom Conditioning

### Implementing Custom Conditioner

```python
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

class MyConditioningInterface:
    """Custom conditioning interface."""

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Load your conditioning model
        self.encoder = self._load_encoder()

    def _load_encoder(self):
        """Load conditioning encoder (e.g., text, image, audio)."""
        # Example: load a pretrained model
        encoder = nn.Sequential(...)
        return encoder.to(self.device).eval()

    def encode(
        self,
        inputs: Any,
        negative_inputs: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode conditioning inputs.

        Args:
            inputs: Raw conditioning inputs (text, images, etc.)
            negative_inputs: Optional negative conditioning

        Returns:
            Dictionary with:
            - "context": Conditioning tensor [batch, seq_len, dim]
            - "context_mask": Optional attention mask [batch, seq_len]
            - "embedding": Optional global embedding [batch, dim]
        """
        with torch.no_grad():
            # Encode positive
            context = self.encoder(inputs)

            # Encode negative if provided
            if negative_inputs is not None:
                neg_context = self.encoder(negative_inputs)
            else:
                neg_context = None

        return {
            "context": context,
            "context_mask": None,  # Optional
            "embedding": None,     # Optional global embedding
            "negative_context": neg_context,
        }

    def prepare_batch_conditioning(
        self,
        inputs: list,
        guidance_scale: float = 7.5,
    ) -> Dict[str, torch.Tensor]:
        """Prepare conditioning for a batch with classifier-free guidance."""
        conditioning = self.encode(inputs)

        # For CFG, duplicate batch
        if guidance_scale != 1.0:
            # Conditional + unconditional
            context = conditioning["context"]
            uncond_context = torch.zeros_like(context)

            conditioning["context"] = torch.cat([uncond_context, context], dim=0)
            conditioning["guidance_scale"] = guidance_scale

        return conditioning
```

### Using Custom Conditioner

```python
from atlas.solvers import AdvancedHierarchicalDiffusionSampler

# Create sampler
sampler = AdvancedHierarchicalDiffusionSampler(...)

# Create and attach custom conditioner
my_conditioner = MyConditioningInterface(config, device)
sampler.set_conditioner(my_conditioner)

# Use in sampling
conditioning = my_conditioner.prepare_batch_conditioning(
    inputs=["input 1", "input 2"],
    guidance_scale=7.5,
)

samples = sampler.sample(
    shape=(2, 4, 64, 64),
    timesteps=timesteps,
    condition=conditioning,
)
```

---

## Custom Samplers

### Implementing a Sampler

```python
from atlas.solvers.base_sampler import BaseSampler
import torch

class MyCustomSampler(BaseSampler):
    """Custom sampling algorithm."""

    def __init__(self, score_model, noise_schedule, device, **kwargs):
        super().__init__(score_model, noise_schedule, device)

        self.custom_param = kwargs.get("custom_param", 1.0)

    def sample(
        self,
        shape: tuple,
        timesteps: int,
        condition=None,
        x_init=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Main sampling loop.

        Args:
            shape: Output shape (batch, channels, height, width)
            timesteps: Number of steps or timestep sequence
            condition: Optional conditioning
            x_init: Optional initialization

        Returns:
            Generated samples
        """
        # Initialize
        if x_init is None:
            x = torch.randn(shape, device=self.device)
        else:
            x = x_init.to(self.device)

        # Get timestep sequence
        if isinstance(timesteps, int):
            ts = torch.linspace(1.0, 0.0, timesteps + 1)
        else:
            ts = torch.as_tensor(timesteps)

        # Sampling loop
        for i in range(len(ts) - 1):
            t_cur = ts[i]
            t_next = ts[i + 1]

            # Get current noise level
            sigma_cur = self.noise_schedule(t_cur)

            # Score network prediction
            score = self.score_model(x, t_cur, condition)

            # Your custom update rule
            x = self._update_step(x, score, sigma_cur, t_next)

        return x

    def _update_step(self, x, score, sigma, t_next):
        """Implement your update rule."""
        # Example: simple Euler step
        sigma_next = self.noise_schedule(t_next)
        dt = sigma_next - sigma

        x_next = x + dt * score
        return x_next
```

### Using Custom Sampler

```python
# Instantiate
sampler = MyCustomSampler(
    score_model=model,
    noise_schedule=karras_noise_schedule,
    device=device,
    custom_param=2.0,
)

# Sample
samples = sampler.sample(
    shape=(4, 4, 64, 64),
    timesteps=50,
    condition=conditioning,
)
```

---

## Testing Custom Components

### Unit Testing

```python
import pytest
import torch

def test_custom_kernel():
    """Test custom kernel operator."""
    kernel = MyCustomKernel(epsilon=0.1, device=torch.device("cpu"))

    # Test shapes
    x = torch.randn(10, 5)
    v = torch.randn(10, 3)

    result = kernel.apply(x, v)
    assert result.shape == v.shape, "Output shape mismatch"

    # Test symmetry (if applicable)
    v_flat = v.reshape(v.shape[0], -1)
    result1 = kernel.apply(x, v)
    result2 = kernel.apply_transpose(x, v)
    torch.testing.assert_close(result1, result2, msg="Kernel not symmetric")

def test_custom_schedule():
    """Test custom noise schedule."""
    t = torch.linspace(0, 1, 100)
    sigma = my_custom_schedule(t)

    # Check monotonicity
    assert torch.all(sigma[1:] <= sigma[:-1]), "Schedule not monotonic"

    # Check bounds
    assert sigma.min() > 0, "Sigma must be positive"
    assert sigma.max() < 100, "Sigma too large"
```

### Integration Testing

```python
def test_end_to_end_generation():
    """Test complete generation pipeline with custom components."""
    # Create components
    model = CustomScoreNetwork(config)
    kernel = MyCustomKernel(epsilon=0.1, device=device)
    sampler = MyCustomSampler(model, my_custom_schedule, device)

    # Generate
    samples = sampler.sample((2, 3, 32, 32), timesteps=10)

    # Verify
    assert samples.shape == (2, 3, 32, 32)
    assert torch.isfinite(samples).all()
    assert samples.std() > 0.1  # Not collapsed
```

---

## Best Practices

### 1. Follow Interfaces

Ensure your custom components match expected interfaces:
- Score networks: `forward(x, timesteps, condition)`
- Kernels: `apply(x, v)`, `apply_transpose(x, v)`
- Schedules: `schedule(t) -> sigma`

### 2. Handle Device/Dtype

Always respect input device and dtype:
```python
def forward(self, x, t):
    # Move to correct device/dtype
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    # ...
```

### 3. Document Parameters

```python
class MyComponent:
    """
    My custom component.

    Args:
        param1: Description
        param2: Description

    Attributes:
        attr1: Description

    Example:
        >>> component = MyComponent(param1=1.0)
        >>> output = component.process(input)
    """
```

### 4. Add Type Hints

```python
from typing import Optional, Tuple
import torch

def process(
    x: torch.Tensor,
    condition: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    ...
```

### 5. Profile Performance

```python
import time
import torch

def benchmark_component(component, input_shape, n_iters=100):
    x = torch.randn(input_shape, device=device)

    # Warmup
    for _ in range(10):
        _ = component(x)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_iters):
        _ = component(x)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Avg time: {elapsed/n_iters*1000:.2f}ms")
```

---

## Contributing Back

If your extension is generally useful, consider contributing it back:

1. **Fork** the ATLAS repository
2. **Create** a feature branch
3. **Implement** with tests and documentation
4. **Submit** a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
