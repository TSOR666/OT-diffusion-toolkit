"""Self-check helpers for the SBDS solver."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .kernels import KernelDerivativeRFF
from .metrics import MetricsLogger
from .schedule import EnhancedAdaptiveNoiseSchedule, create_standard_timesteps
from .solver import EnhancedScoreBasedSBDiffusionSolver
from .transport import HilbertSinkhornDivergence

__all__ = [
    "test_sbds_implementation",
    "test_mathematical_correctness",
]


def test_sbds_implementation() -> torch.Tensor:
    """Exercise the SBDS solver on a small toy example."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class SimpleScoreModel(nn.Module):
        def __init__(self, dim: int = 2) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, dim),
            )

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_col = t.reshape(-1, 1)
            if x.dim() > 2:
                original_shape = x.shape
                flat_x = x.reshape(x.shape[0], -1)
                xt = torch.cat([flat_x, t_col], dim=1)
                out = self.net(xt)
                return out.reshape(original_shape)
            xt = torch.cat([x, t_col], dim=1)
            return self.net(xt)

    score_model = SimpleScoreModel(dim=2).to(device)
    noise_schedule = EnhancedAdaptiveNoiseSchedule(schedule_type="cosine", device=device)

    solver = EnhancedScoreBasedSBDiffusionSolver(
        score_model=score_model,
        noise_schedule=noise_schedule,
        device=device,
        eps=0.01,
        sb_iterations=3,
        computational_tier="auto",
        use_hilbert_sinkhorn=True,
        debiased_divergence=True,
        selective_sb=True,
    )

    timesteps = create_standard_timesteps(num_steps=50, schedule_type="linear")
    batch_size = 16
    dim = 2
    shape = (batch_size, dim)

    metrics_logger = MetricsLogger()

    print("Starting sampling...")
    samples = solver.sample(
        shape=shape,
        timesteps=timesteps,
        verbose=True,
        metrics_logger=metrics_logger,
    )

    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample statistics: mean={samples.mean():.4f}, std={samples.std():.4f}")

    summary = metrics_logger.get_summary()
    print("\nPerformance metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

    convergence = solver.estimate_convergence_rate(n_samples=batch_size, dim=dim)
    print("\nConvergence rate estimates:")
    for key, value in convergence.items():
        print(f"  {key}: {value:.6f}")

    return samples



def test_mathematical_correctness() -> None:
    """Run a lightweight battery of mathematical sanity checks."""
    print("Running mathematical correctness tests...")

    rff = KernelDerivativeRFF(input_dim=2, feature_dim=128, sigma=1.0)
    x = torch.randn(10, 2)
    y = torch.randn(15, 2)

    K = rff.compute_kernel(x, y)
    dK = rff.compute_kernel_derivative(x, y, order=1)
    assert dK.shape == (2, 10, 15), f"Wrong derivative shape: {dK.shape}"

    h = 1e-5
    for i in range(2):
        x_plus = x.clone()
        x_plus[:, i] += h
        K_plus = rff.compute_kernel(x_plus, y)
        dK_numerical = (K_plus - K) / h
        error = torch.max(torch.abs(dK[i] - dK_numerical))
        print(f"  Derivative error for dim {i}: {error:.6f}")
        assert error < 1e-3, f"Large derivative error: {error}"

    solver = EnhancedScoreBasedSBDiffusionSolver(
        score_model=lambda xx, tt: -xx,
        noise_schedule=lambda t: np.exp(-5 * t),
        device=torch.device("cpu"),
    )

    drift = solver._compute_drift(torch.randn(5, 3), 0.5, 0.01)
    assert drift.shape == (5, 3), f"Wrong drift shape: {drift.shape}"

    large_pos = torch.tensor([100.0])
    large_neg = torch.tensor([-100.0])
    exp_pos = solver._stable_exp(large_pos)
    exp_neg = solver._stable_exp(large_neg)
    assert torch.isfinite(exp_pos)
    assert torch.isfinite(exp_neg)
    assert exp_pos <= torch.exp(torch.tensor(50.0))

    schedule = EnhancedAdaptiveNoiseSchedule()
    schedule._initialize_rff(2)
    x_features = torch.randn(100, 256)
    y_features = torch.randn(100, 256)
    mmd_sq = schedule._compute_mmd(x_features, y_features)
    assert mmd_sq >= 0, "MMD^2 should be non-negative"
    mmd_same = schedule._compute_mmd(x_features, x_features)
    assert mmd_same < 1e-6, "MMD^2 for identical samples should be ~0"

    sinkhorn = HilbertSinkhornDivergence(epsilon=0.1, max_iter=100)
    divergence = sinkhorn.compute_divergence(torch.randn(20, 2), torch.randn(25, 2))
    assert torch.isfinite(divergence)
    assert divergence >= 0
    div_xx = sinkhorn.compute_divergence(torch.randn(20, 2), torch.randn(20, 2))
    assert abs(div_xx) < 1e-4

    print("\nAll mathematical tests passed!")
