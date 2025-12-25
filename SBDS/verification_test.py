"""
Verification script for SBDS module correctness.

This script tests the corrected code with real torch operations and validates:
1. Output shapes match expectations
2. All outputs are finite (no NaN/Inf)
3. Gradients are finite when applicable
4. Mathematical correctness of key computations
"""

from __future__ import annotations

import math
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("\\", 1)[0] if "\\" in str(__file__) else str(__file__).rsplit("/", 1)[0])

from sbds.kernel import KernelDerivativeRFF
from sbds.fft_ot import FFTOptimalTransport
from sbds.sinkhorn import HilbertSinkhornDivergence
from sbds.noise_schedule import EnhancedAdaptiveNoiseSchedule
from sbds.solver import EnhancedScoreBasedSBDiffusionSolver
from sbds.utils import create_standard_timesteps, spectral_gradient


def check_finite(tensor: torch.Tensor, name: str) -> None:
    """Assert tensor contains only finite values."""
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise AssertionError(
            f"{name} contains non-finite values: {nan_count} NaN, {inf_count} Inf"
        )


def check_shape(tensor: torch.Tensor, expected: Tuple[int, ...], name: str) -> None:
    """Assert tensor has expected shape."""
    if tensor.shape != expected:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}"
        )


class SimpleScoreModel(nn.Module):
    """Simple score model for testing."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D), t: (B,)
        t = t.reshape(-1, 1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def test_noise_schedule_beta() -> None:
    """Test that get_beta returns correct values for cosine schedule."""
    print("Testing noise schedule beta computation...")

    schedule = EnhancedAdaptiveNoiseSchedule(schedule_type="cosine")

    # Test at t=0.5 (mid-point)
    t = 0.5
    s = 0.008
    arg = (t + s) / (1 + s) * np.pi / 2
    expected_beta = np.pi * np.tan(arg) / (1 + s)
    computed_beta = schedule.get_beta(t)

    rel_error = abs(computed_beta - expected_beta) / expected_beta
    if rel_error > 1e-6:
        raise AssertionError(
            f"Beta computation error: expected {expected_beta:.6f}, got {computed_beta:.6f}"
        )

    # Test at multiple points
    for t_val in [0.1, 0.3, 0.7, 0.9]:
        beta = schedule.get_beta(t_val)
        if not math.isfinite(beta) or beta < 0:
            raise AssertionError(f"Invalid beta at t={t_val}: {beta}")

    print("  Beta computation: PASS")


def test_kernel_rff() -> None:
    """Test KernelDerivativeRFF output shapes and finiteness."""
    print("Testing KernelDerivativeRFF...")

    input_dim = 8
    feature_dim = 256
    batch_x = 10
    batch_y = 15

    rff = KernelDerivativeRFF(
        input_dim=input_dim,
        feature_dim=feature_dim,
        sigma=1.0,
        orthogonal=True,
        derivative_order=2,
    )

    x = torch.randn(batch_x, input_dim)
    y = torch.randn(batch_y, input_dim)

    # Test features
    features = rff.compute_features(x)
    check_shape(features, (batch_x, feature_dim), "features")
    check_finite(features, "features")
    print(f"  Features shape: {features.shape} - PASS")

    # Test kernel
    K = rff.compute_kernel(x, y)
    check_shape(K, (batch_x, batch_y), "kernel")
    check_finite(K, "kernel")
    print(f"  Kernel shape: {K.shape} - PASS")

    # Test first derivative: shape should be (input_dim, batch_x, batch_y)
    dK = rff.compute_kernel_derivative(x, y, order=1)
    check_shape(dK, (input_dim, batch_x, batch_y), "first derivative")
    check_finite(dK, "first derivative")
    print(f"  First derivative shape: {dK.shape} - PASS")

    # Test score approximation
    score = rff.compute_score_approximation(x, y)
    check_shape(score, (batch_x, input_dim), "score approximation")
    check_finite(score, "score approximation")
    print(f"  Score approximation shape: {score.shape} - PASS")


def test_fft_ot() -> None:
    """Test FFTOptimalTransport with grid-structured data."""
    print("Testing FFTOptimalTransport...")

    fft_ot = FFTOptimalTransport(
        epsilon=0.1,
        max_iter=50,
        kernel_type="gaussian",
        multiscale=True,
        scale_levels=2,
    )

    # Create 2D grid densities
    grid_size = 16
    mu = torch.rand(grid_size, grid_size)
    mu = mu / mu.sum()
    nu = torch.rand(grid_size, grid_size)
    nu = nu / nu.sum()

    # Run Sinkhorn
    objective, u, v = fft_ot._sinkhorn_fft(mu, nu, epsilon=0.1)

    check_finite(objective, "FFT-OT objective")
    check_finite(u, "FFT-OT dual u")
    check_finite(v, "FFT-OT dual v")
    check_shape(u, (grid_size, grid_size), "dual potential u")
    check_shape(v, (grid_size, grid_size), "dual potential v")

    print(f"  Objective: {objective.item():.6f} - PASS")
    print(f"  Dual potentials shape: {u.shape} - PASS")

    # Test gradient computation
    gradients = fft_ot._compute_gradient_on_grid(u)
    if len(gradients) != 2:
        raise AssertionError(f"Expected 2 gradients for 2D grid, got {len(gradients)}")
    for i, grad in enumerate(gradients):
        check_finite(grad, f"gradient {i}")
    print(f"  Gradients computed: {len(gradients)} axes - PASS")


def test_sinkhorn_divergence() -> None:
    """Test HilbertSinkhornDivergence."""
    print("Testing HilbertSinkhornDivergence...")

    # Test that cauchy kernel is now properly rejected
    try:
        _ = HilbertSinkhornDivergence(kernel_type="cauchy")
        raise AssertionError("Should have rejected cauchy kernel")
    except ValueError as e:
        if "cauchy" not in str(e).lower():
            raise
        print("  Cauchy kernel rejection: PASS")

    sinkhorn = HilbertSinkhornDivergence(
        epsilon=0.1,
        max_iter=100,
        kernel_type="gaussian",
        debiased=True,
        use_rff=True,
        rff_features=256,
    )

    x = torch.randn(20, 4)
    y = torch.randn(25, 4)

    divergence = sinkhorn.compute_divergence(x, y)
    check_finite(divergence, "divergence")

    # Debiased divergence of X with itself should be ~0
    div_xx = sinkhorn.compute_divergence(x, x)
    if abs(div_xx.item()) > 1e-3:
        raise AssertionError(f"Self-divergence should be ~0, got {div_xx.item():.6f}")

    print(f"  Divergence X-Y: {divergence.item():.6f} - PASS")
    print(f"  Self-divergence: {div_xx.item():.8f} - PASS")


def test_solver_sampling() -> None:
    """Test EnhancedScoreBasedSBDiffusionSolver sampling."""
    print("Testing EnhancedScoreBasedSBDiffusionSolver...")

    device = torch.device("cpu")
    dim = 4
    batch_size = 8

    score_model = SimpleScoreModel(dim).to(device)
    noise_schedule = EnhancedAdaptiveNoiseSchedule(schedule_type="cosine")

    solver = EnhancedScoreBasedSBDiffusionSolver(
        score_model=score_model,
        noise_schedule=noise_schedule,
        device=device,
        eps=0.1,
        sb_iterations=2,
        computational_tier="rff",
        use_hilbert_sinkhorn=True,
        corrector_steps=0,
        rff_features=128,
        use_fft_ot=False,  # Disable to avoid batch warnings
    )

    timesteps = create_standard_timesteps(num_steps=10, schedule_type="linear")
    shape = (batch_size, dim)

    samples = solver.sample(
        shape=shape,
        timesteps=timesteps,
        verbose=False,
    )

    check_shape(samples, shape, "samples")
    check_finite(samples, "samples")
    print(f"  Sample shape: {samples.shape} - PASS")
    print(f"  Sample mean: {samples.mean().item():.4f}, std: {samples.std().item():.4f} - PASS")


def test_solver_drift_computation() -> None:
    """Test that drift computation uses correct device."""
    print("Testing solver drift computation...")

    device = torch.device("cpu")
    dim = 4
    batch_size = 4

    score_model = SimpleScoreModel(dim).to(device)
    noise_schedule = EnhancedAdaptiveNoiseSchedule(schedule_type="cosine")

    solver = EnhancedScoreBasedSBDiffusionSolver(
        score_model=score_model,
        noise_schedule=noise_schedule,
        device=device,
        eps=0.1,
    )

    x = torch.randn(batch_size, dim, device=device)
    t = 0.5
    dt = 0.1

    drift = solver._compute_drift(x, t, dt)

    check_shape(drift, (batch_size, dim), "drift")
    check_finite(drift, "drift")
    if drift.device != x.device:
        raise AssertionError(f"Drift device mismatch: {drift.device} vs {x.device}")

    print(f"  Drift shape: {drift.shape} - PASS")
    print(f"  Drift device: {drift.device} - PASS")


def test_spectral_gradient() -> None:
    """Test spectral gradient computation."""
    print("Testing spectral_gradient...")

    # 2D test
    u = torch.randn(8, 16, 16)
    grads = spectral_gradient(u, grid_spacing=[1.0, 1.0])

    if len(grads) != 2:
        raise AssertionError(f"Expected 2 gradients, got {len(grads)}")

    for i, g in enumerate(grads):
        check_shape(g, u.shape, f"gradient {i}")
        check_finite(g, f"gradient {i}")

    print(f"  2D gradient shapes: {[g.shape for g in grads]} - PASS")


def test_backward_pass() -> None:
    """Test that gradients flow correctly through the solver components."""
    print("Testing backward pass...")

    rff = KernelDerivativeRFF(input_dim=4, feature_dim=64, sigma=1.0)
    x = torch.randn(8, 4, requires_grad=True)
    y = torch.randn(10, 4)

    K = rff.compute_kernel(x, y)
    loss = K.sum()
    loss.backward()

    if x.grad is None:
        raise AssertionError("Gradient not computed for input")
    check_finite(x.grad, "input gradient")

    print(f"  Gradient computed: shape {x.grad.shape} - PASS")


def main() -> None:
    """Run all verification tests."""
    print("=" * 60)
    print("SBDS Verification Test Suite")
    print("=" * 60)

    try:
        test_noise_schedule_beta()
        test_kernel_rff()
        test_fft_ot()
        test_sinkhorn_divergence()
        test_solver_drift_computation()
        test_solver_sampling()
        test_spectral_gradient()
        test_backward_pass()

        print("\n" + "=" * 60)
        print("VERIFICATION PASSED")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"VERIFICATION FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
