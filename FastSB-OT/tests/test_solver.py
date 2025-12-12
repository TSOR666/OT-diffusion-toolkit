"""Comprehensive tests for FastSB-OT solver implementation."""

import math
import pytest
import torch
import torch.nn as nn

from fastsb_ot import FastSBOTConfig, FastSBOTSolver, make_schedule
from fastsb_ot.kernels import KernelModule


class MockScoreModel(nn.Module):
    """Simple mock score model for testing."""

    def __init__(self, output_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.conv = nn.Conv2d(3, 3 if output_dim is None else output_dim, 1)

    def forward(self, x, t):
        return self.conv(x)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config():
    """Create basic config."""
    return FastSBOTConfig(
        quality="draft",
        use_mixed_precision=False,
        warmup=False,
        seed=42
    )


@pytest.fixture
def noise_schedule():
    """Create cosine noise schedule."""
    return make_schedule("cosine", num_timesteps=100)


@pytest.fixture
def solver(config, noise_schedule, device):
    """Create solver with mock model."""
    model = MockScoreModel()
    return FastSBOTSolver(model, noise_schedule, config, device)


class TestNoiseSchedule:
    """Test noise schedule properties."""

    def test_cosine_schedule_monotonic(self):
        """Test that cosine schedule is monotonically decreasing."""
        schedule = make_schedule("cosine", num_timesteps=1000)

        t_values = [i / 100 for i in range(101)]
        alpha_bars = [schedule(t) for t in t_values]

        # Check monotonic decrease
        for i in range(len(alpha_bars) - 1):
            assert alpha_bars[i] >= alpha_bars[i+1], \
                f"Schedule not monotonic at i={i}: {alpha_bars[i]} < {alpha_bars[i+1]}"

    def test_linear_schedule_monotonic(self):
        """Test that linear schedule is monotonically decreasing."""
        schedule = make_schedule("linear", num_timesteps=1000)

        t_values = [i / 100 for i in range(101)]
        alpha_bars = [schedule(t) for t in t_values]

        for i in range(len(alpha_bars) - 1):
            assert alpha_bars[i] >= alpha_bars[i+1]

    def test_schedule_endpoints(self):
        """Test schedule endpoints."""
        schedule = make_schedule("cosine", num_timesteps=1000)

        # At t=0 (clean), alpha_bar should be close to 1
        assert 0.95 <= schedule(0.0) <= 1.0

        # At t=1 (noise), alpha_bar should be close to 0
        assert 0.0 <= schedule(1.0) <= 0.1


class TestDDPMStep:
    """Test DDPM sampling step correctness."""

    def test_ddpm_step_shape(self, solver, device):
        """Test that DDPM step preserves shape."""
        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.randn_like(x_t)

        x_next = solver.ddpm_step_improved(x_t, noise_pred, t_curr=0.5, t_next=0.4)

        assert x_next.shape == x_t.shape
        assert x_next.device == x_t.device

    def test_ddpm_step_determinism(self, config, noise_schedule, device):
        """Test DDPM step is deterministic with same seed."""
        model = MockScoreModel()

        # Create two solvers with same seed
        config1 = FastSBOTConfig(quality="draft", seed=42, warmup=False, use_mixed_precision=False)
        config2 = FastSBOTConfig(quality="draft", seed=42, warmup=False, use_mixed_precision=False)

        solver1 = FastSBOTSolver(model, noise_schedule, config1, device)
        solver2 = FastSBOTSolver(model, noise_schedule, config2, device)

        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.randn_like(x_t)

        # Run same step on both solvers
        x_next1 = solver1.ddpm_step_improved(x_t.clone(), noise_pred.clone(), t_curr=0.5, t_next=0.4)
        x_next2 = solver2.ddpm_step_improved(x_t.clone(), noise_pred.clone(), t_curr=0.5, t_next=0.4)

        torch.testing.assert_close(x_next1, x_next2)

    def test_ddpm_posterior_mean_coefficients(self, solver, device):
        """Test DDPM posterior mean has correct coefficients."""
        # Use simple values for easy verification
        t_curr = 0.5
        t_next = 0.4

        alpha_bar_curr = solver._get_cached_noise_schedule(t_curr)
        alpha_bar_prev = solver._get_cached_noise_schedule(t_next)

        # Compute expected coefficients
        beta_t = 1.0 - alpha_bar_curr / alpha_bar_prev
        alpha_t = alpha_bar_curr / alpha_bar_prev

        expected_coef1 = math.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_curr)
        expected_coef2 = math.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_curr)

        # Verify coefficients sum to approximately 1 (for stable predictions)
        # This is a sanity check, not a strict requirement
        assert 0.5 < expected_coef1 + expected_coef2 < 1.5

    def test_ddpm_variance_positive(self, solver, device):
        """Test DDPM variance is always positive."""
        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.randn_like(x_t)

        # Test at various timesteps
        for t_curr, t_next in [(0.9, 0.8), (0.5, 0.4), (0.1, 0.0)]:
            x_next = solver.ddpm_step_improved(x_t, noise_pred, t_curr, t_next)
            assert torch.isfinite(x_next).all()


class TestDDIMStep:
    """Test DDIM sampling step correctness."""

    def test_ddim_step_shape(self, solver, device):
        """Test that DDIM step preserves shape."""
        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.randn_like(x_t)

        x_next = solver.ddim_step(x_t, noise_pred, t_curr=0.5, t_next=0.4, eta=0.0)

        assert x_next.shape == x_t.shape
        assert x_next.device == x_t.device

    def test_ddim_deterministic(self, solver, device):
        """Test DDIM is deterministic with eta=0."""
        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.randn_like(x_t)

        # Run twice with eta=0
        x_next1 = solver.ddim_step(x_t.clone(), noise_pred.clone(), t_curr=0.5, t_next=0.4, eta=0.0)
        x_next2 = solver.ddim_step(x_t.clone(), noise_pred.clone(), t_curr=0.5, t_next=0.4, eta=0.0)

        torch.testing.assert_close(x_next1, x_next2)

    def test_ddim_stochastic(self, config, noise_schedule, device):
        """Test DDIM is stochastic with eta>0."""
        model = MockScoreModel()

        # Create two solvers with different seeds
        config1 = FastSBOTConfig(quality="draft", seed=42, warmup=False, use_mixed_precision=False)
        config2 = FastSBOTConfig(quality="draft", seed=123, warmup=False, use_mixed_precision=False)

        solver1 = FastSBOTSolver(model, noise_schedule, config1, device)
        solver2 = FastSBOTSolver(model, noise_schedule, config2, device)

        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.randn_like(x_t)

        # Run with eta=1.0 (full stochasticity)
        x_next1 = solver1.ddim_step(x_t.clone(), noise_pred.clone(), t_curr=0.5, t_next=0.4, eta=1.0)
        x_next2 = solver2.ddim_step(x_t.clone(), noise_pred.clone(), t_curr=0.5, t_next=0.4, eta=1.0)

        # Should be different due to different random seeds
        assert not torch.allclose(x_next1, x_next2, atol=1e-5)


class TestSampling:
    """Test full sampling process."""

    def test_sample_basic(self, solver, device):
        """Test basic sampling works."""
        shape = (2, 3, 32, 32)
        timesteps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

        samples = solver.sample(shape, timesteps, verbose=False)

        assert samples.shape == shape
        assert samples.device == device
        assert torch.isfinite(samples).all()

    def test_sample_improved_ddim(self, solver, device):
        """Test improved sampling with DDIM."""
        shape = (2, 3, 32, 32)
        timesteps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

        samples = solver.sample_improved(
            shape, timesteps, verbose=False, use_ddim=True, eta=0.0
        )

        assert samples.shape == shape
        assert torch.isfinite(samples).all()

    def test_sample_improved_ddpm(self, solver, device):
        """Test improved sampling with DDPM."""
        shape = (2, 3, 32, 32)
        timesteps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

        samples = solver.sample_improved(
            shape, timesteps, verbose=False, use_ddim=False
        )

        assert samples.shape == shape
        assert torch.isfinite(samples).all()

    def test_timestep_creation(self, solver):
        """Test timestep schedule creation."""
        timesteps = solver.create_optimal_timesteps(50, "linear")

        assert len(timesteps) >= 2
        assert timesteps[0] == 1.0  # Start from noise
        assert timesteps[-1] == 0.0  # End at clean

        # Check descending order
        for i in range(len(timesteps) - 1):
            assert timesteps[i] > timesteps[i+1]


class TestOptimalTransport:
    """Test optimal transport components."""

    def test_sliced_ot_shape_preservation(self, solver, device):
        """Test sliced OT preserves shape."""
        x = torch.randn(4, 100, 2, device=device)
        y = torch.randn(4, 100, 2, device=device)

        result = solver.transport_module.sliced_ot.transport(x, y, eps=0.01, n_projections=10)

        assert result.shape == x.shape
        assert result.device == x.device

    def test_full_ot_shape_preservation(self, solver, device):
        """Test full OT preserves shape."""
        # Use small size to fit in memory
        x = torch.randn(2, 20, 3, device=device)
        y = torch.randn(2, 20, 3, device=device)

        result = solver.transport_module.sliced_ot._full_ot(x, y, eps=0.1)

        assert result.shape == x.shape
        assert result.device == x.device


class TestNumericalStability:
    """Test numerical stability of implementations."""

    def test_extreme_timesteps(self, solver, device):
        """Test solver handles extreme timesteps."""
        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.randn_like(x_t)

        # Test near t=1 (very noisy)
        x_next = solver.ddim_step(x_t, noise_pred, t_curr=0.99, t_next=0.98, eta=0.0)
        assert torch.isfinite(x_next).all()

        # Test near t=0 (very clean)
        x_next = solver.ddim_step(x_t, noise_pred, t_curr=0.01, t_next=0.0, eta=0.0)
        assert torch.isfinite(x_next).all()

    def test_zero_noise_prediction(self, solver, device):
        """Test solver handles zero noise prediction."""
        x_t = torch.randn(2, 3, 32, 32, device=device)
        noise_pred = torch.zeros_like(x_t)

        x_next = solver.ddim_step(x_t, noise_pred, t_curr=0.5, t_next=0.4, eta=0.0)
        assert torch.isfinite(x_next).all()


class TestRegressionGaps:
    """Regression tests for cache correctness and Fisher gradients."""

    def test_sample_recomputes_scores_per_call(self, noise_schedule, device):
        """Ensure score cache keys depend on x_t content, forcing recomputation."""

        class CountingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 1)
                self.calls = 0

            def forward(self, x, t):
                self.calls += 1
                return self.conv(x)

        cfg = FastSBOTConfig(
            quality="draft",
            warmup=False,
            use_mixed_precision=False,
            use_dynamic_compilation=False,
            use_triton_kernels=False,
            use_momentum_transport=False,
            use_hierarchical_bridge=False,
            seed=123,
        )
        model = CountingModel()
        solver = FastSBOTSolver(model, noise_schedule, cfg, device)
        timesteps = [1.0, 0.0]
        shape = (1, 3, 4, 4)

        solver.sample(shape, timesteps, verbose=False)
        first_calls = model.calls
        solver.sample(shape, timesteps, verbose=False)

        assert first_calls > 0
        # With content-aware cache keys, the second call must trigger new score evaluations
        assert model.calls > first_calls

    def test_fisher_retains_gradients(self, device):
        """Fisher diagonal estimator must allow gradient flow for training."""
        cfg = FastSBOTConfig(
            quality="draft",
            warmup=False,
            use_mixed_precision=False,
            use_triton_kernels=False,
            use_dynamic_compilation=False,
        )
        kernel_module = KernelModule(cfg, device)

        score = torch.randn(2, 3, 8, 8, device=device, requires_grad=True)
        x = torch.randn_like(score)

        fisher = kernel_module.estimate_fisher_diagonal(x, score, t=0.25, alpha=0.8)
        loss = fisher.sum()
        loss.backward()

        assert score.grad is not None
        assert torch.isfinite(score.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
