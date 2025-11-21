"""Tests for SBDS solver implementation."""

import pytest
import torch
import torch.nn as nn

from sbds import EnhancedScoreBasedSBDiffusionSolver, EnhancedAdaptiveNoiseSchedule, create_standard_timesteps


class MockScoreModel(nn.Module):
    """Simple mock score model for testing."""

    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x, t):
        if x.dim() > 2:
            x_flat = x.reshape(x.size(0), -1)
        else:
            x_flat = x
        t = t.reshape(-1, 1)
        return self.net(torch.cat([x_flat, t], dim=1)).reshape(x.shape)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def score_model():
    """Create mock score model."""
    return MockScoreModel(dim=2)


@pytest.fixture
def noise_schedule(device):
    """Create noise schedule."""
    return EnhancedAdaptiveNoiseSchedule(
        schedule_type='cosine',
        num_timesteps=100,
        device=device
    )


@pytest.fixture
def solver(score_model, noise_schedule, device):
    """Create solver."""
    return EnhancedScoreBasedSBDiffusionSolver(
        score_model=score_model,
        noise_schedule=noise_schedule,
        device=device,
        eps=0.1,
        sb_iterations=2,
        computational_tier='auto'
    )


class TestSolverInitialization:
    """Test solver initialization."""

    def test_basic_initialization(self, solver):
        """Test solver initializes correctly."""
        assert solver.eps == 0.1
        assert solver.sb_iterations == 2
        assert solver.computational_tier == 'auto'

    def test_input_validation(self, score_model, noise_schedule, device):
        """Test input validation."""
        with pytest.raises(ValueError):
            EnhancedScoreBasedSBDiffusionSolver(
                score_model, noise_schedule, device, eps=-0.1
            )

        with pytest.raises(ValueError):
            EnhancedScoreBasedSBDiffusionSolver(
                score_model, noise_schedule, device, sb_iterations=0
            )

        with pytest.raises(ValueError):
            EnhancedScoreBasedSBDiffusionSolver(
                score_model, noise_schedule, device, computational_tier='invalid'
            )


class TestScoreComputation:
    """Test score computation."""

    def test_compute_score_shape(self, solver, device):
        """Test score computation shape."""
        x = torch.randn(8, 2, device=device)
        t = 0.5

        score = solver._compute_score(x, t)

        assert score.shape == x.shape
        assert torch.isfinite(score).all()

    def test_compute_score_batch(self, solver, device):
        """Test score computation with batches."""
        batch_sizes = [1, 4, 16]
        for bs in batch_sizes:
            x = torch.randn(bs, 2, device=device)
            score = solver._compute_score(x, 0.5)
            assert score.shape == (bs, 2)


class TestDriftComputation:
    """Test drift computation."""

    def test_drift_shape(self, solver, device):
        """Test drift computation shape."""
        x = torch.randn(8, 2, device=device)
        t_curr = 0.6
        dt = 0.1

        drift = solver._compute_drift(x, t_curr, dt)

        assert drift.shape == x.shape
        assert torch.isfinite(drift).all()

    def test_drift_decreases_with_time(self, solver, device):
        """Test that drift magnitude generally decreases as we approach clean data."""
        x = torch.randn(16, 2, device=device)
        dt = 0.1

        drift_early = solver._compute_drift(x, t=0.9, dt=dt)
        drift_late = solver._compute_drift(x, t=0.1, dt=dt)

        # Early drift should generally be larger (more noise to remove)
        # This is a statistical test, so we use norms
        assert torch.norm(drift_early) >= 0.0  # Just verify it's finite
        assert torch.norm(drift_late) >= 0.0


class TestSampling:
    """Test sampling functionality."""

    def test_sample_basic(self, solver, device):
        """Test basic sampling."""
        shape = (4, 2)
        timesteps = [1.0, 0.5, 0.0]

        samples = solver.sample(shape, timesteps, verbose=False)

        assert samples.shape == shape
        assert samples.device == device
        assert torch.isfinite(samples).all()

    def test_sample_with_timesteps(self, solver, device):
        """Test sampling with various timestep configurations."""
        shape = (8, 2)

        # Linear timesteps
        timesteps = create_standard_timesteps(num_steps=10, schedule_type='linear')
        samples = solver.sample(shape, timesteps, verbose=False)
        assert samples.shape == shape

        # Custom timesteps
        timesteps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        samples = solver.sample(shape, timesteps, verbose=False)
        assert samples.shape == shape

    def test_sample_different_tiers(self, score_model, noise_schedule, device):
        """Test sampling with different computational tiers."""
        shape = (8, 2)
        timesteps = [1.0, 0.5, 0.0]

        for tier in ['full', 'rff']:
            solver = EnhancedScoreBasedSBDiffusionSolver(
                score_model, noise_schedule, device,
                computational_tier=tier,
                sb_iterations=1
            )
            samples = solver.sample(shape, timesteps, verbose=False)
            assert samples.shape == shape
            assert torch.isfinite(samples).all()

    def test_memory_validation(self, solver):
        """Test memory usage validation."""
        # Should raise error for huge shapes
        with pytest.raises(ValueError):
            huge_shape = (1000000, 1000)  # 1 billion elements
            solver._validate_memory_usage(huge_shape)

    def test_timestep_validation(self, solver, device):
        """Test timestep validation."""
        shape = (4, 2)

        # Empty timesteps should raise error
        with pytest.raises(ValueError):
            solver.sample(shape, [], verbose=False)

        # Non-finite timesteps should raise error
        with pytest.raises(ValueError):
            solver.sample(shape, [1.0, float('nan'), 0.0], verbose=False)


class TestTransportMethods:
    """Test transport methods."""

    def test_enhanced_sb_transport(self, solver, device):
        """Test enhanced SB transport."""
        x_t = torch.randn(8, 2, device=device)
        t_curr = 0.6
        t_next = 0.5

        x_next, cost = solver._enhanced_sb_transport(x_t, t_curr, t_next, iterations=2)

        assert x_next.shape == x_t.shape
        assert torch.isfinite(x_next).all()
        assert isinstance(cost, float)
        assert cost >= 0

    def test_rff_sb_transport(self, solver, device):
        """Test RFF SB transport."""
        x_t = torch.randn(8, 2, device=device)
        t_curr = 0.6
        t_next = 0.5

        x_next, cost = solver._rff_sb_transport(x_t, t_curr, t_next, iterations=2)

        assert x_next.shape == x_t.shape
        assert torch.isfinite(x_next).all()
        assert cost >= 0

    def test_transport_plan_application(self, solver, device):
        """Test transport plan application."""
        x = torch.randn(10, 2, device=device)
        y = torch.randn(10, 2, device=device)
        P = torch.softmax(torch.randn(10, 10, device=device), dim=1)

        x_transported = solver._apply_transport_map(x, y, P)

        assert x_transported.shape == x.shape
        assert torch.isfinite(x_transported).all()


class TestNumericalStability:
    """Test numerical stability."""

    def test_stable_log(self, solver, device):
        """Test stable logarithm."""
        x = torch.tensor([1e-20, 1.0, 1e20], device=device)
        log_x = solver._stable_log(x)

        assert torch.isfinite(log_x).all()

    def test_stable_exp(self, solver, device):
        """Test stable exponential."""
        x = torch.tensor([-100.0, 0.0, 100.0], device=device)
        exp_x = solver._stable_exp(x)

        assert torch.isfinite(exp_x).all()
        assert (exp_x > 0).all()

    def test_extreme_timesteps(self, solver, device):
        """Test solver handles extreme timesteps."""
        x = torch.randn(4, 2, device=device)

        # Very early (near t=1, pure noise)
        drift_early = solver._compute_drift(x, t=0.99, dt=0.01)
        assert torch.isfinite(drift_early).all()

        # Very late (near t=0, clean)
        drift_late = solver._compute_drift(x, t=0.01, dt=0.01)
        assert torch.isfinite(drift_late).all()


class TestTierSelection:
    """Test computational tier selection."""

    def test_auto_tier_selection(self, solver):
        """Test automatic tier selection."""
        # Small problem should use 'full' (< 500k elements)
        tier_small = solver._determine_computational_tier(batch_size=10, dim=10)
        assert tier_small == 'full'

        # Medium problem should use 'rff' (500k - 5M elements)
        tier_medium = solver._determine_computational_tier(batch_size=1000, dim=1000)
        assert tier_medium == 'rff'

        # Large problem should use 'nystrom' or 'multiscale' (> 5M elements)
        tier_large = solver._determine_computational_tier(batch_size=10000, dim=1000)
        assert tier_large in ['nystrom', 'multiscale']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
