"""Tests for DPM-Solver++ implementation."""

import pytest
import torch

from SPOT.dpm_solver import DPMSolverPP
from SPOT.schedules import CosineSchedule, LinearSchedule


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def schedule(device):
    """Create noise schedule."""
    return CosineSchedule(device=device)


class TestDPMSolverInitialization:
    """Test DPM-Solver initialization."""

    def test_initialization_orders(self):
        """Test initialization with different orders."""
        for order in [1, 2, 3]:
            solver = DPMSolverPP(order=order)
            assert solver.order == order

    def test_invalid_order(self):
        """Test that invalid order raises error."""
        with pytest.raises(ValueError):
            DPMSolverPP(order=0)

        with pytest.raises(ValueError):
            DPMSolverPP(order=4)

    def test_with_schedule(self, schedule):
        """Test initialization with schedule."""
        solver = DPMSolverPP(order=2, schedule=schedule)
        assert solver.schedule is schedule


class TestTimesteps:
    """Test timestep generation."""

    def test_timestep_count(self, device):
        """Test correct number of timesteps."""
        solver = DPMSolverPP(order=2)
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        # Returns num_steps + 1 timesteps (including endpoints)
        assert len(timesteps) == 51

    def test_timestep_endpoints(self, device):
        """Test timestep endpoints are correct."""
        solver = DPMSolverPP(order=2)
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        assert timesteps[0] == 1.0  # Start at noise
        assert timesteps[-1] == 0.0  # End at clean

    def test_timestep_monotonic(self, device):
        """Test timesteps are monotonically decreasing."""
        solver = DPMSolverPP(order=2)
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        for i in range(len(timesteps) - 1):
            assert timesteps[i] > timesteps[i+1]

    def test_different_orders_different_schedules(self, device):
        """Test different orders produce different timestep distributions."""
        solver1 = DPMSolverPP(order=1)
        solver3 = DPMSolverPP(order=3)

        timesteps1 = solver1.get_timesteps(num_steps=50, device=device)
        timesteps3 = solver3.get_timesteps(num_steps=50, device=device)

        # Same length
        assert len(timesteps1) == len(timesteps3)

        # Same endpoints
        assert timesteps1[0] == timesteps3[0]
        assert timesteps1[-1] == timesteps3[-1]

        # Different intermediate values
        assert timesteps1[25] != timesteps3[25]

    def test_invalid_num_steps(self, device):
        """Test invalid num_steps raises error."""
        solver = DPMSolverPP(order=2)

        with pytest.raises(ValueError):
            solver.get_timesteps(num_steps=0, device=device)


class TestFirstOrderUpdate:
    """Test first-order update."""

    def test_first_order_shape(self, schedule, device):
        """Test first-order update preserves shape."""
        solver = DPMSolverPP(order=1, schedule=schedule)
        x = torch.randn(4, 3, 32, 32, device=device)
        model_output = torch.randn_like(x)
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        x_next = solver._first_order_update(x, model_output, timesteps, idx=25, schedule=schedule)

        assert x_next.shape == x.shape
        assert torch.isfinite(x_next).all()

    def test_first_order_determinism(self, schedule, device):
        """Test first-order update is deterministic."""
        solver = DPMSolverPP(order=1, schedule=schedule)
        x = torch.randn(4, 3, 32, 32, device=device)
        model_output = torch.randn_like(x)
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        x_next1 = solver._first_order_update(x.clone(), model_output.clone(), timesteps, idx=25, schedule=schedule)
        x_next2 = solver._first_order_update(x.clone(), model_output.clone(), timesteps, idx=25, schedule=schedule)

        torch.testing.assert_close(x_next1, x_next2)


class TestSecondOrderUpdate:
    """Test second-order update."""

    def test_second_order_shape(self, schedule, device):
        """Test second-order update preserves shape."""
        solver = DPMSolverPP(order=2, schedule=schedule)
        x = torch.randn(4, 3, 32, 32, device=device)
        model_outputs = [torch.randn_like(x), torch.randn_like(x)]
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        x_next = solver._second_order_update(x, model_outputs, timesteps, idx=25, schedule=schedule)

        assert x_next.shape == x.shape
        assert torch.isfinite(x_next).all()

    def test_second_order_fallback(self, schedule, device):
        """Test second-order falls back to first-order on numerical issues."""
        solver = DPMSolverPP(order=2, schedule=schedule)
        x = torch.randn(4, 3, 32, 32, device=device)
        model_outputs = [torch.randn_like(x), torch.randn_like(x)]

        # Create timesteps with very small gap (should trigger fallback)
        timesteps = [1.0, 0.999999, 0.999998, 0.5, 0.0]

        x_next = solver._second_order_update(x, model_outputs, timesteps, idx=1, schedule=schedule)

        # Should still produce valid output (fallback to first-order)
        assert x_next.shape == x.shape


class TestThirdOrderUpdate:
    """Test third-order update."""

    def test_third_order_shape(self, schedule, device):
        """Test third-order update preserves shape."""
        solver = DPMSolverPP(order=3, schedule=schedule)
        x = torch.randn(4, 3, 32, 32, device=device)
        model_outputs = [torch.randn_like(x), torch.randn_like(x), torch.randn_like(x)]
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        x_next = solver._third_order_update(x, model_outputs, timesteps, idx=25, schedule=schedule)

        assert x_next.shape == x.shape
        assert torch.isfinite(x_next).all()


class TestMultistepUpdate:
    """Test multistep update router."""

    def test_multistep_with_deque(self, schedule, device):
        """Test multistep works with deque."""
        from collections import deque

        solver = DPMSolverPP(order=3, schedule=schedule)
        x = torch.randn(4, 3, 32, 32, device=device)
        model_outputs = deque([torch.randn_like(x), torch.randn_like(x), torch.randn_like(x)])
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        x_next = solver.multistep_update(x, model_outputs, timesteps, current_idx=25)

        assert x_next.shape == x.shape
        assert torch.isfinite(x_next).all()

    def test_multistep_order_selection(self, schedule, device):
        """Test multistep correctly selects order based on history."""
        solver = DPMSolverPP(order=3, schedule=schedule)
        x = torch.randn(4, 3, 32, 32, device=device)
        timesteps = solver.get_timesteps(num_steps=50, device=device)

        # With 1 output, should use first-order
        x1 = solver.multistep_update(x, [torch.randn_like(x)], timesteps, current_idx=0)
        assert torch.isfinite(x1).all()

        # With 2 outputs, should use second-order
        x2 = solver.multistep_update(x, [torch.randn_like(x), torch.randn_like(x)], timesteps, current_idx=1)
        assert torch.isfinite(x2).all()

        # With 3 outputs, should use third-order
        x3 = solver.multistep_update(
            x, [torch.randn_like(x), torch.randn_like(x), torch.randn_like(x)],
            timesteps, current_idx=2
        )
        assert torch.isfinite(x3).all()


class TestNumericalStability:
    """Test numerical stability."""

    def test_extreme_timesteps(self, schedule, device):
        """Test solver handles extreme timesteps."""
        solver = DPMSolverPP(order=2, schedule=schedule)
        x = torch.randn(2, 3, 16, 16, device=device)
        model_output = torch.randn_like(x)

        # Very early (near t=1)
        timesteps_early = [0.999, 0.998, 0.5, 0.0]
        x_next = solver._first_order_update(x, model_output, timesteps_early, idx=0, schedule=schedule)
        assert torch.isfinite(x_next).all()

        # Very late (near t=0)
        timesteps_late = [1.0, 0.5, 0.001, 0.0]
        x_next = solver._first_order_update(x, model_output, timesteps_late, idx=2, schedule=schedule)
        assert torch.isfinite(x_next).all()

    def test_different_schedules(self, device):
        """Test solver works with different schedules."""
        schedules = [
            CosineSchedule(device=device),
            LinearSchedule(device=device)
        ]

        for sched in schedules:
            solver = DPMSolverPP(order=2, schedule=sched)
            x = torch.randn(2, 3, 16, 16, device=device)
            model_output = torch.randn_like(x)
            timesteps = solver.get_timesteps(num_steps=20, device=device)

            x_next = solver._first_order_update(x, model_output, timesteps, idx=10, schedule=sched)
            assert torch.isfinite(x_next).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
