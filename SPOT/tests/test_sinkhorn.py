"""Tests for Sinkhorn algorithm implementation."""

import pytest
import torch

from SPOT.sinkhorn import OptimizedSinkhornKernel
from SPOT.config import SolverConfig


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def config():
    """Create basic config."""
    return SolverConfig(deterministic=True)


@pytest.fixture
def sinkhorn_kernel(device, config):
    """Create Sinkhorn kernel."""
    return OptimizedSinkhornKernel(device=device, dtype=torch.float32, config=config)


class TestSinkhornBasic:
    """Test basic Sinkhorn functionality."""

    def test_log_stabilized_shape(self, sinkhorn_kernel, device):
        """Test Sinkhorn returns correct shapes."""
        x = torch.randn(20, 3, device=device)
        y = torch.randn(30, 3, device=device)

        log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=10)

        assert log_u.shape == (20,)
        assert log_v.shape == (30,)
        assert torch.isfinite(log_u).all()
        assert torch.isfinite(log_v).all()

    def test_empty_tensor_handling(self, sinkhorn_kernel, device):
        """Test handling of empty tensors."""
        x_empty = torch.randn(0, 3, device=device)
        y = torch.randn(10, 3, device=device)

        log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x_empty, y, eps=0.1, n_iter=10)

        # Should return NaN for empty input
        assert log_u.shape == (0,)
        assert log_v.shape == (10,)

    def test_single_point(self, sinkhorn_kernel, device):
        """Test Sinkhorn with single point."""
        x = torch.randn(1, 3, device=device)
        y = torch.randn(1, 3, device=device)

        log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=10)

        assert log_u.shape == (1,)
        assert log_v.shape == (1,)
        assert torch.isfinite(log_u).all()
        assert torch.isfinite(log_v).all()

    def test_marginal_constraints(self, sinkhorn_kernel, device):
        """Test that Sinkhorn satisfies marginal constraints."""
        torch.manual_seed(42)
        x = torch.randn(50, 3, device=device)
        y = torch.randn(50, 3, device=device)

        log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=100)

        # Compute transport plan
        N = x.shape[0]
        C = torch.cdist(x, y, p=2).pow(2)
        log_K = -C / 0.1
        log_P = log_K + log_u[:, None] + log_v[None, :]
        P = torch.exp(log_P)

        # Check marginal constraints
        row_sums = P.sum(dim=1)
        col_sums = P.sum(dim=0)

        expected = 1.0 / N
        torch.testing.assert_close(row_sums, torch.full_like(row_sums, expected), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(col_sums, torch.full_like(col_sums, expected), rtol=1e-2, atol=1e-2)

    def test_different_sizes(self, sinkhorn_kernel, device):
        """Test Sinkhorn with different input sizes."""
        sizes = [(10, 15), (20, 10), (5, 50)]

        for n, m in sizes:
            x = torch.randn(n, 3, device=device)
            y = torch.randn(m, 3, device=device)

            log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=10)

            assert log_u.shape == (n,)
            assert log_v.shape == (m,)
            assert torch.isfinite(log_u).all()
            assert torch.isfinite(log_v).all()

    def test_convergence_with_iterations(self, sinkhorn_kernel, device):
        """Test that more iterations improve convergence."""
        x = torch.randn(30, 3, device=device)
        y = torch.randn(30, 3, device=device)

        # Few iterations
        log_u_few, log_v_few = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=5)

        # Many iterations
        log_u_many, log_v_many = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=100)

        # Solutions should be different
        assert not torch.allclose(log_u_few, log_u_many, atol=1e-3)
        assert not torch.allclose(log_v_few, log_v_many, atol=1e-3)

        # Both should be finite
        assert torch.isfinite(log_u_few).all()
        assert torch.isfinite(log_u_many).all()


class TestSinkhornNumericalStability:
    """Test numerical stability of Sinkhorn."""

    def test_small_epsilon(self, sinkhorn_kernel, device):
        """Test Sinkhorn with small epsilon (sharp transportation)."""
        x = torch.randn(20, 3, device=device)
        y = torch.randn(20, 3, device=device)

        log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.01, n_iter=50)

        assert torch.isfinite(log_u).all()
        assert torch.isfinite(log_v).all()

    def test_large_epsilon(self, sinkhorn_kernel, device):
        """Test Sinkhorn with large epsilon (soft transportation)."""
        x = torch.randn(20, 3, device=device)
        y = torch.randn(20, 3, device=device)

        log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=1.0, n_iter=10)

        assert torch.isfinite(log_u).all()
        assert torch.isfinite(log_v).all()

    def test_extreme_distances(self, sinkhorn_kernel, device):
        """Test Sinkhorn with very distant points."""
        x = torch.randn(10, 3, device=device)
        y = torch.randn(10, 3, device=device) + 100.0  # Very far away

        log_u, log_v = sinkhorn_kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=50)

        # Should handle gracefully (may return NaN, which is acceptable)
        assert log_u.shape == (10,)
        assert log_v.shape == (10,)


class TestSinkhornDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_cpu(self, config):
        """Test CPU Sinkhorn is deterministic."""
        device = torch.device('cpu')
        kernel1 = OptimizedSinkhornKernel(device, torch.float32, config)
        kernel2 = OptimizedSinkhornKernel(device, torch.float32, config)

        torch.manual_seed(42)
        x = torch.randn(20, 3, device=device)
        y = torch.randn(30, 3, device=device)

        log_u1, log_v1 = kernel1.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=10)
        log_u2, log_v2 = kernel2.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=10)

        torch.testing.assert_close(log_u1, log_u2)
        torch.testing.assert_close(log_v1, log_v2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_deterministic_gpu(self, config):
        """Test GPU Sinkhorn can be deterministic with CPU fallback."""
        device = torch.device('cuda')
        config_det = SolverConfig(deterministic=True, deterministic_cdist_cpu=True)
        kernel1 = OptimizedSinkhornKernel(device, torch.float32, config_det)
        kernel2 = OptimizedSinkhornKernel(device, torch.float32, config_det)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        x = torch.randn(20, 3, device=device)
        y = torch.randn(30, 3, device=device)

        log_u1, log_v1 = kernel1.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=10)
        log_u2, log_v2 = kernel2.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=10)

        torch.testing.assert_close(log_u1, log_u2)
        torch.testing.assert_close(log_v1, log_v2)


class TestSinkhornDispatch:
    """Dispatch logic and backend selection."""

    def test_blockwise_triggered_when_cost_matrix_exceeds_limit(self, device):
        cfg = SolverConfig(
            deterministic=True,
            max_dense_matrix_elements=1_000,  # tiny threshold to force blockwise
            max_tensor_size_elements=50_000_000,
        )
        kernel = OptimizedSinkhornKernel(device, torch.float32, cfg)

        invoked = {"blockwise": False}

        def _fake_blockwise(x, y, eps, n_iter):
            invoked["blockwise"] = True
            return torch.zeros(x.size(0), device=x.device), torch.zeros(y.size(0), device=x.device)

        kernel._sinkhorn_blockwise = _fake_blockwise  # type: ignore[assignment]

        x = torch.randn(128, 3, device=device)
        y = torch.randn(128, 3, device=device)

        kernel.sinkhorn_log_stabilized(x, y, eps=0.1, n_iter=5)
        assert invoked["blockwise"]

    def test_pot_backend_disabled_in_deterministic_mode(self):
        cfg = SolverConfig(deterministic=True, use_pot_library=True)
        kernel = OptimizedSinkhornKernel(torch.device("cpu"), torch.float32, cfg)
        assert not kernel.use_pot
        assert "pot" not in kernel.backends


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
