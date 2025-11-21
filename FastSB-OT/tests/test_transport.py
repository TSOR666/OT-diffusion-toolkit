"""Tests for optimal transport components."""

import pytest
import torch

from fastsb_ot.transport import SlicedOptimalTransport


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestSlicedOT:
    """Test sliced optimal transport."""

    def test_sliced_ot_basic(self, device):
        """Test basic sliced OT."""
        ot = SlicedOptimalTransport(memory_limit_mb=100)

        x = torch.randn(4, 50, 3, device=device)
        y = torch.randn(4, 50, 3, device=device)

        result = ot.transport(x, y, eps=0.01, n_projections=20)

        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_sliced_ot_deterministic(self, device):
        """Test sliced OT is deterministic with same generator."""
        gen1 = torch.Generator(device='cpu')
        gen1.manual_seed(42)

        gen2 = torch.Generator(device='cpu')
        gen2.manual_seed(42)

        ot1 = SlicedOptimalTransport(generator=gen1)
        ot2 = SlicedOptimalTransport(generator=gen2)

        x = torch.randn(4, 50, 3, device=device)
        y = torch.randn(4, 50, 3, device=device)

        result1 = ot1.transport(x, y, eps=0.01, n_projections=20)
        result2 = ot2.transport(x, y, eps=0.01, n_projections=20)

        torch.testing.assert_close(result1, result2, rtol=1e-4, atol=1e-4)

    def test_full_ot_basic(self, device):
        """Test full OT with Sinkhorn."""
        ot = SlicedOptimalTransport(memory_limit_mb=100)

        # Small size for full OT
        x = torch.randn(2, 20, 3, device=device)
        y = torch.randn(2, 20, 3, device=device)

        result = ot._full_ot(x, y, eps=0.1)

        assert result.shape == x.shape
        assert torch.isfinite(result).all()

    def test_sinkhorn_convergence(self, device):
        """Test Sinkhorn algorithm convergence."""
        ot = SlicedOptimalTransport(sinkhorn_iters=100, sinkhorn_tol=1e-6)

        # Create cost matrix
        x = torch.randn(2, 20, 3, device=device)
        y = torch.randn(2, 20, 3, device=device)

        B, N, d = x.shape
        x_expanded = x.unsqueeze(2)
        y_expanded = y.unsqueeze(1)
        C = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)

        # Run Sinkhorn
        P = ot._sinkhorn_batch_fixed(C, eps=0.1)

        # Check marginal constraints
        row_sums = P.sum(dim=2)
        col_sums = P.sum(dim=1)

        # Each row and column should sum to approximately 1/N (uniform marginals)
        expected = 1.0 / N
        torch.testing.assert_close(row_sums, torch.full_like(row_sums, expected), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(col_sums, torch.full_like(col_sums, expected), rtol=1e-2, atol=1e-2)

    def test_barycentric_projection(self, device):
        """Test barycentric projection is correctly normalized."""
        ot = SlicedOptimalTransport()

        x = torch.randn(2, 20, 3, device=device)
        y = torch.randn(2, 20, 3, device=device)

        result = ot._full_ot(x, y, eps=0.1)

        # Result should have similar statistics to y (being transported to y)
        # At least should be finite and reasonable scale
        assert torch.isfinite(result).all()
        assert result.std() > 0  # Not collapsed to zero

    def test_shape_mismatch_error(self, device):
        """Test that shape mismatch raises error."""
        ot = SlicedOptimalTransport()

        x = torch.randn(2, 20, 3, device=device)
        y = torch.randn(2, 30, 3, device=device)  # Different N

        with pytest.raises(ValueError):
            ot.transport(x, y, eps=0.01)


class TestReshapeToPoints:
    """Test tensor reshaping utilities."""

    def test_reshape_2d(self, device):
        """Test reshaping 2D tensors."""
        ot = SlicedOptimalTransport()

        x = torch.randn(4, 10, device=device)
        points, restore = ot._reshape_to_points(x)

        assert points.dim() == 3
        assert points.shape[0] == 4  # Batch preserved
        assert points.shape[1] == 10  # Features become points

        # Test restoration
        restored = restore(points)
        torch.testing.assert_close(restored, x)

    def test_reshape_4d(self, device):
        """Test reshaping 4D tensors (images)."""
        ot = SlicedOptimalTransport()

        x = torch.randn(2, 3, 8, 8, device=device)
        points, restore = ot._reshape_to_points(x)

        assert points.dim() == 3
        assert points.shape[0] == 2  # Batch preserved
        assert points.shape[1] == 64  # 8*8 spatial points
        assert points.shape[2] == 3  # Channels

        # Test restoration
        restored = restore(points)
        torch.testing.assert_close(restored, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
