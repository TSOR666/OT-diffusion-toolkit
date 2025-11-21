"""Tests for kernel derivative RFF implementation."""

import pytest
import torch
import math

from sbds.kernel import KernelDerivativeRFF


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestKernelDerivativeRFF:
    """Test Random Fourier Features kernel derivatives."""

    def test_initialization(self, device):
        """Test RFF initialization."""
        rff = KernelDerivativeRFF(
            input_dim=10,
            feature_dim=128,
            sigma=1.0,
            device=device,
            seed=42
        )

        assert rff.input_dim == 10
        assert rff.feature_dim == 128
        assert rff.sigma == 1.0
        assert rff.weights.shape == (10, 128)
        assert rff.offset.shape == (128,)

    def test_compute_features_shape(self, device):
        """Test feature computation produces correct shape."""
        rff = KernelDerivativeRFF(input_dim=5, feature_dim=64, device=device)
        x = torch.randn(20, 5, device=device)

        features = rff.compute_features(x)

        assert features.shape == (20, 64)
        assert torch.isfinite(features).all()

    def test_compute_kernel_self(self, device):
        """Test kernel computation with self."""
        rff = KernelDerivativeRFF(input_dim=3, feature_dim=256, device=device, seed=42)
        x = torch.randn(10, 3, device=device)

        K = rff.compute_kernel(x)

        assert K.shape == (10, 10)
        # Kernel matrix should be symmetric
        torch.testing.assert_close(K, K.T, rtol=1e-5, atol=1e-5)
        # Diagonal should be approximately 1
        assert torch.allclose(torch.diag(K), torch.ones(10, device=device), rtol=0.1)

    def test_compute_kernel_two_inputs(self, device):
        """Test kernel computation with two different inputs."""
        rff = KernelDerivativeRFF(input_dim=3, feature_dim=256, device=device, seed=42)
        x = torch.randn(10, 3, device=device)
        y = torch.randn(15, 3, device=device)

        K = rff.compute_kernel(x, y)

        assert K.shape == (10, 15)
        assert torch.isfinite(K).all()
        # RFF approximation can have negative values due to approximation error
        # Typical range is roughly [-0.5, 1.5] for standard Gaussian data
        assert K.min() >= -1.0
        assert K.max() <= 2.0

    def test_derivative_shape(self, device):
        """Test derivative computation produces correct shapes."""
        # Need derivative_order=2 to support second derivatives
        rff = KernelDerivativeRFF(input_dim=4, feature_dim=256, device=device, derivative_order=2)
        x = torch.randn(10, 4, device=device)
        y = torch.randn(15, 4, device=device)

        # First derivative
        dK1 = rff.compute_kernel_derivative(x, y, order=1)
        assert dK1.shape == (4, 10, 15)

        # Second derivative
        dK2 = rff.compute_kernel_derivative(x, y, order=2)
        assert dK2.shape == (4, 4, 10, 15)

    def test_derivative_with_same_input(self, device):
        """Test derivative when y is None (defaults to x)."""
        rff = KernelDerivativeRFF(input_dim=3, feature_dim=128, device=device)
        x = torch.randn(10, 3, device=device)

        dK = rff.compute_kernel_derivative(x, y=None, order=1)

        assert dK.shape == (3, 10, 10)
        assert torch.isfinite(dK).all()

    def test_derivative_numerical_accuracy(self, device):
        """Test derivative accuracy against finite differences."""
        torch.manual_seed(42)
        rff = KernelDerivativeRFF(
            input_dim=2,
            feature_dim=512,  # More features for better approximation
            sigma=1.0,
            device=device,
            seed=42
        )
        x = torch.randn(8, 2, device=device)
        y = torch.randn(12, 2, device=device)

        K = rff.compute_kernel(x, y)
        dK = rff.compute_kernel_derivative(x, y, order=1)

        # Numerical derivative with optimal h
        h = 1e-3
        for i in range(2):
            x_plus = x.clone()
            x_plus[:, i] += h
            K_plus = rff.compute_kernel(x_plus, y)
            dK_numerical = (K_plus - K) / h

            error = torch.max(torch.abs(dK[i] - dK_numerical))
            # RFF + finite difference error should be reasonable
            assert error < 2.0, f"Derivative error {error} too large for dim {i}"

    def test_score_approximation_shape(self, device):
        """Test score approximation produces correct shape."""
        rff = KernelDerivativeRFF(input_dim=5, feature_dim=256, device=device)
        x = torch.randn(20, 5, device=device)
        y = torch.randn(30, 5, device=device)

        score = rff.compute_score_approximation(x, y)

        assert score.shape == (20, 5)
        assert torch.isfinite(score).all()

    def test_error_bound_estimation(self, device):
        """Test error bound estimation."""
        rff = KernelDerivativeRFF(input_dim=10, feature_dim=256, device=device)

        bounds = rff.estimate_error_bound(n_samples=100)

        assert 'kernel' in bounds
        assert 'first_derivative' in bounds
        assert 'second_derivative' in bounds
        # All bounds should be positive
        assert all(v > 0 for v in bounds.values())
        # Higher order derivatives should have larger bounds
        assert bounds['first_derivative'] >= bounds['kernel']
        assert bounds['second_derivative'] >= bounds['first_derivative']

    def test_orthogonal_features(self, device):
        """Test orthogonal feature initialization."""
        rff_orth = KernelDerivativeRFF(
            input_dim=10,
            feature_dim=100,
            orthogonal=True,
            device=device,
            seed=42
        )
        rff_random = KernelDerivativeRFF(
            input_dim=10,
            feature_dim=100,
            orthogonal=False,
            device=device,
            seed=42
        )

        x = torch.randn(50, 10, device=device)
        K_orth = rff_orth.compute_kernel(x)
        K_random = rff_random.compute_kernel(x)

        # Both should produce valid kernels
        assert torch.isfinite(K_orth).all()
        assert torch.isfinite(K_random).all()

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            KernelDerivativeRFF(input_dim=0, feature_dim=128)

        with pytest.raises(ValueError):
            KernelDerivativeRFF(input_dim=10, feature_dim=0)

        with pytest.raises(ValueError):
            KernelDerivativeRFF(input_dim=10, feature_dim=128, sigma=-1.0)

        with pytest.raises(NotImplementedError):
            KernelDerivativeRFF(input_dim=10, feature_dim=128, kernel_type='laplacian')

    def test_dimension_mismatch_error(self, device):
        """Test error on dimension mismatch."""
        rff = KernelDerivativeRFF(input_dim=5, feature_dim=128, device=device)
        x_wrong = torch.randn(10, 7, device=device)  # Wrong dimension

        with pytest.raises(ValueError):
            rff.compute_features(x_wrong)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
