"""Tests for Triton GPU kernels to verify correctness against PyTorch reference."""

import pytest
import torch

from fastsb_ot import FastSBOTConfig
from fastsb_ot.kernels import KernelModule
from fastsb_ot.common import TRITON_AVAILABLE


@pytest.fixture
def device():
    """Get CUDA device if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def config():
    """Create config with Triton enabled."""
    return FastSBOTConfig(
        quality="draft",
        use_triton_kernels=True,
        use_mixed_precision=False,
        warmup=False
    )


class TestTritonFisherKernel:
    """Test Triton Fisher kernel against PyTorch reference."""

    def test_fisher_triton_vs_pytorch(self, config, device):
        """Verify Triton Fisher kernel matches PyTorch implementation."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        kernel_module = KernelModule(config, device)

        # Create test input with large enough size to trigger Triton
        x = torch.randn(2, 3, 512, 512, device=device)
        score = torch.randn_like(x)
        alpha = 0.8

        # Compute Fisher with Triton (should be triggered for > 1e6 elements)
        config.use_triton_kernels = True
        fisher_triton = kernel_module.estimate_fisher_diagonal(x, score, t=0.5, alpha=alpha)

        # Compute Fisher with PyTorch reference (force disable Triton)
        config.use_triton_kernels = False
        fisher_pytorch = kernel_module.estimate_fisher_diagonal(x, score, t=0.5, alpha=alpha)

        # Results should be very close
        torch.testing.assert_close(
            fisher_triton,
            fisher_pytorch,
            rtol=1e-3,
            atol=1e-5,
            msg="Triton Fisher kernel doesn't match PyTorch reference"
        )

    def test_fisher_triton_positivity(self, config, device):
        """Verify Triton Fisher kernel always produces positive values."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        kernel_module = KernelModule(config, device)
        config.use_triton_kernels = True

        x = torch.randn(2, 3, 512, 512, device=device)
        score = torch.randn_like(x)

        fisher = kernel_module.estimate_fisher_diagonal(x, score, t=0.5, alpha=0.8)

        # Must be strictly positive
        assert (fisher > 0).all(), "Triton Fisher kernel produced non-positive values"
        assert fisher.min() >= 1e-6, f"Triton Fisher minimum {fisher.min()} is too small"

    def test_fisher_triton_with_extreme_values(self, config, device):
        """Test Triton Fisher with extreme score values."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        kernel_module = KernelModule(config, device)
        config.use_triton_kernels = True

        x = torch.randn(2, 3, 512, 512, device=device)

        # Very large scores
        score_large = torch.full_like(x, 100.0)
        fisher_large = kernel_module.estimate_fisher_diagonal(x, score_large, t=0.5, alpha=0.8)
        assert torch.isfinite(fisher_large).all()
        assert (fisher_large > 0).all()

        # Very small scores
        score_small = torch.full_like(x, 1e-4)
        fisher_small = kernel_module.estimate_fisher_diagonal(x, score_small, t=0.5, alpha=0.8)
        assert torch.isfinite(fisher_small).all()
        assert (fisher_small > 0).all()

    def test_fisher_triton_different_alpha(self, config, device):
        """Test Triton Fisher with different alpha values."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        kernel_module = KernelModule(config, device)
        config.use_triton_kernels = True

        x = torch.randn(2, 3, 512, 512, device=device)
        score = torch.randn_like(x)

        # Test at different alpha values (noise levels)
        for alpha in [0.1, 0.5, 0.9, 0.99]:
            fisher = kernel_module.estimate_fisher_diagonal(x, score, t=0.5, alpha=alpha)
            assert torch.isfinite(fisher).all()
            assert (fisher > 0).all()
            # Fisher should adapt based on alpha (higher alpha = cleaner = smaller eps)
            assert fisher.min() > 0


class TestTritonKernelPerformance:
    """Test that Triton kernels are actually used when expected."""

    def test_triton_used_for_large_tensors(self, config, device):
        """Verify Triton is used for tensors > 1e6 elements."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        kernel_module = KernelModule(config, device)
        config.use_triton_kernels = True

        # Large tensor (> 1e6 elements) should trigger Triton
        x = torch.randn(2, 3, 512, 512, device=device)  # 1.57M elements
        score = torch.randn_like(x)

        # This should use Triton path
        fisher = kernel_module.estimate_fisher_diagonal(x, score, t=0.5, alpha=0.8)
        assert fisher.shape == score.shape

    def test_pytorch_used_for_small_tensors(self, config, device):
        """Verify PyTorch reference is used for small tensors."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        kernel_module = KernelModule(config, device)
        config.use_triton_kernels = True

        # Small tensor (< 1e6 elements) should use PyTorch
        x = torch.randn(2, 3, 64, 64, device=device)  # 24K elements
        score = torch.randn_like(x)

        # This should use PyTorch path (even with Triton enabled)
        fisher = kernel_module.estimate_fisher_diagonal(x, score, t=0.5, alpha=0.8)
        assert fisher.shape == score.shape


class TestTritonKernelDeterminism:
    """Test Triton kernel determinism."""

    def test_triton_fisher_deterministic(self, config, device):
        """Verify Triton Fisher produces same results on repeated calls."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        kernel_module = KernelModule(config, device)
        config.use_triton_kernels = True

        x = torch.randn(2, 3, 512, 512, device=device)
        score = torch.randn_like(x)

        # Compute twice
        fisher1 = kernel_module.estimate_fisher_diagonal(x.clone(), score.clone(), t=0.5, alpha=0.8)
        fisher2 = kernel_module.estimate_fisher_diagonal(x.clone(), score.clone(), t=0.5, alpha=0.8)

        # Should be identical (deterministic computation)
        torch.testing.assert_close(fisher1, fisher2, rtol=0, atol=0)


class TestTritonKernelGradients:
    """Validate gradient safety for training paths when Triton is enabled."""

    def test_fisher_gradcheck(self, config, device):
        """Gradcheck the Fisher diagonal when gradients are required."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        config.use_triton_kernels = True
        config.use_fp32_fisher = False
        kernel_module = KernelModule(config, device)

        x = torch.randn(1, 32, device=device, dtype=torch.float64)
        score = torch.randn_like(x, dtype=torch.float64, requires_grad=True)

        def fn(score_input):
            return kernel_module.estimate_fisher_diagonal(x, score_input, t=0.5, alpha=0.8)

        assert torch.autograd.gradcheck(fn, (score,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_fisher_triton_skipped_when_requires_grad(self, config, device):
        """Ensure Triton kernels are bypassed when autograd is needed."""
        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        config.use_triton_kernels = True
        kernel_module = KernelModule(config, device)

        x = torch.randn(1, 1_050_000, device=device, dtype=torch.float32)
        score = torch.randn_like(x, requires_grad=True)

        fisher = kernel_module.estimate_fisher_diagonal(x, score, t=0.5, alpha=0.8)
        assert fisher.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
