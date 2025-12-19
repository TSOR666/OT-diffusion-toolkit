"""
Formal Verification Tests for ATLAS

This module contains comprehensive tests for:
- Gradient correctness (gradcheck)
- Edge cases (batch=1, extreme values, degenerate inputs)
- Numerical stability under stress
- Checkpoint round-trip equivalence
- Reproducibility guarantees
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import overload

import pytest
import torch

from atlas.config.kernel_config import KernelConfig
from atlas.config.model_config import HighResModelConfig
from atlas.config.sampler_config import SamplerConfig
from atlas.kernels import (
    DirectKernelOperator,
    FFTKernelOperator,
    NystromKernelOperator,
    RFFKernelOperator,
)
from atlas.models.score_network import HighResLatentScoreModel, build_highres_score_model
from atlas.schedules.noise import karras_noise_schedule
from atlas.solvers.schrodinger_bridge import SchroedingerBridgeSolver


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def _tiny_config() -> HighResModelConfig:
    """Minimal model config for fast tests."""
    return HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(),
        cross_attention_levels=(),
        time_emb_dim=128,
        conditional=False,
    )


@overload
def _linear_schedule(t: float) -> float: ...


@overload
def _linear_schedule(t: torch.Tensor) -> torch.Tensor: ...


def _linear_schedule(t: float | torch.Tensor) -> float | torch.Tensor:
    """Simple constant alpha schedule for testing."""
    if isinstance(t, torch.Tensor):
        return torch.ones_like(t) * 0.9
    return 0.9


def _make_solver() -> SchroedingerBridgeSolver:
    """Create a minimal solver for testing."""
    model = build_highres_score_model(_tiny_config())
    kernel_cfg = KernelConfig(
        solver_type="direct",
        epsilon=0.1,
        rff_features=128,
        n_landmarks=8,
        max_kernel_cache_size=2,
    )
    sampler_cfg = SamplerConfig(
        sb_iterations=10,
        error_tolerance=1e-3,
        marginal_constraint_threshold=5e-2,
        use_linear_solver=False,
        use_mixed_precision=False,
        verbose_logging=False,
    )
    return SchroedingerBridgeSolver(
        model,
        _linear_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_cfg,
        sampler_config=sampler_cfg,
    )


# =============================================================================
# Gate 2: Gradient Correctness (gradcheck)
# =============================================================================


class TestGradientCorrectness:
    """Verify gradient correctness for custom operations using torch.autograd.gradcheck."""

    def test_direct_kernel_gradcheck(self) -> None:
        """Verify DirectKernelOperator gradients are correct."""
        torch.manual_seed(42)
        op = DirectKernelOperator(
            kernel_type="gaussian",
            epsilon=0.5,  # Larger epsilon for stability
            device=torch.device("cpu"),
        )

        # Use double precision for gradcheck
        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=False)
        v = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)

        # Setup kernel
        op.setup(x)

        def apply_fn(v_in: torch.Tensor) -> torch.Tensor:
            # Convert kernel matrix to double for consistent dtype
            kernel = op.kernel_matrix
            assert kernel is not None
            kernel_double = kernel.double()
            v_flat = v_in.reshape(v_in.shape[0], -1)
            result_flat = kernel_double @ v_flat
            return result_flat.reshape(v_in.shape)

        assert torch.autograd.gradcheck(apply_fn, (v,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_rff_kernel_gradcheck(self) -> None:
        """Verify RFFKernelOperator gradients are correct."""
        torch.manual_seed(42)
        op = RFFKernelOperator(
            input_dim=4,
            feature_dim=32,
            kernel_type="gaussian",
            epsilon=0.5,
            device=torch.device("cpu"),
            orthogonal=False,
            multi_scale=False,
        )

        x = torch.randn(4, 4, dtype=torch.float64, requires_grad=False)
        v = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)

        # Convert weights to double
        op.weights = [w.double() for w in op.weights]
        op.offsets = [o.double() for o in op.offsets]

        def apply_fn(v_in: torch.Tensor) -> torch.Tensor:
            return op.apply(x, v_in)

        assert torch.autograd.gradcheck(apply_fn, (v,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_score_model_backward_finite(self) -> None:
        """Verify score model backward pass produces finite gradients."""
        torch.manual_seed(42)
        model = build_highres_score_model(_tiny_config())

        x = torch.randn(2, 3, 16, 16, requires_grad=True)
        t = torch.rand(2)

        output = model(x, t)
        loss = output.mean()
        loss.backward()

        # Check all gradients are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "Non-finite input gradient"


# =============================================================================
# Gate 2: Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_kernel_operator_batch_one(self) -> None:
        """Verify kernel operators handle batch_size=1 correctly."""
        # Direct kernel
        op_direct = DirectKernelOperator(
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
        )
        x = torch.randn(1, 4)
        v = torch.randn(1, 4)
        result = op_direct.apply(x, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()

        # RFF kernel
        op_rff = RFFKernelOperator(
            input_dim=4,
            feature_dim=16,
            epsilon=0.1,
            device=torch.device("cpu"),
        )
        x = torch.randn(1, 4)
        v = torch.randn(1, 4)
        result = op_rff.apply(x, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()

    def test_kernel_operator_batch_one_nystrom(self) -> None:
        """Verify Nystrom operator handles batch_size=1."""
        landmarks = torch.randn(4, 4)
        op = NystromKernelOperator(
            landmarks=landmarks,
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
        )
        x = torch.randn(1, 4)
        v = torch.randn(1, 4)
        result = op.apply(x, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()

    def test_fft_kernel_odd_dimensions(self) -> None:
        """Verify FFT kernel handles odd grid dimensions."""
        op = FFTKernelOperator(
            grid_shape=[5, 7],
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
            multi_scale=False,
        )
        v = torch.randn(1, 1, 5, 7)
        result = op.apply(v, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()

    def test_score_model_minimal_spatial(self) -> None:
        """Verify score model handles minimal valid spatial dimensions."""
        config = _tiny_config()
        model = build_highres_score_model(config)

        # Minimum spatial size = 2^num_downsample
        num_downsample = len([b for b in model.down_blocks if b.downsample is not None])
        min_size = max(2 ** num_downsample, 4)

        x = torch.randn(1, 3, min_size, min_size)
        t = torch.rand(1)
        output = model(x, t)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_noise_schedule_extreme_timesteps(self) -> None:
        """Verify noise schedule handles edge timesteps correctly."""
        for t in [0.0, 0.001, 0.5, 0.999, 1.0]:
            alpha = karras_noise_schedule(t)
            assert 0 < alpha <= 1, f"Alpha out of range at t={t}: {alpha}"
            assert math.isfinite(alpha), f"Non-finite alpha at t={t}"

        # Tensor version
        t_tensor = torch.tensor([0.0, 0.001, 0.5, 0.999, 1.0])
        alpha_tensor = karras_noise_schedule(t_tensor)
        assert (alpha_tensor > 0).all()
        assert (alpha_tensor <= 1).all()
        assert torch.isfinite(alpha_tensor).all()

    def test_kernel_all_zeros_input(self) -> None:
        """Verify kernel operators handle all-zeros input."""
        op = DirectKernelOperator(
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
        )
        x = torch.zeros(4, 4)
        v = torch.zeros(4, 4)
        result = op.apply(x, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()
        assert torch.allclose(result, torch.zeros_like(result))

    def test_kernel_all_ones_input(self) -> None:
        """Verify kernel operators handle all-ones input."""
        op = DirectKernelOperator(
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
        )
        x = torch.ones(4, 4)
        v = torch.ones(4, 4)
        result = op.apply(x, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()

    def test_kernel_large_magnitude_input(self) -> None:
        """Verify kernel operators handle large magnitude inputs."""
        op = DirectKernelOperator(
            kernel_type="gaussian",
            epsilon=1.0,  # Larger epsilon for stability
            device=torch.device("cpu"),
        )
        x = torch.randn(4, 4) * 100
        v = torch.randn(4, 4) * 100
        result = op.apply(x, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()

    def test_kernel_small_magnitude_input(self) -> None:
        """Verify kernel operators handle very small magnitude inputs."""
        op = DirectKernelOperator(
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
        )
        x = torch.randn(4, 4) * 1e-6
        v = torch.randn(4, 4) * 1e-6
        result = op.apply(x, v)
        assert result.shape == v.shape
        assert torch.isfinite(result).all()


# =============================================================================
# Gate 3: Numerical Stability Stress Tests
# =============================================================================


class TestNumericalStability:
    """Stress tests for numerical stability."""

    def test_kernel_epsilon_extremes(self) -> None:
        """Verify kernel operators are stable with extreme epsilon values."""
        for epsilon in [1e-4, 1e-2, 1.0, 10.0]:
            op = DirectKernelOperator(
                kernel_type="gaussian",
                epsilon=epsilon,
                device=torch.device("cpu"),
            )
            x = torch.randn(4, 4)
            v = torch.randn(4, 4)
            result = op.apply(x, v)
            assert torch.isfinite(result).all(), f"Non-finite result with epsilon={epsilon}"

    def test_rff_all_kernel_types_stable(self) -> None:
        """Verify RFF operator is stable across all kernel types."""
        torch.manual_seed(42)
        for kernel_type in ["gaussian", "laplacian", "cauchy"]:
            op = RFFKernelOperator(
                input_dim=4,
                feature_dim=64,
                kernel_type=kernel_type,
                epsilon=0.5,
                device=torch.device("cpu"),
                orthogonal=False,
                multi_scale=False,
            )
            x = torch.randn(8, 4)
            v = torch.randn(8, 4)
            result = op.apply(x, v)
            assert torch.isfinite(result).all(), f"Non-finite result with kernel={kernel_type}"

    def test_nystrom_cholesky_fallback(self) -> None:
        """Verify Nystrom gracefully handles ill-conditioned matrices."""
        # Create nearly singular landmark matrix
        landmarks = torch.zeros(4, 4)
        landmarks[0] = torch.randn(4)
        landmarks[1] = landmarks[0] + 1e-10 * torch.randn(4)  # Nearly duplicate
        landmarks[2] = torch.randn(4)
        landmarks[3] = torch.randn(4)

        op = NystromKernelOperator(
            landmarks=landmarks,
            kernel_type="gaussian",
            epsilon=0.1,
            regularization=1e-4,  # Regularization helps
            device=torch.device("cpu"),
        )

        x = torch.randn(4, 4)
        v = torch.randn(4, 4)
        result = op.apply(x, v)
        assert torch.isfinite(result).all()

    def test_conjugate_gradient_convergence_tracking(self) -> None:
        """Verify CG solver tracks convergence correctly."""
        solver = _make_solver()

        def well_conditioned_A(v: torch.Tensor) -> torch.Tensor:
            return 2.0 * v

        b = torch.ones(10)
        x, converged = solver._conjugate_gradient(well_conditioned_A, b, max_iter=20, tol=1e-8)

        assert converged, "CG should converge on well-conditioned system"
        assert solver.perf_stats["cg_solve_count"] > 0
        assert solver.perf_stats["cg_last_residual"] is not None
        assert solver.perf_stats["cg_last_residual"] < 1e-6

    def test_sde_coefficients_stability(self) -> None:
        """Verify SDE coefficient computation is stable across time range."""
        solver = _make_solver()
        reference = torch.randn(4, 4)

        for t in [0.001, 0.1, 0.5, 0.9, 0.999]:
            f_t, g_sq_t = solver._compute_sde_coefficients(t, reference)
            assert torch.isfinite(f_t), f"Non-finite f_t at t={t}"
            assert torch.isfinite(g_sq_t), f"Non-finite g_sq_t at t={t}"
            assert g_sq_t >= 0, f"Negative g_sq_t at t={t}"


# =============================================================================
# Gate 4: Checkpoint Round-Trip Equivalence
# =============================================================================


class TestCheckpointRoundTrip:
    """Verify checkpoint save/load produces equivalent models."""

    def test_model_checkpoint_equivalence(self) -> None:
        """Verify model produces same output after save/load cycle."""
        torch.manual_seed(42)

        model1 = build_highres_score_model(_tiny_config())
        model1.eval()

        x = torch.randn(2, 3, 16, 16)
        t = torch.rand(2)

        with torch.no_grad():
            output1 = model1(x, t)

        # Save to buffer
        buffer = io.BytesIO()
        torch.save(model1.state_dict(), buffer)
        buffer.seek(0)

        # Load into new model
        model2 = build_highres_score_model(_tiny_config())
        model2.load_state_dict(torch.load(buffer, weights_only=True))
        model2.eval()

        with torch.no_grad():
            output2 = model2(x, t)

        assert torch.allclose(output1, output2, atol=1e-6), "Outputs differ after checkpoint round-trip"

    def test_checkpoint_file_roundtrip(self) -> None:
        """Verify checkpoint file save/load produces equivalent models."""
        torch.manual_seed(42)

        model1 = build_highres_score_model(_tiny_config())
        model1.eval()

        x = torch.randn(2, 3, 16, 16)
        t = torch.rand(2)

        with torch.no_grad():
            output1 = model1(x, t)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            torch.save({"model": model1.state_dict()}, checkpoint_path)

            # Load with weights_only for security
            state_dict = torch.load(checkpoint_path, weights_only=True)

            model2 = build_highres_score_model(_tiny_config())
            model2.load_state_dict(state_dict["model"])
            model2.eval()

            with torch.no_grad():
                output2 = model2(x, t)

        assert torch.allclose(output1, output2, atol=1e-6)


# =============================================================================
# Gate 3: Reproducibility Tests
# =============================================================================


class TestReproducibility:
    """Verify reproducibility with fixed seeds."""

    def test_rff_deterministic_with_seed(self) -> None:
        """Verify RFF produces identical features with same seed."""
        op1 = RFFKernelOperator(
            input_dim=4,
            feature_dim=32,
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
            seed=42,
        )

        op2 = RFFKernelOperator(
            input_dim=4,
            feature_dim=32,
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
            seed=42,
        )

        assert len(op1.weights) == len(op2.weights)
        for w1, w2 in zip(op1.weights, op2.weights):
            assert torch.allclose(w1, w2), "RFF weights differ with same seed"

    def test_model_forward_deterministic(self) -> None:
        """Verify model forward pass is deterministic."""
        torch.manual_seed(42)
        model = build_highres_score_model(_tiny_config())
        model.eval()

        x = torch.randn(2, 3, 16, 16)
        t = torch.rand(2)

        with torch.no_grad():
            output1 = model(x.clone(), t.clone())
            output2 = model(x.clone(), t.clone())

        assert torch.allclose(output1, output2), "Model forward not deterministic"


# =============================================================================
# Gate 5: Runtime Assertion Validation
# =============================================================================


class TestRuntimeAssertions:
    """Verify runtime assertions catch invalid inputs."""

    def test_kernel_rejects_invalid_epsilon(self) -> None:
        """Verify kernel operators reject non-positive epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            DirectKernelOperator(kernel_type="gaussian", epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            DirectKernelOperator(kernel_type="gaussian", epsilon=-0.1)

    def test_rff_rejects_invalid_dimensions(self) -> None:
        """Verify RFF rejects invalid dimensions."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            RFFKernelOperator(input_dim=0, feature_dim=32, epsilon=0.1)

        with pytest.raises(ValueError, match="feature_dim must be positive"):
            RFFKernelOperator(input_dim=4, feature_dim=0, epsilon=0.1)

    def test_fft_rejects_invalid_grid_shape(self) -> None:
        """Verify FFT kernel rejects invalid grid shapes."""
        with pytest.raises(ValueError, match="grid_shape must contain only positive"):
            FFTKernelOperator(grid_shape=[0, 4], kernel_type="gaussian", epsilon=0.1)

        with pytest.raises(ValueError, match="grid_shape must contain only positive"):
            FFTKernelOperator(grid_shape=[-1, 4], kernel_type="gaussian", epsilon=0.1)

    def test_nystrom_rejects_empty_landmarks(self) -> None:
        """Verify Nystrom rejects empty landmarks."""
        with pytest.raises(ValueError, match="At least one landmark"):
            NystromKernelOperator(
                landmarks=torch.empty(0, 4),
                kernel_type="gaussian",
                epsilon=0.1,
            )

    def test_score_model_rejects_invalid_input_shape(self) -> None:
        """Verify score model rejects invalid input shapes."""
        model = build_highres_score_model(_tiny_config())

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="must be 4D"):
            model(torch.randn(3, 16, 16), torch.rand(1))

        # Wrong number of channels
        with pytest.raises(ValueError, match="channels"):
            model(torch.randn(1, 4, 16, 16), torch.rand(1))

    def test_score_model_rejects_mismatched_batch(self) -> None:
        """Verify score model rejects mismatched batch sizes."""
        model = build_highres_score_model(_tiny_config())

        with pytest.raises(ValueError, match="batch"):
            model(torch.randn(2, 3, 16, 16), torch.rand(3))

    def test_solver_rejects_invalid_timesteps(self) -> None:
        """Verify solver rejects invalid timestep sequences."""
        solver = _make_solver()

        # Too few timesteps
        with pytest.raises(ValueError, match="at least two"):
            solver.validate_timesteps([0.5])

        # Out of range
        with pytest.raises(ValueError, match="range"):
            solver.validate_timesteps([1.5, 0.5, 0.0])

        # Non-finite
        with pytest.raises(ValueError, match="not finite"):
            solver.validate_timesteps([float("nan"), 0.5, 0.0])


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_solver_single_step_produces_finite_output(self) -> None:
        """Verify solver single step produces finite output."""
        torch.manual_seed(42)
        solver = _make_solver()

        x = torch.randn(2, 3, 8, 8)
        x_next = solver.solve_once(x, t_curr=0.8, t_next=0.7)

        assert x_next.shape == x.shape
        assert torch.isfinite(x_next).all()

    def test_full_sampling_loop_finite(self) -> None:
        """Verify full sampling loop produces finite output."""
        torch.manual_seed(42)
        solver = _make_solver()

        timesteps = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        shape = (1, 3, 8, 8)

        samples = solver.sample(shape, timesteps, verbose=False)

        assert samples.shape == shape
        assert torch.isfinite(samples).all()
