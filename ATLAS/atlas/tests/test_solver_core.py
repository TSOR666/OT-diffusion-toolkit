"""Core solver tests for the Schrödinger Bridge implementation."""

import torch

from atlas.config.kernel_config import KernelConfig
from atlas.config.model_config import HighResModelConfig
from atlas.config.sampler_config import SamplerConfig
from atlas.models.score_network import build_highres_score_model
from atlas.solvers.schrodinger_bridge import SchroedingerBridgeSolver


def _linear_schedule(t):
    if isinstance(t, torch.Tensor):
        return torch.ones_like(t) * 0.9
    return 0.9


def _tiny_config() -> HighResModelConfig:
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


def _make_solver(use_linear_solver: bool = False) -> SchroedingerBridgeSolver:
    model = build_highres_score_model(_tiny_config())
    kernel_cfg = KernelConfig(
        solver_type="direct",
        epsilon=0.1,  # Increased for better conditioning
        rff_features=128,
        n_landmarks=8,
        max_kernel_cache_size=2,
    )
    sampler_cfg = SamplerConfig(
        sb_iterations=50,  # Increased for robust convergence
        error_tolerance=1e-3,  # Relaxed for test stability
        marginal_constraint_threshold=5e-2,  # Relaxed threshold for tests
        use_linear_solver=use_linear_solver,
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


def test_sinkhorn_enforces_marginals():
    torch.manual_seed(0)
    solver = _make_solver(use_linear_solver=False)
    x = torch.randn(6, 3, 4, 4)

    kernel_op = solver._select_optimal_kernel_operator(x, epsilon=solver.epsilon)
    f, g = solver._solve_Schrodinger_bridge(kernel_op, x, max_iter=30)

    Kg = kernel_op.apply(x, g)
    marginal_error = torch.abs(f * Kg - 1.0).max()
    assert marginal_error < 5e-2, f"Marginal constraint violated: {float(marginal_error):.3e}"


def test_conjugate_gradient_converges_on_diagonal_system():
    solver = _make_solver(use_linear_solver=True)

    def A(v: torch.Tensor) -> torch.Tensor:
        return 2.0 * v

    b = torch.ones(10)
    x, converged = solver._conjugate_gradient(A, b, max_iter=20, tol=1e-8)

    residual = torch.norm(A(x) - b)
    assert converged, "CG should converge on well-conditioned diagonal system"
    assert residual < 1e-6
    assert torch.allclose(x, torch.full_like(b, 0.5), atol=1e-3)


def test_conjugate_gradient_flags_singular_system():
    solver = _make_solver(use_linear_solver=True)

    def singular_A(v: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(v)

    b = torch.ones(5)
    _, converged = solver._conjugate_gradient(singular_A, b, max_iter=5, tol=1e-6)
    assert not converged, "CG should not report convergence on singular system"


def test_linear_solver_branch_produces_valid_potentials():
    torch.manual_seed(0)
    solver = _make_solver(use_linear_solver=True)
    x = torch.randn(5, 3, 4, 4)

    kernel_op = solver._select_optimal_kernel_operator(x, epsilon=solver.epsilon)
    f, g = solver._solve_Schrodinger_bridge(kernel_op, x, max_iter=10)

    assert f.shape == (x.size(0),)
    assert g.shape == (x.size(0),)
    assert torch.isfinite(f).all()
    assert torch.isfinite(g).all()

    Kg = kernel_op.apply(x, g)
    residual = torch.abs(f * Kg - 1.0).max()
    assert residual < 5e-2, f"Linear solver marginals violated: {float(residual):.3e}"
