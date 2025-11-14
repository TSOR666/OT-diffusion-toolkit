import torch
import torch.nn as nn

import pytest

from typing import List

from atlas.config import KernelConfig, SamplerConfig
from atlas.solvers import (
    AdvancedHierarchicalDiffusionSampler,
    SchroedingerBridgeSolver,
)


class _ZeroScore(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning=None) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros_like(x)


class _TrackTrainingStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning=None) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def _constant_noise(t):
    if isinstance(t, torch.Tensor):
        return torch.full_like(t, 0.9)
    return 0.9


@pytest.fixture()
def solver() -> SchroedingerBridgeSolver:
    return SchroedingerBridgeSolver(
        score_model=_ZeroScore(),
        noise_schedule=_constant_noise,
        device=torch.device("cpu"),
        kernel_config=KernelConfig(
            kernel_type="gaussian",
            epsilon=0.1,
            solver_type="direct",
            max_kernel_cache_size=4,
        ),
        sampler_config=SamplerConfig(
            sb_iterations=2,
            use_linear_solver=False,
            verbose_logging=False,
        ),
    )


@pytest.fixture()
def hierarchical_sampler() -> AdvancedHierarchicalDiffusionSampler:
    return AdvancedHierarchicalDiffusionSampler(
        score_model=_ZeroScore(),
        noise_schedule=_constant_noise,
        device=torch.device("cpu"),
        kernel_config=KernelConfig(
            kernel_type="gaussian",
            epsilon=0.1,
            solver_type="direct",
        ),
        sampler_config=SamplerConfig(
            sb_iterations=2,
            use_linear_solver=False,
            verbose_logging=False,
        ),
    )


def test_schrodinger_bridge_solver_step(solver: SchroedingerBridgeSolver) -> None:
    x = torch.randn(2, 4)
    next_x = solver.solve_once(x, 1.0, 0.5)

    assert next_x.shape == x.shape


def test_conjugate_gradient_handles_zero_curvature(solver: SchroedingerBridgeSolver) -> None:
    b = torch.ones(4)

    def zero_operator(_: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(b)

    solution, converged = solver._conjugate_gradient(zero_operator, b, max_iter=3)

    assert not converged
    assert torch.all(torch.isfinite(solution))


def test_validate_timesteps_sorts_and_checks_spacing(solver: SchroedingerBridgeSolver) -> None:
    schedule = solver.validate_timesteps([0.2, 1.0, 0.5])
    assert schedule == [1.0, 0.5, 0.2]


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0, 0.5],
        [1.1, 0.5],
        [float("nan"), 0.5],
        [0.3, 0.3 + 5e-7],
    ],
)
def test_validate_timesteps_rejects_invalid_sequences(
    solver: SchroedingerBridgeSolver, values: List[float]
) -> None:
    with pytest.raises(ValueError):
        solver.validate_timesteps(values)


def test_solver_sample_rejects_invalid_shape(solver: SchroedingerBridgeSolver) -> None:
    with pytest.raises(ValueError):
        solver.sample((1,), [1.0, 0.5], verbose=False)


def test_kernel_cache_key_precision(solver: SchroedingerBridgeSolver) -> None:
    solver.clear_kernel_cache()
    x = torch.randn(4, 3)

    solver._select_optimal_kernel_operator(x, 0.123456789)
    solver._select_optimal_kernel_operator(x, 0.123456780)

    assert len(solver.kernel_operators) == 2


def test_clear_kernel_cache(solver: SchroedingerBridgeSolver) -> None:
    solver._select_optimal_kernel_operator(torch.randn(3, 4), solver.epsilon)
    assert len(solver.kernel_operators) >= 1

    solver.clear_kernel_cache()
    assert len(solver.kernel_operators) == 0


def test_invalid_kernel_type_raises() -> None:
    with pytest.raises(ValueError):
        SchroedingerBridgeSolver(
            score_model=_ZeroScore(),
            noise_schedule=_constant_noise,
            device=torch.device("cpu"),
            kernel_config=KernelConfig(kernel_type="unsupported"),
            sampler_config=SamplerConfig(verbose_logging=False),
        )


def test_invalid_kernel_cache_size_raises() -> None:
    with pytest.raises(ValueError):
        SchroedingerBridgeSolver(
            score_model=_ZeroScore(),
            noise_schedule=_constant_noise,
            device=torch.device("cpu"),
            kernel_config=KernelConfig(kernel_type="gaussian", max_kernel_cache_size=0),
            sampler_config=SamplerConfig(verbose_logging=False),
        )


def test_hierarchical_sampler_rejects_duplicate_timesteps(
    hierarchical_sampler: AdvancedHierarchicalDiffusionSampler,
) -> None:
    with pytest.raises(ValueError):
        hierarchical_sampler.sample((1, 1, 4), [0.5, 0.5], verbose=False)


def test_sampler_initializes_in_eval_mode() -> None:
    model = _TrackTrainingStateModel()
    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=model,
        noise_schedule=_constant_noise,
        device=torch.device("cpu"),
    )
    assert sampler.score_model.training is False

    sampler.set_model_training_mode(True)
    assert sampler.score_model.training is True

