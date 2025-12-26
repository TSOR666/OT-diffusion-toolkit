from __future__ import annotations

import torch
import torch.nn as nn

from atlas.kernels.base import KernelOperator
from atlas.kernels.fft import FFTKernelOperator
from atlas.kernels.rff import RFFKernelOperator
from atlas.solvers.schrodinger_bridge import SchroedingerBridgeSolver


class _DummyModel(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.zeros_like(x)


def _dummy_schedule(_: float | torch.Tensor) -> float:
    return 0.9


def _build_solver() -> SchroedingerBridgeSolver:
    return SchroedingerBridgeSolver(
        _DummyModel(),
        _dummy_schedule,
        device=torch.device("cpu"),
    )


class _FixedKernelOperator(KernelOperator):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__(epsilon=1.0, device=weights.device)
        self._weights = weights

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._weights

    def get_error_bound(self, n_samples: int) -> float:
        return 0.0


def test_rff_laplacian_sampling_stability() -> None:
    op = RFFKernelOperator(
        input_dim=10,
        feature_dim=512,
        kernel_type="laplacian",
        epsilon=1.0,
        device=torch.device("cpu"),
        orthogonal=False,
        multi_scale=False,
        seed=42,
    )
    weights = op.weights[0]
    assert torch.isfinite(weights).all()
    assert float(weights.abs().max()) < 1e5


def test_cauchy_sampling_is_finite_and_centered() -> None:
    op = RFFKernelOperator(
        input_dim=10,
        feature_dim=4096,
        kernel_type="cauchy",
        epsilon=1.0,
        device=torch.device("cpu"),
        orthogonal=False,
        multi_scale=False,
        seed=42,
    )
    weights = op.weights[0]
    assert torch.isfinite(weights).all()
    assert float(weights.median().abs()) < 0.2


def test_alpha_sigma_conversion_stability() -> None:
    solver = _build_solver()
    alpha_values = torch.tensor([1e-8, 0.01, 0.5, 0.99, 1.0 - 1e-8])
    sigma = solver._compute_sigma(alpha_values)
    assert torch.isfinite(sigma).all()
    assert bool((sigma > 0).all())


def test_fft_kernel_normalization_stability() -> None:
    grid_shape = [16, 16]
    op = FFTKernelOperator(
        grid_shape=grid_shape,
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
        multi_scale=False,
    )
    kernel = torch.fft.irfftn(op.kernel_fft, s=grid_shape)
    kernel_sum = kernel.sum()
    expected = torch.tensor(1.0, device=kernel_sum.device, dtype=kernel_sum.dtype)
    assert torch.isfinite(kernel).all()
    assert torch.isclose(kernel_sum, expected, atol=1e-4)


def test_transport_map_row_stochasticity() -> None:
    solver = _build_solver()
    batch = 4
    weights = torch.rand(batch, batch)
    kernel_op = _FixedKernelOperator(weights)
    f = torch.ones(batch)
    g = torch.ones(batch)
    x_curr = torch.zeros(batch, batch)
    x_next_pred = torch.eye(batch)
    transport_map = solver._construct_transport_map(
        f=f,
        g=g,
        kernel_op=kernel_op,
        x_curr=x_curr,
        x_next_pred=x_next_pred,
    )
    z = torch.zeros(batch, batch)
    z_next = transport_map(z)
    row_sums = z_next.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
