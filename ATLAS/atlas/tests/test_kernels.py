import pytest
import torch

from atlas.kernels import (
    DirectKernelOperator,
    FFTKernelOperator,
    NystromKernelOperator,
    RFFKernelOperator,
)

def _random_points(batch: int, dim: int) -> torch.Tensor:
    return torch.randn(batch, dim)


def _grid_coords_2d(height: int, width: int) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32)
    xs = torch.arange(width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)


def test_direct_kernel_operator_shape() -> None:
    op = DirectKernelOperator(kernel_type="gaussian", epsilon=0.1, device=torch.device("cpu"))
    x = _random_points(8, 4)
    v = torch.randn(8)
    out = op.apply(x, v)
    assert out.shape == v.shape


def test_rff_kernel_operator_shape() -> None:
    op = RFFKernelOperator(
        input_dim=4,
        feature_dim=16,
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
        multi_scale=False,
    )
    x = _random_points(8, 4)
    v = torch.randn(8)
    out = op.apply(x, v)
    assert out.shape == v.shape


def test_nystrom_kernel_operator_shape() -> None:
    landmarks = _random_points(6, 4)
    op = NystromKernelOperator(
        landmarks=landmarks,
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
    )
    x = _random_points(5, 4)
    v = torch.randn(5)
    out = op.apply(x, v)
    assert out.shape == v.shape


def test_fft_kernel_operator_shape() -> None:
    op = FFTKernelOperator(
        grid_shape=[4, 4],
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
    )
    x = torch.randn(2, 1, 4, 4)
    v = torch.randn(2, 1, 4, 4)
    out = op.apply(x, v)
    assert out.shape == v.shape


def test_fft_kernel_operator_custom_scales() -> None:
    scales = [0.25, 0.5, 1.0, 2.0]
    op = FFTKernelOperator(
        grid_shape=[4, 4],
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
        multi_scale=True,
        scale_factors=scales,
    )
    assert len(op.kernel_ffts) == len(scales)
    assert op.weights.shape[0] == len(scales)
    assert torch.isclose(op.weights.sum(), torch.tensor(1.0, device=op.weights.device))


def test_fft_kernel_operator_flattened_last_dim_matches_grid() -> None:
    op = FFTKernelOperator(
        grid_shape=[4, 4],
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
        multi_scale=False,
    )
    x = _grid_coords_2d(4, 4)
    v_flat = torch.randn(2, 16)
    out_flat = op.apply(x, v_flat)
    assert out_flat.shape == v_flat.shape

    v_grid = v_flat.reshape(2, 4, 4)
    out_grid = op.apply(x, v_grid)
    assert torch.allclose(out_flat, out_grid.reshape(2, 16), atol=1e-5, rtol=1e-5)


def test_fft_kernel_operator_flattened_first_dim_respects_coordinate_order() -> None:
    torch.manual_seed(0)
    op = FFTKernelOperator(
        grid_shape=[4, 4],
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
        multi_scale=False,
    )
    x = _grid_coords_2d(4, 4)
    v = torch.randn(16, 3)
    out = op.apply(x, v)
    assert out.shape == v.shape

    perm = torch.randperm(16)
    x_perm = x[perm]
    v_perm = v[perm]
    out_perm = op.apply(x_perm, v_perm)
    assert torch.allclose(out_perm, out[perm], atol=1e-5, rtol=1e-5)


def test_fft_kernel_operator_preserves_constant_field() -> None:
    op = FFTKernelOperator(
        grid_shape=[4, 4],
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
        multi_scale=True,
        scale_factors=[0.5, 1.0, 2.0],
    )
    x = _grid_coords_2d(4, 4)
    v = torch.ones(16)
    out = op.apply(x, v)
    assert out.shape == v.shape
    assert torch.allclose(out, v, atol=1e-5, rtol=1e-5)


def test_fft_kernel_operator_clear_cache_raises() -> None:
    op = FFTKernelOperator(
        grid_shape=[4, 4],
        kernel_type="gaussian",
        epsilon=0.1,
        device=torch.device("cpu"),
        multi_scale=False,
    )
    x = _grid_coords_2d(4, 4)
    v = torch.randn(16)
    op.clear_cache()
    with pytest.raises(RuntimeError):
        _ = op.apply(x, v)
