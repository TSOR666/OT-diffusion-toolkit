import torch

from atlas.kernels import (
    DirectKernelOperator,
    FFTKernelOperator,
    NystromKernelOperator,
    RFFKernelOperator,
)

def _random_points(batch: int, dim: int) -> torch.Tensor:
    return torch.randn(batch, dim)


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
