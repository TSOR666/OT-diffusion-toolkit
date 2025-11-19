from typing import Tuple, Union

import torch
import torch.nn.functional as F


try:
    from torch.nn.functional import gaussian_blur  # type: ignore
except ImportError:  # pragma: no cover - older torch versions
    try:
        from torchvision.transforms.functional import gaussian_blur  # type: ignore
    except ImportError:  # pragma: no cover - fallback implementation
        def gaussian_blur(
            img: torch.Tensor,
            kernel_size: Union[int, Tuple[int, int]],
            sigma: Union[float, Tuple[float, float]],
        ) -> torch.Tensor:
            """Fallback Gaussian blur using separable convolution."""

            return separable_gaussian_blur(img, kernel_size, sigma)


def separable_gaussian_blur(
    x: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    sigma: Union[float, Tuple[float, float]],
) -> torch.Tensor:
    """Apply separable Gaussian blur for efficient blurring."""

    device = x.device

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(sigma, (int, float)):
        sigma = (sigma, sigma)

    kernel_size_x, kernel_size_y = kernel_size
    sigma_x, sigma_y = sigma

    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, steps=size, device=device, dtype=x.dtype)
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    kernel_x = _gaussian_kernel(kernel_size_x, sigma_x)
    kernel_y = _gaussian_kernel(kernel_size_y, sigma_y)

    padding_x = kernel_size_x // 2
    padding_y = kernel_size_y // 2

    added_batch_dim = False
    if x.dim() == 3:
        # Treat 3D input as (C, H, W), matching torchvision semantics
        x = x.unsqueeze(0)
        added_batch_dim = True
    elif x.dim() != 4:
        raise ValueError(
            "Expected input tensor with shape (C, H, W) or (N, C, H, W) for Gaussian blur."
        )

    channels = x.shape[1]
    window_x = kernel_x.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    window_y = kernel_y.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    padded = F.pad(x, (padding_x, padding_x, 0, 0), mode='reflect')
    out_h = F.conv2d(padded, window_x, padding=0, groups=channels)
    padded = F.pad(out_h, (0, 0, padding_y, padding_y), mode='reflect')
    out = F.conv2d(padded, window_y, padding=0, groups=channels)

    if added_batch_dim:
        out = out.squeeze(0)

    return out
