from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = ["gaussian_blur", "separable_gaussian_blur"]


try:
    from torch.nn.functional import gaussian_blur  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - older torch versions
    try:
        from torchvision.transforms.functional import gaussian_blur
    except ImportError:  # pragma: no cover - fallback implementation
        def gaussian_blur(
            img: torch.Tensor,
            kernel_size: list[int],
            sigma: Optional[list[float]] = None,
        ) -> torch.Tensor:
            """Fallback Gaussian blur using separable convolution."""
            if sigma is None:
                sigma = [
                    0.3 * ((k - 1) * 0.5 - 1) + 0.8
                    for k in kernel_size
                ]
            if len(kernel_size) != 2 or len(sigma) != 2:
                raise ValueError(
                    "Fallback gaussian_blur expects 2D kernel_size and sigma, got "
                    f"kernel_size={kernel_size}, sigma={sigma}"
                )
            kernel_size_t = (int(kernel_size[0]), int(kernel_size[1]))
            sigma_t = (float(sigma[0]), float(sigma[1]))
            return separable_gaussian_blur(img, kernel_size_t, sigma_t)


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

    if kernel_size_x <= 0 or kernel_size_y <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if kernel_size_x % 2 == 0 or kernel_size_y % 2 == 0:
        raise ValueError(
            f"Gaussian blur requires odd kernel sizes for symmetric padding; got {kernel_size}. "
            f"Use odd values such as 3, 5, 7."
        )
    if sigma_x < 0 or sigma_y < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        if sigma <= 1e-6:
            kernel = torch.zeros(size, device=device, dtype=x.dtype)
            kernel[size // 2] = 1.0
            return kernel
        coords = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, steps=size, device=device, dtype=x.dtype)
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_sum = kernel.sum()
        if kernel_sum < 1e-10:
            kernel = torch.zeros_like(kernel)
            kernel[size // 2] = 1.0
            return kernel
        return kernel / kernel_sum

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

    if x.shape[2] < kernel_size_y or x.shape[3] < kernel_size_x:
        raise ValueError(
            f"Input spatial dimensions {x.shape[2:]} must be at least kernel_size {kernel_size}."
        )

    channels = x.shape[1]
    window_x = kernel_x.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    window_y = kernel_y.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    pad_mode = 'reflect'
    if padding_x >= x.shape[3] or padding_y >= x.shape[2]:
        pad_mode = 'replicate'
    padded = F.pad(x, (padding_x, padding_x, 0, 0), mode=pad_mode)
    out_h = F.conv2d(padded, window_x, padding=0, groups=channels)
    padded = F.pad(out_h, (0, 0, padding_y, padding_y), mode=pad_mode)
    out = F.conv2d(padded, window_y, padding=0, groups=channels)

    if added_batch_dim:
        out = out.squeeze(0)

    return out
