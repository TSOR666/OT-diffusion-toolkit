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

    kernel_x = torch.exp(
        -torch.arange(-(kernel_size_x // 2), kernel_size_x // 2 + 1, device=device, dtype=x.dtype) ** 2
        / (2 * sigma_x ** 2)
    )
    kernel_x = kernel_x / kernel_x.sum()

    kernel_y = torch.exp(
        -torch.arange(-(kernel_size_y // 2), kernel_size_y // 2 + 1, device=device, dtype=x.dtype) ** 2
        / (2 * sigma_y ** 2)
    )
    kernel_y = kernel_y / kernel_y.sum()

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

    if channels > 1:
        out = x.clone()
        for c in range(channels):
            channel = x[:, c:c + 1]
            padded = F.pad(channel, (padding_x, padding_x, 0, 0), mode='reflect')
            out_h = F.conv2d(padded, kernel_x.view(1, 1, 1, -1), padding=0)
            padded = F.pad(out_h, (0, 0, padding_y, padding_y), mode='reflect')
            out[:, c:c + 1] = F.conv2d(padded, kernel_y.view(1, 1, -1, 1), padding=0)
        return out

    padded = F.pad(x, (padding_x, padding_x, 0, 0), mode='reflect')
    out_h = F.conv2d(padded, kernel_x.view(1, 1, 1, -1), padding=0)
    padded = F.pad(out_h, (0, 0, padding_y, padding_y), mode='reflect')
    out = F.conv2d(padded, kernel_y.view(1, 1, -1, 1), padding=0)

    if added_batch_dim:
        out = out.squeeze(0)

    return out
