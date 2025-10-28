"""Utility helpers for the SBDS package."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


def create_standard_timesteps(num_steps: int = 50, schedule_type: str = "linear") -> List[float]:
    """Create standard timesteps for the reverse process."""

    if schedule_type == "linear":
        return torch.linspace(1.0, 0.0, num_steps + 1).tolist()
    if schedule_type == "quadratic":
        return (torch.linspace(1.0, 0.0, num_steps + 1) ** 2).tolist()
    if schedule_type == "log":
        return torch.exp(torch.linspace(0.0, -10.0, num_steps + 1)).tolist()
    raise ValueError(f"Unknown schedule type: {schedule_type}")


def spectral_gradient(
    u: torch.Tensor,
    grid_spacing: Optional[List[float]] = None,
    apply_filter: bool = True,
) -> List[torch.Tensor]:
    """Compute gradient using FFT for spectral accuracy."""

    grid_shape = u.shape
    ndim = len(grid_shape)

    if grid_spacing is None:
        grid_spacing = [1.0] * ndim

    gradients = []

    for dim in range(ndim):
        size = grid_shape[dim]
        if size % 2 == 0:
            freq = torch.cat(
                [
                    torch.arange(0, size // 2, device=u.device),
                    torch.tensor([0], device=u.device),
                    torch.arange(-size // 2 + 1, 0, device=u.device),
                ]
            )
        else:
            freq = torch.cat(
                [
                    torch.arange(0, (size - 1) // 2 + 1, device=u.device),
                    torch.arange(-(size - 1) // 2, 0, device=u.device),
                ]
            )

        freq = freq * (2 * np.pi / (size * grid_spacing[dim]))
        freq_shape = [1] * ndim
        freq_shape[dim] = size
        freq = freq.reshape(freq_shape)

        u_fft = torch.fft.fftn(u)

        if apply_filter:
            filter_shape = [grid_shape[i] for i in range(ndim)]
            filter_tensor = torch.ones(filter_shape, device=u.device)
            center = [s // 2 for s in filter_shape]
            indices = torch.meshgrid(
                [torch.arange(s, device=u.device) for s in filter_shape],
                indexing="ij",
            )
            dist_sq = sum(
                ((idx - c) / (s * 0.5)) ** 2 for idx, c, s in zip(indices, center, filter_shape)
            )
            filter_tensor = 0.5 * (1 + torch.cos(torch.clamp(dist_sq * np.pi, 0, np.pi)))
            u_fft = u_fft * filter_tensor

        grad_fft = u_fft * (1j * freq)
        gradients.append(torch.fft.ifftn(grad_fft).real)

    return gradients
