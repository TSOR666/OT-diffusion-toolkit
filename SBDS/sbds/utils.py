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
    """
    Compute spatial gradients via FFT on the last dimensions of `u`.

    The transform is restricted to spatial axes to avoid mixing batch/channel
    dimensions. An optional low-pass filter attenuates high frequencies.
    """

    if u.dim() < 1:
        raise ValueError("Input tensor must have at least one dimension")

    if u.dim() == 1:
        spatial_dims = [-1]
    elif u.dim() == 2:
        spatial_dims = [-2, -1]
    else:
        spatial_dims = list(range(-2, 0))
    spatial_shape = [u.size(d) for d in spatial_dims]
    ndim = len(spatial_dims)

    if grid_spacing is None:
        grid_spacing = [1.0] * ndim
    if len(grid_spacing) != ndim:
        raise ValueError(f"grid_spacing length {len(grid_spacing)} does not match spatial dims {ndim}")

    gradients = []

    # FFT over spatial dimensions only
    u_fft = torch.fft.fftn(u, dim=spatial_dims)

    # Optional low-pass filter using radial frequency magnitude
    filter_tensor = None
    if apply_filter:
        freq_grids = torch.meshgrid(
            [torch.fft.fftfreq(n=s, d=grid_spacing[i], device=u.device) for i, s in enumerate(spatial_shape)],
            indexing="ij",
        )
        freq_radius = torch.sqrt(sum(f ** 2 for f in freq_grids))
        # Cosine window: 1 at DC, tapering toward Nyquist
        nyquist = max(f.abs().max().item() for f in freq_grids) + 1e-12
        filter_tensor = 0.5 * (1 + torch.cos(np.pi * torch.clamp(freq_radius / nyquist, 0, 1)))
        # Reshape to broadcast over non-spatial dims
        for _ in range(u.dim() - ndim):
            filter_tensor = filter_tensor.unsqueeze(0)
        u_fft = u_fft * filter_tensor

    for axis, dim in enumerate(spatial_dims):
        size = spatial_shape[axis]
        freq_vec = torch.fft.fftfreq(size, d=grid_spacing[axis], device=u.device) * (2 * np.pi)
        shape = [1] * u.dim()
        shape[dim] = size
        freq = freq_vec.reshape(shape)
        grad_fft = u_fft * (1j * freq)
        gradients.append(torch.fft.ifftn(grad_fft, dim=spatial_dims).real)

    return gradients
