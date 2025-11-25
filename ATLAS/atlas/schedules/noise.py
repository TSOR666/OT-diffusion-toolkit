from typing import Union

import torch


def karras_noise_schedule(
    t: Union[float, torch.Tensor],
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> Union[float, torch.Tensor]:
    """
    Continuous noise schedule from Karras et al. (2022).

    Args:
        t: Continuous time value in [0, 1] where 0 corresponds to maximum noise
           and 1 to minimum noise (start → end of sampling).
        sigma_min: Minimum noise level (end of sampling).
        sigma_max: Maximum noise level (start of sampling).
        rho: Controls the curvature of the schedule (higher = more curved).

    Returns:
        Cumulative alpha values (alpha_bar) compatible with DDPM-style updates.
    """
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive.")
    if sigma_min >= sigma_max:
        raise ValueError("sigma_min must be strictly less than sigma_max.")
    if rho <= 0:
        raise ValueError("rho must be positive.")

    if isinstance(t, torch.Tensor):
        device = t.device
        dtype = t.dtype if torch.is_floating_point(t) else torch.float32
        t = t.to(dtype=dtype).clamp(0.0, 1.0)
        sigma_min_root = torch.tensor(sigma_min, device=device, dtype=dtype) ** (1.0 / rho)
        sigma_max_root = torch.tensor(sigma_max, device=device, dtype=dtype) ** (1.0 / rho)
        sigma = (sigma_max_root + t * (sigma_min_root - sigma_max_root)) ** rho
        alpha_bar = 1.0 / (1.0 + sigma ** 2)
        return alpha_bar
    else:
        t = float(max(0.0, min(1.0, t)))
        sigma_min_root = sigma_min ** (1.0 / rho)
        sigma_max_root = sigma_max ** (1.0 / rho)
        sigma = (sigma_max_root + t * (sigma_min_root - sigma_max_root)) ** rho
        alpha_bar = 1.0 / (1.0 + sigma ** 2)
        return alpha_bar
