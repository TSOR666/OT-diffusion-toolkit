from typing import Union

import torch


def karras_noise_schedule(t: Union[float, torch.Tensor], sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0) -> Union[float, torch.Tensor]:
    """
    Continuous noise schedule from Karras et al. (2022).
    
    Maps time `t in [0, 1]` to an alpha value compatible with DDPM-style updates.
    """
    if isinstance(t, torch.Tensor):
        device = t.device
        t = t.float().clamp(0.0, 1.0)
        sigma_min_root = sigma_min ** (1.0 / rho)
        sigma_max_root = sigma_max ** (1.0 / rho)
        sigma = (sigma_max_root + t * (sigma_min_root - sigma_max_root)) ** rho
        alpha = 1.0 / (1.0 + sigma ** 2)
        return alpha.to(device)
    else:
        t = float(max(0.0, min(1.0, t)))
        sigma_min_root = sigma_min ** (1.0 / rho)
        sigma_max_root = sigma_max ** (1.0 / rho)
        sigma = (sigma_max_root + t * (sigma_min_root - sigma_max_root)) ** rho
        alpha = 1.0 / (1.0 + sigma ** 2)
        return alpha
