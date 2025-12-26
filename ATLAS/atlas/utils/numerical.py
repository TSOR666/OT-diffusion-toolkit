from __future__ import annotations

import torch


def get_practical_eps(dtype: torch.dtype) -> float:
    """Return a numerically stable epsilon for the given dtype."""
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-3
    if dtype == torch.float32:
        return 1e-6
    return 1e-8
