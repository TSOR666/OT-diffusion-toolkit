from typing import Iterable, List

import numpy as np


def linear_timesteps(num_steps: int, start: float = 1.0, end: float = 0.01) -> List[float]:
    """Return monotonically decreasing linear time points."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    schedule = np.linspace(start, end, num_steps, dtype=np.float64)
    return schedule.tolist()


def cosine_timesteps(
    num_steps: int,
    start: float = 1.0,
    end: float = 0.0,
    offset: float = 0.008,
) -> List[float]:
    """Cosine schedule matching Nichol & Dhariwal (2021)."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    steps = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    offset = float(max(offset, 0.0))
    f = np.cos(((steps + offset) / (1.0 + offset)) * np.pi / 2.0) ** 2
    f = f / f[0]  # normalize to start value
    schedule = np.clip(f * start, end, start)
    return schedule.tolist()


def custom_timesteps(values: Iterable[float]) -> List[float]:
    """Normalize, deduplicate, and sort a custom iterable of timesteps."""

    arr = np.array(list(values), dtype=np.float32)
    if arr.size == 0:
        raise ValueError("custom_timesteps requires at least one value.")
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.unique(arr)
    arr = np.sort(arr)[::-1]
    return arr.tolist()
