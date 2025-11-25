from typing import Iterable

import numpy as np


def linear_timesteps(
    num_steps: int,
    start: float = 1.0,
    end: float = 0.01,
    dtype=np.float32,
) -> np.ndarray:
    """
    Monotonically decreasing linear time points (high → low noise).

    Args:
        num_steps: Number of timesteps (must be positive).
        start: Starting time value (default: 1.0, max noise).
        end: Ending time value (default: 0.01, near-zero noise).
        dtype: NumPy dtype for the returned array (default: float32).
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if start <= end:
        raise ValueError(f"start must be greater than end for a descending schedule (start={start}, end={end}).")
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError(f"start and end must be within [0, 1]; got start={start}, end={end}.")

    schedule = np.linspace(start, end, num_steps, dtype=dtype)
    return schedule


def cosine_timesteps(
    num_steps: int,
    start: float = 1.0,
    end: float = 0.0,
    offset: float = 0.008,
    dtype=np.float32,
) -> np.ndarray:
    """
    Cosine schedule matching Nichol & Dhariwal (2021), descending from start to end.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if offset < 0.0:
        raise ValueError(f"offset must be non-negative, got {offset}.")

    steps = np.linspace(0.0, 1.0, num_steps, dtype=dtype)
    f = np.cos(((steps + offset) / (1.0 + offset)) * np.pi / 2.0) ** 2
    f = f / f[0]  # normalize to start value
    schedule = np.clip(f * start, end, start)
    return schedule


def custom_timesteps(values: Iterable[float], dtype=np.float32) -> np.ndarray:
    """
    Normalize, deduplicate, and sort a custom iterable of timesteps in descending order.
    """
    arr = np.array(list(values), dtype=dtype)
    if arr.size == 0:
        raise ValueError("custom_timesteps requires at least one value.")
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.unique(arr)
    arr = np.sort(arr)[::-1]
    return arr
