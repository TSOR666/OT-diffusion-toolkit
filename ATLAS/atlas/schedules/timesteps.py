from typing import Iterable, List

import numpy as np


def linear_timesteps(num_steps: int, start: float = 1.0, end: float = 0.01) -> List[float]:
    """Return monotonically decreasing linear timesteps."""

    return np.linspace(start, end, num_steps).tolist()


def cosine_timesteps(num_steps: int, start: float = 1.0, end: float = 0.0) -> List[float]:
    """Return cosine-spaced timesteps for smoother transitions."""

    steps = np.linspace(0, np.pi / 2, num_steps)
    schedule = start * (np.cos(steps) ** 2)
    schedule = np.clip(schedule, end, start)
    return schedule.tolist()


def custom_timesteps(values: Iterable[float]) -> List[float]:
    """Normalize and sort a custom iterable of timesteps."""

    arr = np.array(list(values), dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.sort(arr)[::-1]
    return arr.tolist()
