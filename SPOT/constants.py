"""Constant values shared across the SPOT solver package."""
from __future__ import annotations

__all__ = [
    "DEFAULT_GPU_MEMORY_FRACTION",
    "MAX_BLOCK_SIZE",
    "MIN_BLOCK_SIZE",
    "EPSILON_MIN",
    "EPSILON_CLAMP",
    "ROW_SUM_MIN",
    "DEFAULT_SINKHORN_ITERATIONS",
    "DEFAULT_DPM_ORDER",
]

# Memory management
DEFAULT_GPU_MEMORY_FRACTION = 0.25
MAX_BLOCK_SIZE = 1000
MIN_BLOCK_SIZE = 1

# Numerical thresholds
EPSILON_MIN = 1e-6
EPSILON_CLAMP = 1e-12
ROW_SUM_MIN = 1e-12

# Performance
DEFAULT_SINKHORN_ITERATIONS = 10
DEFAULT_DPM_ORDER = 3
