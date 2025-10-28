from typing import Optional

import torch


def reset_peak_memory() -> None:
    """Reset CUDA peak memory statistics if available."""

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb() -> float:
    """Return current CUDA peak memory usage in megabytes."""

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def warn_on_high_memory(peak_mb: float, threshold_mb: float = 4096.0) -> None:
    """Emit a warning if memory usage crosses the threshold.

    Args:
        peak_mb: Current peak memory usage in megabytes
        threshold_mb: Memory threshold in megabytes (default: 4096 MB)
    """
    import os
    import warnings

    # Allow disabling warnings via environment variable
    if os.environ.get('ATLAS_DISABLE_MEMORY_WARNINGS') == '1':
        return

    if peak_mb > threshold_mb:
        warnings.warn(
            f"High memory usage detected: {peak_mb:.2f} MB exceeds threshold of {threshold_mb:.2f} MB",
            ResourceWarning,
            stacklevel=2
        )
