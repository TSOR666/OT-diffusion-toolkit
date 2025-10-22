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


def warn_on_high_memory(logger: Optional[object], threshold_ratio: float = 0.8) -> None:
    """Emit a warning via logger if memory usage crosses the threshold."""

    if not torch.cuda.is_available() or logger is None:
        return

    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / (1024 * 1024)
    current_mb = get_peak_memory_mb()
    if current_mb > threshold_ratio * total_mb:
        logger.warning(f"High memory usage detected: {current_mb:.2f} MB")
