import os
import warnings
from typing import Dict, Optional

import torch

MB_TO_BYTES = 1024 ** 2


class MemoryWarning(UserWarning):
    """Warning about high GPU memory usage."""


def reset_peak_memory(device: Optional[int] = None) -> None:
    """Reset CUDA peak memory statistics if available."""
    if not torch.cuda.is_available():
        return
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_mb(device: Optional[int] = None) -> float:
    """Return peak CUDA memory usage in megabytes."""
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(device) / MB_TO_BYTES


def get_current_memory_mb(device: Optional[int] = None) -> float:
    """Return current CUDA memory allocated in megabytes."""
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = torch.cuda.current_device()
    return torch.cuda.memory_allocated(device) / MB_TO_BYTES


def get_memory_summary(device: Optional[int] = None) -> Dict[str, float]:
    """Return current, peak, reserved, free, and total memory (MB)."""
    if not torch.cuda.is_available():
        return {
            "current_mb": 0.0,
            "peak_mb": 0.0,
            "reserved_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
        }
    if device is None:
        device = torch.cuda.current_device()
    current = torch.cuda.memory_allocated(device) / MB_TO_BYTES
    peak = torch.cuda.max_memory_allocated(device) / MB_TO_BYTES
    reserved = torch.cuda.memory_reserved(device) / MB_TO_BYTES
    total = torch.cuda.get_device_properties(device).total_memory / MB_TO_BYTES
    free = max(0.0, total - reserved)
    return {
        "current_mb": current,
        "peak_mb": peak,
        "reserved_mb": reserved,
        "free_mb": free,
        "total_mb": total,
    }


def warn_on_high_memory(
    peak_mb: float,
    threshold_mb: Optional[float] = None,
    device: Optional[int] = None,
) -> None:
    """Emit a warning if memory usage crosses the threshold."""
    if os.environ.get("ATLAS_DISABLE_MEMORY_WARNINGS", "0") == "1":
        return
    if threshold_mb is None:
        threshold_mb = float(os.environ.get("ATLAS_MEMORY_THRESHOLD_MB", "4096"))
    if peak_mb <= threshold_mb:
        return

    context = ""
    suggestions = []
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        try:
            props = torch.cuda.get_device_properties(device)
            total_mb = props.total_memory / MB_TO_BYTES
            percent_used = (peak_mb / total_mb) * 100
            context = f" ({percent_used:.1f}% of {total_mb:.0f} MB on cuda:{device})"
            if percent_used > 90:
                suggestions.append("Reduce batch size immediately (near OOM).")
            elif percent_used > 80:
                suggestions.append("Reduce batch size.")
            suggestions.extend(
                [
                    "Enable mixed precision (fp16/bf16).",
                    "Use gradient checkpointing.",
                    "Call torch.cuda.empty_cache().",
                ]
            )
        except Exception:
            pass
    suggestion_text = "\nConsider: " + " ".join(suggestions) if suggestions else ""
    warnings.warn(
        f"High memory usage detected: {peak_mb:.2f} MB exceeds threshold of {threshold_mb:.2f} MB{context}.{suggestion_text}",
        MemoryWarning,
        stacklevel=2,
    )


__all__ = [
    "reset_peak_memory",
    "get_peak_memory_mb",
    "get_current_memory_mb",
    "get_memory_summary",
    "warn_on_high_memory",
    "MemoryWarning",
]
