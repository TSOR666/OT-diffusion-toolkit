"""Caching utilities for FastSB-OT."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

from . import common

logger = common.logger

__all__ = ["MemoryEfficientCacheFixed"]


class MemoryEfficientCacheFixed:
    """Thread-safe cache with smart GPU cleanup and immutable returns.

    Fixes:
    - Type validation and view storage compaction to avoid hidden leaks
    - CUDA operations performed outside the main lock
    - Multi-GPU memory tracking with adaptive downscaling
    - Thread-safe stats reporting
    """

    def __init__(self, max_size_mb: int = 1024, max_entries: int = 100,
                 cuda_flush_watermark: float = 0.8, flush_threshold_mb: int = 32) -> None:
        self.cache: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self.max_size_bytes: int = max_size_mb * 1024 * 1024
        self._original_max_size_bytes: int = self.max_size_bytes
        self.max_entries: int = max_entries
        self.current_size: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.cuda_flush_watermark: float = cuda_flush_watermark
        self.flush_threshold_bytes: int = flush_threshold_mb * 1024 * 1024
        self._lock: threading.Lock = threading.Lock()
        self._last_cuda_flush: float = 0.0
        self._pending_flush_bytes: int = 0
        self._adaptive_resize_attempts: int = 0
        self._large_item_warned: bool = False

    def get(self, key: Any, clone: bool = False) -> Optional[torch.Tensor]:
        """Get from cache with optional cloning for immutability."""
        with self._lock:
            if key in self.cache:
                self.hits += 1
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.clone() if clone else value
            self.misses += 1
            return None

    def put(self, key: Any, value: torch.Tensor) -> None:
        """Put tensor in cache with proper cleanup."""
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(value).__name__}")

        if value.requires_grad:
            value = value.detach()

        # Check for storage overhead (view memory leak prevention) outside lock
        if value.numel() > 0:
            view_size = value.element_size() * value.nelement()
            storage_size = value.untyped_storage().nbytes()
            if storage_size > view_size * 1.5:
                value = value.clone()

        evicted_cuda_bytes = 0

        with self._lock:
            value_size = value.element_size() * value.nelement()

            if value_size > self.max_size_bytes // 2:
                if not self._large_item_warned:
                    size_mb = value_size / (1024 * 1024)
                    max_mb = self.max_size_bytes / (1024 * 1024)
                    logger.debug(
                        f"Cache rejecting large tensor ({size_mb:.1f} MB > {max_mb/2:.1f} MB). "
                        "Consider increasing cache_size_mb if hit rate is low."
                    )
                    self._large_item_warned = True
                return

            while (self.current_size + value_size > self.max_size_bytes or
                   len(self.cache) >= self.max_entries):
                if not self.cache:
                    break

                old_key, old_value = self.cache.popitem(last=False)
                old_size = old_value.element_size() * old_value.nelement()
                self.current_size -= old_size

                if old_value.device.type == "cuda":
                    evicted_cuda_bytes += old_size
                del old_value

            self.cache[key] = value
            self.current_size += value_size

            # Track pending flush bytes under lock
            self._pending_flush_bytes += evicted_cuda_bytes

            pending_bytes = self._pending_flush_bytes
            last_flush_time = self._last_cuda_flush

        if evicted_cuda_bytes > 0:
            self._try_cuda_flush_unlocked(pending_bytes, last_flush_time)

        self._maybe_downscale_cache()

    def _try_cuda_flush_unlocked(self, pending_bytes: int, last_flush_time: float) -> None:
        """Attempt CUDA cache flush without holding the main lock."""
        if not torch.cuda.is_available():
            with self._lock:
                self._pending_flush_bytes = 0
            return

        current_time = time.time()
        should_flush = False

        try:
            allocated, reserved = self._get_total_cuda_memory()
            unused = reserved - allocated

            if pending_bytes >= self.flush_threshold_bytes and unused >= self.flush_threshold_bytes:
                should_flush = True
            elif current_time - last_flush_time >= 1.0:
                if unused > self.flush_threshold_bytes or pending_bytes >= self.flush_threshold_bytes:
                    should_flush = True

            if should_flush:
                torch.cuda.empty_cache()
                with self._lock:
                    self._pending_flush_bytes = 0
                    self._last_cuda_flush = current_time
                self._maybe_downscale_cache(force=True)
            elif pending_bytes >= self.flush_threshold_bytes and current_time - last_flush_time >= 1.0:
                with self._lock:
                    self._pending_flush_bytes = 0
        except Exception as e:
            logger.debug(f"CUDA flush error: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            with self._lock:
                self._pending_flush_bytes = 0
                self._last_cuda_flush = current_time

    def _get_total_cuda_memory(self) -> tuple[int, int]:
        """Get total allocated and reserved memory across all CUDA devices."""
        if not torch.cuda.is_available():
            return 0, 0

        allocated = 0
        reserved = 0

        try:
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                allocated += torch.cuda.memory_allocated(device_id)
                reserved += torch.cuda.memory_reserved(device_id)
        except Exception:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

        return allocated, reserved

    def clear(self) -> None:
        """Clear cache and free GPU memory."""
        with self._lock:
            self.cache.clear()
            self.current_size = 0
            self._pending_flush_bytes = 0

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._maybe_downscale_cache(force=True)

    def reset(self) -> None:
        """Alias for clear() for backward compatibility."""
        self.clear()

    def _maybe_downscale_cache(self, force: bool = False) -> None:
        """Adapt cache limits to available GPU memory to reduce manual tuning."""
        if not torch.cuda.is_available():
            return
        if not hasattr(torch.cuda, "mem_get_info"):
            return
        if self._adaptive_resize_attempts >= 3 and not force:
            return

        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            return

        target_limit = max(int(free_bytes * 0.15), self.flush_threshold_bytes * 2)

        with self._lock:
            current_max = self.max_size_bytes
            current_usage = self.current_size

        should_downscale = False
        if force:
            should_downscale = True
        elif current_usage > current_max * 0.75 and target_limit < current_max:
            should_downscale = True
        elif target_limit < current_max * 0.7:
            should_downscale = True

        if should_downscale:
            new_cap = max(target_limit, self.flush_threshold_bytes * 2)
            new_cap = min(new_cap, self._original_max_size_bytes)

            if new_cap < current_max:
                logger.debug(
                    "MemoryEfficientCacheFixed adapting capacity: %.1fMB -> %.1fMB (free %.1fMB)",
                    current_max / 1e6,
                    new_cap / 1e6,
                    free_bytes / 1e6
                )
                with self._lock:
                    self.max_size_bytes = new_cap
                self._adaptive_resize_attempts += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0,
                "size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "entries": len(self.cache),
                "pending_flush_mb": self._pending_flush_bytes / (1024 * 1024),
            }
