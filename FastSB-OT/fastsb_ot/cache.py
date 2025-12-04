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
    """Thread-safe cache with smart GPU cleanup and immutable returns"""

    def __init__(self, max_size_mb: int = 1024, max_entries: int = 100,
                 cuda_flush_watermark: float = 0.8, flush_threshold_mb: int = 32):
        # TYPE SAFETY FIX: Add explicit type annotations for cache
        self.cache: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self.max_size_bytes: int = max_size_mb * 1024 * 1024
        self.max_entries: int = max_entries
        self.current_size: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.cuda_flush_watermark: float = cuda_flush_watermark
        self.flush_threshold_bytes: int = flush_threshold_mb * 1024 * 1024
        self._lock: threading.Lock = threading.Lock()
        self._last_cuda_flush: float = 0.0
        self._pending_flush_bytes: int = 0  # TYPE FIX: Explicit int type
        self._adaptive_resize_attempts: int = 0

    def get(self, key: Any, clone: bool = False) -> Optional[torch.Tensor]:
        """Get from cache with optional cloning for immutability"""
        with self._lock:
            if key in self.cache:
                self.hits += 1
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.clone() if clone else value
            self.misses += 1
            return None

    def put(self, key: Any, value: torch.Tensor) -> None:
        """Put tensor in cache with proper cleanup"""
        if value.requires_grad:
            value = value.detach()

        with self._lock:
            value_size = value.element_size() * value.nelement()

            # Log when rejecting large items
            if value_size > self.max_size_bytes // 2:
                if not hasattr(self, '_large_item_warned'):
                    size_mb = value_size / (1024 * 1024)
                    max_mb = self.max_size_bytes / (1024 * 1024)
                    logger.debug(f"Cache rejecting large tensor ({size_mb:.1f} MB > {max_mb/2:.1f} MB). "
                                 f"Consider increasing cache_size_mb if hit rate is low.")
                    self._large_item_warned = True
                self._maybe_downscale_cache()
                return

            while (self.current_size + value_size > self.max_size_bytes or
                   len(self.cache) >= self.max_entries):
                if not self.cache:
                    break

                old_key, old_value = self.cache.popitem(last=False)
                old_size = old_value.element_size() * old_value.nelement()
                self.current_size -= old_size

                if old_value.device.type == 'cuda':
                    self._pending_flush_bytes += old_size
                del old_value
                self._conditional_cuda_flush()

            self.cache[key] = value
            self.current_size += value_size
            self._maybe_downscale_cache()

    def _conditional_cuda_flush(self) -> None:
        """Only flush when significant unused memory is reserved, reset counter properly"""
        if not torch.cuda.is_available():
            self._pending_flush_bytes = 0
            return

        current_time = time.time()

        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            unused = reserved - allocated

            # Flush immediately if both conditions met, regardless of time gate
            should_flush = False

            if self._pending_flush_bytes >= self.flush_threshold_bytes and unused >= self.flush_threshold_bytes:
                # Both pending and unused exceed threshold - flush immediately
                should_flush = True
            elif current_time - self._last_cuda_flush >= 1.0:
                # Time gate passed, check if either condition warrants flush
                if reserved > 0 and unused > self.flush_threshold_bytes:
                    should_flush = True
                elif self._pending_flush_bytes >= self.flush_threshold_bytes:
                    should_flush = True

            if should_flush:
                torch.cuda.empty_cache()
                self._last_cuda_flush = current_time
                self._pending_flush_bytes = 0
                self._maybe_downscale_cache()
            elif self._pending_flush_bytes >= self.flush_threshold_bytes:
                # Reset if we've accumulated enough but can't flush yet
                # Only reset if time gate hasn't passed
                if current_time - self._last_cuda_flush < 1.0:
                    pass  # Keep accumulating
                else:
                    self._pending_flush_bytes = 0
        except Exception:  # POLISH: Narrowed except
            torch.cuda.empty_cache()
            self._last_cuda_flush = current_time
            self._pending_flush_bytes = 0

    def clear(self) -> None:
        """Clear cache and free GPU memory"""
        with self._lock:
            for value in self.cache.values():
                del value

            self.cache.clear()
            self.current_size = 0
            self._pending_flush_bytes = 0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self._maybe_downscale_cache(force=True)

    def reset(self) -> None:
        """Alias for clear() for backward compatibility"""
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
        if force or self.current_size > self.max_size_bytes * 0.75 or target_limit < self.max_size_bytes:
            new_cap = max(target_limit, self.flush_threshold_bytes * 2)
            if new_cap < self.max_size_bytes:
                logger.debug(
                    "MemoryEfficientCacheFixed adapting capacity: %.1fMB -> %.1fMB (free %.1fMB)",
                    self.max_size_bytes / 1e6,
                    new_cap / 1e6,
                    free_bytes / 1e6
                )
                self.max_size_bytes = new_cap
                self._adaptive_resize_attempts += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "size_mb": self.current_size / (1024 * 1024),
            "entries": len(self.cache),
            "pending_flush_mb": self._pending_flush_bytes / (1024 * 1024)
        }


