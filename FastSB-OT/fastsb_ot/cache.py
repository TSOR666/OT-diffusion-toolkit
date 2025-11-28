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
    """Thread-safe cache with smart GPU cleanup and immutable returns

    CRITICAL FIXES:
    - Tensor view storage leak: Compacts views to prevent hidden memory overhead
    - Performance: Removed synchronous CUDA calls from hot path
    - Adaptive sizing: Bidirectional with hysteresis and recovery
    - Thread safety: Proper locking for async flush operations
    """

    def __init__(self, max_size_mb: int = 1024, max_entries: int = 100,
                 cuda_flush_watermark: float = 0.8, flush_threshold_mb: int = 32,
                 enable_adaptive_sizing: bool = False):
        """Initialize cache with memory management parameters.

        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cached entries
            cuda_flush_watermark: Deprecated (kept for backward compatibility)
            flush_threshold_mb: Minimum MB to accumulate before flushing
            enable_adaptive_sizing: Enable dynamic capacity adjustment (experimental)
        """
        self.cache = OrderedDict()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._original_max_size_bytes = self.max_size_bytes  # Store user intent
        self.max_entries = max_entries
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self.cuda_flush_watermark = cuda_flush_watermark  # Unused, kept for compatibility
        self.flush_threshold_bytes = flush_threshold_mb * 1024 * 1024
        self.enable_adaptive_sizing = enable_adaptive_sizing

        # Thread safety
        self._lock = threading.Lock()

        # Async flush tracking (accessed both inside and outside lock)
        self._last_cuda_flush = 0.0
        self._pending_flush_bytes = 0
        self._flush_cooldown_seconds = 1.0

        # Adaptive sizing state
        self._last_capacity_check = 0.0
        self._capacity_check_interval = 10.0  # Check every 10 seconds max
        self._capacity_upscale_count = 0
        self._capacity_downscale_count = 0

        # Warnings
        self._large_item_warned = False

    def get(self, key: Any, clone: bool = False) -> Optional[torch.Tensor]:
        """Get from cache with optional cloning for immutability.

        Args:
            key: Cache key
            clone: If True, return a clone to prevent external modification

        Returns:
            Cached tensor or None if not found
        """
        with self._lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (LRU update)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.clone() if clone else value
            self.misses += 1
            return None

    def put(self, key: Any, value: torch.Tensor):
        """Put tensor in cache with proper memory accounting and cleanup.

        CRITICAL FIX: Compacts tensor views to prevent storage leaks.
        PERFORMANCE FIX: No synchronous CUDA operations in critical path.

        Args:
            key: Cache key
            value: Tensor to cache (will be compacted if necessary)
        """
        # ============================================================
        # FIX #1: TENSOR VIEW STORAGE LEAK
        # ============================================================
        # Calculate actual storage vs. view size BEFORE entering lock
        storage_size = value.untyped_storage().nbytes()
        view_size = value.element_size() * value.nelement()

        # Storage overhead ratio: >1.1 means we have a view with excess backing storage
        storage_overhead_ratio = storage_size / max(view_size, 1)

        if storage_overhead_ratio > 1.1:
            # CRITICAL: Clone to compact storage
            # This prevents caching a small slice that holds a gigantic tensor in memory
            value = value.clone()
        elif value.requires_grad:
            # Standard case: just detach gradients
            value = value.detach()

        # Now we can safely calculate size knowing storage matches view
        value_size = value.element_size() * value.nelement()

        # ============================================================
        # FIX #2: REMOVE SYNCHRONOUS CUDA CALLS FROM HOT PATH
        # ============================================================
        # Track pending flush bytes for async cleanup, but don't call CUDA APIs here
        evicted_cuda_bytes = 0

        with self._lock:
            # Reject oversized items early
            if value_size > self.max_size_bytes // 2:
                if not self._large_item_warned:
                    size_mb = value_size / (1024 * 1024)
                    max_mb = self.max_size_bytes / (1024 * 1024)
                    logger.debug(
                        f"Cache rejecting large tensor ({size_mb:.1f} MB > {max_mb/2:.1f} MB). "
                        f"Consider increasing cache_size_mb if hit rate is low."
                    )
                    self._large_item_warned = True
                return

            # LRU eviction loop - NO CUDA CALLS HERE
            while (self.current_size + value_size > self.max_size_bytes or
                   len(self.cache) >= self.max_entries):
                if not self.cache:
                    break

                old_key, old_value = self.cache.popitem(last=False)
                old_size = old_value.element_size() * old_value.nelement()
                self.current_size -= old_size

                # Track CUDA memory for async flush
                if old_value.device.type == 'cuda':
                    evicted_cuda_bytes += old_size

                del old_value

            # Insert new entry
            self.cache[key] = value
            self.current_size += value_size

            # Update pending flush counter
            self._pending_flush_bytes += evicted_cuda_bytes

        # ============================================================
        # ASYNC FLUSH: Outside lock, non-blocking
        # ============================================================
        self._try_async_cuda_flush()

        # ============================================================
        # FIX #3: ADAPTIVE SIZING WITH RECOVERY
        # ============================================================
        if self.enable_adaptive_sizing:
            self._maybe_adjust_capacity()

    def _try_async_cuda_flush(self):
        """Asynchronous CUDA cache flush with cooldown.

        PERFORMANCE FIX: Called outside the main lock, respects cooldown.
        Does not block put() operations.
        """
        if not torch.cuda.is_available():
            return

        current_time = time.time()

        # Read pending bytes (atomic read, safe without lock)
        pending = self._pending_flush_bytes

        # Only proceed if we've accumulated enough and cooldown passed
        if (pending >= self.flush_threshold_bytes and
            current_time - self._last_cuda_flush >= self._flush_cooldown_seconds):

            try:
                # Perform the expensive synchronous operation
                torch.cuda.empty_cache()

                # Update state with lock
                with self._lock:
                    self._pending_flush_bytes = 0

                self._last_cuda_flush = current_time

            except RuntimeError as e:
                # CUDA errors (e.g., driver issues) should be logged but not crash
                logger.warning(f"CUDA cache flush failed: {e}")
                with self._lock:
                    self._pending_flush_bytes = 0

    def _maybe_adjust_capacity(self):
        """Bidirectional adaptive capacity adjustment with hysteresis.

        FIX #3: Allows cache to both shrink AND grow based on GPU memory.
        Includes rate limiting to avoid thrashing.
        """
        current_time = time.time()

        # Rate limit: Don't check more than once per interval
        if current_time - self._last_capacity_check < self._capacity_check_interval:
            return

        if not hasattr(torch.cuda, "mem_get_info"):
            return

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except RuntimeError:
            return

        self._last_capacity_check = current_time

        # Target: Use 15% of free memory, but at least 2x flush threshold
        target_capacity = max(
            int(free_bytes * 0.15),
            self.flush_threshold_bytes * 2
        )

        # Never exceed user's original configuration
        target_capacity = min(target_capacity, self._original_max_size_bytes)

        current_capacity = self.max_size_bytes

        # ============================================================
        # DOWNSCALE: Memory pressure detected (with hysteresis)
        # ============================================================
        # Trigger if target is <70% of current capacity (30% hysteresis)
        if target_capacity < current_capacity * 0.7:
            new_capacity = max(target_capacity, self.flush_threshold_bytes * 2)

            logger.debug(
                f"Cache downscaling due to memory pressure: "
                f"{current_capacity/1e6:.1f}MB -> {new_capacity/1e6:.1f}MB "
                f"(free: {free_bytes/1e6:.1f}MB)"
            )

            with self._lock:
                self.max_size_bytes = new_capacity
                self._capacity_downscale_count += 1

        # ============================================================
        # UPSCALE: Memory available, recover capacity (with hysteresis)
        # ============================================================
        # Trigger if target is >150% of current capacity (50% hysteresis)
        elif target_capacity > current_capacity * 1.5:
            new_capacity = target_capacity

            logger.debug(
                f"Cache recovering capacity: "
                f"{current_capacity/1e6:.1f}MB -> {new_capacity/1e6:.1f}MB "
                f"(free: {free_bytes/1e6:.1f}MB)"
            )

            with self._lock:
                self.max_size_bytes = new_capacity
                self._capacity_upscale_count += 1

    def clear(self):
        """Clear cache and free GPU memory."""
        with self._lock:
            # Explicitly delete all tensors
            for value in self.cache.values():
                del value

            self.cache.clear()
            self.current_size = 0
            self._pending_flush_bytes = 0

        # Flush outside lock
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass

    def reset(self):
        """Alias for clear() for backward compatibility."""
        self.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including adaptive sizing info."""
        with self._lock:
            total = self.hits + self.misses
            stats = {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0,
                "size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "original_max_size_mb": self._original_max_size_bytes / (1024 * 1024),
                "entries": len(self.cache),
                "pending_flush_mb": self._pending_flush_bytes / (1024 * 1024),
            }

            if self.enable_adaptive_sizing:
                stats.update({
                    "adaptive_downscales": self._capacity_downscale_count,
                    "adaptive_upscales": self._capacity_upscale_count,
                })

            return stats
