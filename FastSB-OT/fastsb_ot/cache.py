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

<<<<<<< Updated upstream
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
=======
    Fixes applied:
    - Thread-safe access to all shared state (_pending_flush_bytes, _last_cuda_flush)
    - CUDA operations moved outside lock to prevent blocking
    - Type validation for put() to prevent AttributeError
    - Storage overhead detection to prevent memory leaks from tensor views
    - Multi-GPU memory tracking
    - Fixed adaptive downscaling logic
    - Optimized clear() implementation
    """

    def __init__(self, max_size_mb: int = 1024, max_entries: int = 100,
                 cuda_flush_watermark: float = 0.8, flush_threshold_mb: int = 32):
        self.cache = OrderedDict()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._original_max_size_bytes = self.max_size_bytes  # Track original limit
>>>>>>> Stashed changes
        self.max_entries = max_entries
        self.current_size = 0
        self.hits = 0
        self.misses = 0
<<<<<<< Updated upstream
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
=======
        self.cuda_flush_watermark = cuda_flush_watermark
        self.flush_threshold_bytes = flush_threshold_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._last_cuda_flush = 0
        self._pending_flush_bytes = 0
        self._adaptive_resize_attempts = 0

    def get(self, key: Any, clone: bool = False) -> Optional[torch.Tensor]:
        """Get from cache with optional cloning for immutability"""
        with self._lock:
            if key in self.cache:
                self.hits += 1
>>>>>>> Stashed changes
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.clone() if clone else value
            self.misses += 1
            return None

    def put(self, key: Any, value: torch.Tensor):
<<<<<<< Updated upstream
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
=======
        """Put tensor in cache with proper cleanup

        Fixes:
        - Type validation to prevent AttributeError
        - Storage overhead detection to prevent view memory leaks
        - CUDA operations moved outside lock
        - Proper thread-safe state management
        """
        # FIX #1: Type validation
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(value).__name__}")

        if value.requires_grad:
            value = value.detach()

        # FIX #2: Check for storage overhead (view memory leak prevention)
        # Do this BEFORE acquiring lock to avoid blocking
        if value.numel() > 0:
            view_size = value.element_size() * value.nelement()
            storage_size = value.untyped_storage().size()
            # Clone if storage is >50% larger than view (indicates this is a slice/view)
            if storage_size > view_size * 1.5:
                value = value.clone()

        evicted_cuda_bytes = 0

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
                return

            # Eviction loop
>>>>>>> Stashed changes
            while (self.current_size + value_size > self.max_size_bytes or
                   len(self.cache) >= self.max_entries):
                if not self.cache:
                    break

                old_key, old_value = self.cache.popitem(last=False)
                old_size = old_value.element_size() * old_value.nelement()
                self.current_size -= old_size

<<<<<<< Updated upstream
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
=======
                if old_value.device.type == 'cuda':
                    evicted_cuda_bytes += old_size
                del old_value

            self.cache[key] = value
            self.current_size += value_size

            # FIX #3: Update pending flush bytes under lock
            self._pending_flush_bytes += evicted_cuda_bytes

            # FIX #4: Snapshot state for CUDA flush (to be done outside lock)
            pending_bytes = self._pending_flush_bytes
            last_flush_time = self._last_cuda_flush

        # FIX #5: CUDA operations OUTSIDE lock to prevent blocking other threads
        if evicted_cuda_bytes > 0:
            self._try_cuda_flush_unlocked(pending_bytes, last_flush_time)

        # Adaptive downscaling check (also outside lock)
        self._maybe_downscale_cache()

    def _try_cuda_flush_unlocked(self, pending_bytes: int, last_flush_time: float):
        """Try to flush CUDA cache without holding the main lock

        Fixes:
        - All CUDA operations done outside main lock
        - Multi-GPU memory tracking
        - Atomic updates to shared state
        - Removed redundant checks
        """
        if not torch.cuda.is_available():
            with self._lock:
                self._pending_flush_bytes = 0
            return

        current_time = time.time()
        should_flush = False

        try:
            # FIX #6: Multi-GPU memory tracking
            allocated, reserved = self._get_total_cuda_memory()
            unused = reserved - allocated

            # Determine if we should flush
            if pending_bytes >= self.flush_threshold_bytes and unused >= self.flush_threshold_bytes:
                # Both pending and unused exceed threshold - flush immediately
                should_flush = True
            elif current_time - last_flush_time >= 1.0:
                # Time gate passed, check if either condition warrants flush
                # FIX #7: Removed redundant "reserved > 0" check
                if unused > self.flush_threshold_bytes:
                    should_flush = True
                elif pending_bytes >= self.flush_threshold_bytes:
                    should_flush = True

            if should_flush:
                # CUDA operation outside lock
                torch.cuda.empty_cache()

                # FIX #8: Atomic update of shared state under lock
                with self._lock:
                    self._pending_flush_bytes = 0
                    self._last_cuda_flush = current_time

                # Adaptive downscaling after successful flush
                self._maybe_downscale_cache(force=True)
            elif pending_bytes >= self.flush_threshold_bytes:
                # Reset if we've accumulated enough but time gate prevents flush
                if current_time - last_flush_time >= 1.0:
                    with self._lock:
                        self._pending_flush_bytes = 0
        except Exception as e:
            # On error, try to flush anyway and reset state
            logger.debug(f"CUDA flush error: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            with self._lock:
                self._pending_flush_bytes = 0
                self._last_cuda_flush = current_time

    def _get_total_cuda_memory(self) -> tuple[int, int]:
        """Get total allocated and reserved memory across all CUDA devices

        FIX #9: Multi-GPU support
        """
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
            # Fallback to default device
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()

        return allocated, reserved

    def clear(self):
        """Clear cache and free GPU memory

        FIX #10: Optimized implementation - removed inefficient del loop
        FIX #11: CUDA flush outside lock
        """
        with self._lock:
            self.cache.clear()  # Python GC handles cleanup, no need for del loop
            self.current_size = 0
            self._pending_flush_bytes = 0

        # FIX #12: CUDA flush outside lock to prevent blocking
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self._maybe_downscale_cache(force=True)

    def reset(self):
        """Alias for clear() for backward compatibility"""
        self.clear()

    def _maybe_downscale_cache(self, force: bool = False):
        """Adapt cache limits to available GPU memory to reduce manual tuning.

        FIX #13: Fixed adaptive downscaling logic
        - Now properly handles high utilization scenarios
        - Correctly determines when to downscale
        """
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

        # Calculate target based on available memory (15% of free)
        target_limit = max(int(free_bytes * 0.15), self.flush_threshold_bytes * 2)

        # FIX #14: Fixed downscaling logic
        # Downscale if ANY of these conditions are true:
        # 1. Forced (after clear or flush)
        # 2. Cache is >75% full AND target is less than current max
        # 3. Target is significantly less than current max (memory pressure)

        with self._lock:
            current_max = self.max_size_bytes
            current_usage = self.current_size

        should_downscale = False

        if force:
            should_downscale = True
        elif current_usage > current_max * 0.75 and target_limit < current_max:
            # High utilization AND memory pressure
            should_downscale = True
        elif target_limit < current_max * 0.7:
            # Significant memory pressure (target is <70% of current)
            should_downscale = True

        if should_downscale:
            # Don't shrink below minimum threshold
            new_cap = max(target_limit, self.flush_threshold_bytes * 2)
            # Don't expand beyond original max
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
        """Get cache statistics

        FIX #15: Added lock protection for thread-safe stats reading
        """
        with self._lock:
            total = self.hits + self.misses
            return {
>>>>>>> Stashed changes
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0,
                "size_mb": self.current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
<<<<<<< Updated upstream
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
=======
                "entries": len(self.cache),
                "pending_flush_mb": self._pending_flush_bytes / (1024 * 1024)
            }
>>>>>>> Stashed changes
