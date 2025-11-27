"""Kernel utilities for FastSB-OT."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common
from .cache import MemoryEfficientCacheFixed
from .config import FastSBOTConfig

compile_function_fixed = common.compile_function_fixed
TRITON_AVAILABLE = common.TRITON_AVAILABLE
launch_triton_kernel_safe = getattr(common, "launch_triton_kernel_safe", None)
fused_drift_transport_kernel_fixed = getattr(common, "fused_drift_transport_kernel_fixed", None)
fisher_diagonal_kernel_fixed = getattr(common, "fisher_diagonal_kernel_fixed", None)
check_triton_availability = common.check_triton_availability
get_optimal_block_size = common.get_optimal_block_size

__all__ = ["KernelModule"]


class KernelModule(nn.Module):
    """Optimized kernel operations module with Fisher geometry support"""

    def __init__(self, config: FastSBOTConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.kernel_cache = MemoryEfficientCacheFixed(
            config.cache_size_mb // 4,
            config.max_cache_entries // 4,
            config.cuda_cache_flush_watermark,
            config.cuda_cache_flush_threshold_mb
        )
        # FIX: Use OrderedDict for proper LRU eviction instead of static dict
        self.freq_weights_cache = OrderedDict()
        self.freq_weights_lock = threading.Lock()
        self.fisher_cache = MemoryEfficientCacheFixed(
            config.cache_size_mb // 8,
            config.max_cache_entries // 8,
            config.cuda_cache_flush_watermark,
            config.cuda_cache_flush_threshold_mb
        )
        self._freq_grid_cache = OrderedDict()

        self._setup_buffers()

    def _setup_buffers(self):
        """Pre-allocate commonly used buffers"""
        gaussian_1d = torch.tensor([1, 2, 1], dtype=torch.float32) / 4
        gaussian_kernel = gaussian_1d[:, None] @ gaussian_1d[None, :]
        self.register_buffer('gaussian_kernel', gaussian_kernel.view(1, 1, 3, 3), persistent=False)

    def compute_gaussian_kernel_fft(self, shape: Tuple[int, ...], sigma: float, device: torch.device) -> torch.Tensor:
        """Cache frequency grids with LRU, device-agnostic"""
        sigma = max(1e-3, round(float(sigma), 4))

        cache_key = (*shape, sigma, str(device))

        cached = self.kernel_cache.get(cache_key, clone=False)
        if cached is not None:
            # FIX: Verify cached tensor is on correct device
            if cached.device == device:
                return cached.contiguous()
            else:
                # Device mismatch (rare, but possible with multi-GPU)
                # Remove invalid entry and recompute
                pass

        grid_key = tuple(shape)
        grids = self._freq_grid_cache.get(grid_key)

        if grids is not None:
            self._freq_grid_cache.move_to_end(grid_key)
        else:
            if len(self._freq_grid_cache) >= 64:
                self._freq_grid_cache.popitem(last=False)

            n_dims = len(shape)
            grids = []
            for i, size in enumerate(shape):
                if i == n_dims - 1:
                    try:
                        freq = torch.fft.rfftfreq(size, device=torch.device("cpu"))
                    except TypeError:
                        freq = torch.fft.rfftfreq(size).to("cpu")
                else:
                    try:
                        freq = torch.fft.fftfreq(size, device=torch.device("cpu"))
                    except TypeError:
                        freq = torch.fft.fftfreq(size).to("cpu")

                freq_shape = [1] * len(shape)
                if i == n_dims - 1:
                    freq_shape[i] = size // 2 + 1
                else:
                    freq_shape[i] = size
                grids.append(freq.reshape(freq_shape))

            self._freq_grid_cache[grid_key] = tuple(grids)

        grids = tuple(g.to(device) for g in self._freq_grid_cache[grid_key])

        dist_sq = sum(g**2 for g in grids)

        kernel_fft = torch.exp(-2 * (math.pi * sigma)**2 * dist_sq)
        kernel_fft = torch.maximum(kernel_fft, kernel_fft.new_tensor(1e-6))

        self.kernel_cache.put(cache_key, kernel_fft)

        return kernel_fft

    def get_frequency_weights(self, shape: Tuple[int, ...], device_str: str) -> torch.Tensor:
        """Get frequency weights for importance weighting with proper LRU cache.

        FIX: Proper LRU eviction instead of "fill once and stop caching" behavior.
        """
        cache_key = (shape, device_str)

        with self.freq_weights_lock:
            cached = self.freq_weights_cache.get(cache_key)
            if cached is not None:
                # LRU: Move to end to mark as recently used
                self.freq_weights_cache.move_to_end(cache_key)
                return cached.to(torch.device(device_str))

            # Compute new weights
            weights = self._compute_frequency_weights_fixed(shape, device_str)

            # LRU eviction: Remove oldest entry if cache is full
            if len(self.freq_weights_cache) >= 32:
                self.freq_weights_cache.popitem(last=False)  # Evict least recently used

            # Store on CPU to save GPU memory
            self.freq_weights_cache[cache_key] = weights.cpu()

        return weights.to(torch.device(device_str))

    def _compute_frequency_weights_fixed(self, shape: Tuple[int, ...], device_str: str) -> torch.Tensor:
        """Backward compatible frequency weights with proper Nyquist"""
        device = torch.device("cpu")

        coords = []
        for i, size in enumerate(shape):
            if i == len(shape) - 1:
                try:
                    freq = torch.fft.rfftfreq(size, device=device).abs()
                except TypeError:
                    freq = torch.fft.rfftfreq(size).to(device).abs()
            else:
                try:
                    freq = torch.fft.fftfreq(size, device=device).abs()
                except TypeError:
                    freq = torch.fft.fftfreq(size).to(device).abs()

            freq_shape = [1] * len(shape)
            if i == len(shape) - 1:
                freq_shape[i] = size // 2 + 1
            else:
                freq_shape[i] = size
            coords.append(freq.reshape(freq_shape))

        f_mag = torch.sqrt(sum(c**2 for c in coords)) / math.sqrt(len(coords))

        weights = 4 * f_mag * (1 - f_mag)
        weights = torch.clamp(weights, min=1e-2)
        weights = weights ** 2

        return weights

    @compile_function_fixed(dynamic=True, use_global_cache=True)
    @torch.no_grad()
    def estimate_fisher_diagonal(self, x: torch.Tensor, score: torch.Tensor, t: float, alpha: float = None) -> torch.Tensor:
        """Estimate diagonal Fisher information matrix with proper precision handling

        CRITICAL FIX: Removed data_ptr() based caching.
        The cost of computing Fisher (abs + optional conv2d) is negligible compared to:
        1. Hash computation overhead
        2. Risk of cache collision from memory address reuse
        3. Risk of stale data from pointer aliasing

        Fisher estimation is O(N) element-wise operation, not worth dangerous caching.

        Args:
            x: input
            score: current score estimate
            t: a scalar used only for logging/debugging (not used in computation)
            alpha: (alpha_bar) value for adaptive epsilon calculation
        """
        # REMOVED: Dangerous data_ptr() based caching
        # The risk of collision (same address, similar checksum) is too high
        # Cost: ~1% overhead from abs() + optional 3x3 conv2d
        # Benefit of removal: Guaranteed correct Fisher info, no silent corruption

        original_dtype = score.dtype
        if self.config.use_fp32_fisher and score.dtype in [torch.float16, torch.bfloat16]:
            score_fp32 = score.float()
        else:
            score_fp32 = score

        # Use Triton kernel if available and beneficial
        if TRITON_AVAILABLE and self.config.use_triton_kernels and x.is_cuda and x.numel() > 1e6:
            fisher = torch.empty_like(score_fp32)

            score_flat = score_fp32.reshape(-1)
            fisher_flat = fisher.reshape(-1)

            n_elements = score_flat.numel()
            # default alpha if not provided: assume 1t (only as fallback)
            alpha_val = float(alpha if alpha is not None else max(0.0, min(1.0, 1.0 - float(t))))
            launch_triton_kernel_safe(
                fisher_diagonal_kernel_fixed,
                score_flat, fisher_flat,
                n_elements=n_elements,
                kernel_type="fisher",
                alpha=alpha_val
            )

            fisher = fisher_flat.reshape(score.shape)
        else:
            # default alpha if not provided: assume 1-t (only as fallback)
            alpha_val = float(alpha if alpha is not None else max(0.0, min(1.0, 1.0 - float(t))))
            adaptive_eps = 1e-4 + 1e-3 * (1.0 - alpha_val)
            fisher = torch.abs(score_fp32) + adaptive_eps

            # FIX: Validate spatial dimensions before applying 2D convolution
            if x.dim() == 4:
                B, C, H, W = fisher.shape

                # Only apply gaussian smoothing if spatial dimensions are large enough
                if H >= 3 and W >= 3:
                    fisher = fisher.reshape(B * C, 1, H, W)

                    if hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
                        try:
                            if x.is_contiguous(memory_format=torch.channels_last):
                                fisher = fisher.contiguous(memory_format=torch.channels_last)
                        except (TypeError, AttributeError):
                            pass

                    # FIX: Ensure gaussian kernel is on correct device
                    kernel = self.gaussian_kernel.to(device=fisher.device, dtype=fisher.dtype)
                    kernel_size = kernel.shape[-1]
                    padding = kernel_size // 2

                    fisher = F.conv2d(fisher, kernel, padding=padding)
                    fisher = fisher.reshape(B, C, H, W)
                else:
                    # Skip smoothing for small spatial dimensions
                    pass

        if original_dtype == torch.float16:
            fisher = fisher.clamp(max=65504)
        elif original_dtype == torch.bfloat16:
            fisher = fisher.clamp(max=3.38e38)

        if fisher.dtype != original_dtype:
            fisher = fisher.to(original_dtype)

        # REMOVED: Caching - computation is cheap, caching is dangerous
        # self.fisher_cache.put(cache_key, fisher)

        return fisher



