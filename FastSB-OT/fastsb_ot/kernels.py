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
        self.freq_weights_cache = {}
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
            return cached.contiguous()

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
        """Get frequency weights for importance weighting with thread safety"""
        cache_key = (shape, device_str)

        with self.freq_weights_lock:
            cached = self.freq_weights_cache.get(cache_key)
            if cached is not None:
                return cached.to(torch.device(device_str))

            weights = self._compute_frequency_weights_fixed(shape, device_str)

            if len(self.freq_weights_cache) < 32:
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

        Args:
            x: input
            score: current score estimate
            t: a scalar used only for cache bucketing (can be 0.0 if alpha is provided)
            alpha: (t) value; if provided it is used both in computation and cache key
        """
        # Add a tiny content fingerprint to reduce pointer-reuse collisions
        fp = score.reshape(-1)
        if fp.numel() >= 4:
            sample = torch.stack([fp[0], fp[fp.numel()//3], fp[2*fp.numel()//3], fp[-1]]).float()
            # cheap, device-local checksum
            chk = float(sample.sum().item())
        else:
            chk = float(fp.float().sum().item())

        # Include stride in cache key for extra safety
        stride_sig = tuple(score.stride()) if score.is_contiguous() else "non_contig"
        # Include  in the cache key to avoid reusing across very different noise levels
        t_part = f"{t:.6f}"
        a_part = f"{alpha:.6f}" if alpha is not None else "na"
        cache_key = (x.shape, f"{t_part}|ab:{a_part}", str(x.device), str(x.dtype), int(score.data_ptr()),
                     f"{chk:.6e}", stride_sig)

        cached = self.fisher_cache.get(cache_key, clone=False)
        if cached is not None:
            return cached

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
            # default alpha if not provided: assume 1t (only as fallback)
            alpha_val = float(alpha if alpha is not None else max(0.0, min(1.0, 1.0 - float(t))))
            adaptive_eps = 1e-4 + 1e-3 * (1.0 - alpha_val)
            fisher = torch.abs(score_fp32) + adaptive_eps

            if x.dim() == 4:
                B, C = fisher.shape[:2]
                fisher = fisher.reshape(B * C, 1, *fisher.shape[2:])

                if hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
                    try:
                        if x.is_contiguous(memory_format=torch.channels_last):
                            fisher = fisher.contiguous(memory_format=torch.channels_last)
                    except (TypeError, AttributeError):
                        pass

                kernel_size = self.gaussian_kernel.shape[-1]
                padding = kernel_size // 2

                fisher = F.conv2d(fisher, self.gaussian_kernel, padding=padding)
                fisher = fisher.reshape(B, C, *fisher.shape[2:])

        if original_dtype == torch.float16:
            fisher = fisher.clamp(max=65504)
        elif original_dtype == torch.bfloat16:
            fisher = fisher.clamp(max=3.38e38)

        if fisher.dtype != original_dtype:
            fisher = fisher.to(original_dtype)

        # CRITICAL FIX: Ensure Fisher information remains strictly positive
        # after dtype conversion and clamping (required for transport stability)
        fisher = torch.clamp(fisher, min=1e-6)

        self.fisher_cache.put(cache_key, fisher)

        return fisher



