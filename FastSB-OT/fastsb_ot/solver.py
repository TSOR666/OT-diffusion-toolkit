"""Solver entry points for FastSB-OT."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common
from .cache import MemoryEfficientCacheFixed
from .config import FastSBOTConfig
from .kernels import KernelModule
from .transport import HierarchicalBridge, MomentumTransport, TransportModule

logger = common.logger
_randn_like_compat = common._randn_like_compat
compile_function_fixed = common.compile_function_fixed
TRITON_AVAILABLE = common.TRITON_AVAILABLE
check_triton_availability = common.check_triton_availability
log_sum_exp_stabilized = common.log_sum_exp_stabilized

__all__ = ["FastSBOTSolver", "make_schedule", "example_usage"]


class FastSBOTSolver(nn.Module):
    """Production-ready Fast Schrodinger Bridge with Optimal Transport Solver.

    Args:
        score_model: Neural network that predicts the score ``?_x log p_t(x)``. If
        noise_schedule: Function mapping t[0,1]  (t) (cumulative product of (1-)).
                       Should monotonically decrease from (0)1 (clean) to (1)0 (noise).
                       NOT (t) or (t), but (t) = (1-_i).
                       The sampling process goes from t=1 to t=0 (denoising direction).
        config: FastSBOTConfig with solver settings
        device: Torch device for computation
        persistent_cache: Optional dict of caches to reuse across solver instances

    Note: The noise_schedule must return (t), not (t). Following DDPM convention:
    - (0)  1 means clean data (no noise)
    - (1)  0 means pure noise
    - Sampling proceeds from t=1 (noisy) to t=0 (clean)

    For CFG compatibility: The model can be wrapped to provide conditional/unconditional
    scores. Pass the difference through guidance_scale for true CFG.
    """

    def __init__(
        self,
        score_model: nn.Module,
        noise_schedule: Callable,
        config: Optional[FastSBOTConfig] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        persistent_cache: Optional[Dict[str, MemoryEfficientCacheFixed]] = None
    ):
        """Initialize the FastSB-OT Solver with production polish and enhanced sampling."""
        super().__init__()

        self.score_model = score_model
        self.noise_schedule = noise_schedule
        self.device = device
        self.config = config or FastSBOTConfig()

        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Detect and log MPS/ROCm for better user experience
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) detected: Some CUDA-specific optimizations will be disabled. "
                        "Performance may vary compared to CUDA GPUs.")
        elif hasattr(torch, 'hip') or 'rocm' in torch.__version__.lower():
            logger.info("ROCm detected: Using AMD GPU support. Some CUDA-specific features may behave differently.")

        if self.config.use_bfloat16 and device.type == 'cuda':
            self.amp_dtype = torch.bfloat16
        elif self.config.use_mixed_precision and device.type == 'cuda':
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        # Ensure the configured RNG generator lives on the active device (for determinism)
        self._ensure_device_generator()

        self.kernel_module = KernelModule(self.config, device)
        self.transport_module = TransportModule(self.config, device)

        if self.config.use_momentum_transport:
            self.momentum_transport = MomentumTransport(self.config.momentum_beta, device)

        if self.config.use_hierarchical_bridge:
            self.hierarchical_bridge = HierarchicalBridge(self.config.hierarchical_scales, device)

        self.to(device)
        if hasattr(score_model, 'to'):
            self.score_model.to(device)
            self.score_model.eval()

            # Only use channels_last on Ampere+ GPUs
            # Note: .to(memory_format=...) is best-effort and non-fatal if unsupported
            if hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
                if any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in self.score_model.modules()):
                    try:
                        self.score_model.to(memory_format=torch.channels_last)
                    except TypeError:
                        pass  # Older PyTorch versions may not support this

        had_output_attr = hasattr(self.score_model, 'predicts_score') or hasattr(self.score_model, 'predicts_noise')
        self._model_outputs_score = self._infer_model_outputs_score(self.score_model)
        self.score_model.predicts_score = self._model_outputs_score
        self.score_model.predicts_noise = not self._model_outputs_score
        if not had_output_attr:
            logger.debug(
                "Assuming score_model outputs scores. If it predicts noise (epsilon), wrap it with "
                "NoisePredictorToScoreWrapper or set predicts_noise=True."
            )

        if persistent_cache is not None:
            self.score_cache = persistent_cache.get('score_cache', MemoryEfficientCacheFixed(
                self.config.cache_size_mb // 2,
                self.config.max_cache_entries // 2,
                self.config.cuda_cache_flush_watermark,
                self.config.cuda_cache_flush_threshold_mb
            ))
            self.kernel_module.kernel_cache = persistent_cache.get('kernel_cache', self.kernel_module.kernel_cache)
            self.kernel_module.fisher_cache = persistent_cache.get('fisher_cache', self.kernel_module.fisher_cache)
        else:
            self.score_cache = MemoryEfficientCacheFixed(
                self.config.cache_size_mb // 2,
                self.config.max_cache_entries // 2,
                self.config.cuda_cache_flush_watermark,
                self.config.cuda_cache_flush_threshold_mb
            )

        self.noise_schedule_cache = {}

        # Validate noise schedule monotonicity
        self._validate_noise_schedule()

        # Cache for patch overlap masks
        self._overlap_cache = OrderedDict()
        self._overlap_cache_cap = 32  # tiny LRU

        self.perf_stats = {
            'times_per_step': [],
            'memory_per_step': [],
            'score_cache_stats': {},
            'kernel_cache_stats': {},
            'fisher_cache_stats': {},
        }

        self._compile_step_function()

        if self.config.warmup and device.type == 'cuda':
            self._warmup_cuda()

    def _ensure_device_generator(self):
        """Ensure that the configured RNG generator is on the active device for determinism."""
        gen = getattr(self.config, 'generator', None)
        if gen is None:
            return
        # Try to read device; not all versions expose .device
        gen_device = None
        try:
            gen_device = gen.device
        except Exception:
            gen_device = torch.device('cpu')
        if str(gen_device) != str(self.device):
            try:
                new_gen = torch.Generator(device=self.device)
            except TypeError:
                new_gen = torch.Generator()
            # Re-seed deterministically if a seed was provided
            if self.config.seed is not None:
                try:
                    new_gen.manual_seed(self.config.seed)
                except Exception:
                    pass
            else:
                try:
                    new_gen.seed()
                except Exception:
                    pass
            self.config.generator = new_gen

    def _infer_model_outputs_score(self, model: nn.Module) -> bool:
        """Determine whether the wrapped model returns scores or noise."""
        if hasattr(model, 'predicts_noise'):
            return not bool(getattr(model, 'predicts_noise'))
        if hasattr(model, 'predicts_score'):
            return bool(getattr(model, 'predicts_score'))
        return True

    def _validate_noise_schedule(self):
        """Validate that noise_schedule returns monotonically decreasing _bar(t)"""
        # Use more probe points to catch micro-wiggles
        probe_points = [i / 32.0 for i in range(33)]  # 33 points from 0 to 1
        vals = [float(self.noise_schedule(t)) for t in probe_points]

        # Check monotonic decrease (_bar(0)  _bar(1/32)  ...  _bar(1))
        non_monotonic = []
        for i in range(len(vals)-1):
            if vals[i] < vals[i+1] - 1e-9:  # Small tolerance for float precision
                non_monotonic.append((probe_points[i], vals[i], probe_points[i+1], vals[i+1]))

        if non_monotonic:
            raise ValueError(
                f"noise_schedule must return _bar(t) monotonically decreasing in t[0,1]. "
                f"Found non-monotonic segments: {non_monotonic[:3]}... "
                f"_bar should decrease from ~1 at t=0 (clean) to ~0 at t=1 (noise)."
            )

        # Also check reasonable range
        if not (0.8 <= vals[0] <= 1.0):
            logger.warning(f"_bar(0) = {vals[0]:.4f} is unusually low. Expected ~1.0 for clean data.")
        if not (0.0 <= vals[-1] <= 0.2):
            logger.warning(f"_bar(1) = {vals[-1]:.4f} is unusually high. Expected ~0.0 for pure noise.")

    def _get_overlap_mask(self, H_pad: int, W_pad: int, patch_size: int, stride: int, device: torch.device) -> torch.Tensor:
        """Cache overlap masks for patch processing efficiency"""
        key = (H_pad, W_pad, patch_size, stride)
        mask = self._overlap_cache.get(key)

        if mask is None:
            # Build entire mask on CPU to avoid GPU memory spike
            device_cpu = torch.device("cpu")

            # Compute overlap mask
            n_patches_h = (H_pad - patch_size) // stride + 1
            n_patches_w = (W_pad - patch_size) // stride + 1
            n_patches = n_patches_h * n_patches_w

            ones = torch.ones(
                (1, patch_size * patch_size, n_patches),
                device=device_cpu,  # Build on CPU
                dtype=torch.float32
            )
            mask = F.fold(
                ones,
                output_size=(H_pad, W_pad),
                kernel_size=patch_size,
                stride=stride
            )  # Shape: (1, 1, H_pad, W_pad)

            # LRU eviction
            if len(self._overlap_cache) >= self._overlap_cache_cap:
                self._overlap_cache.popitem(last=False)

            # Store on CPU to save GPU memory
            self._overlap_cache[key] = mask  # Already on CPU

        return mask.to(device)

    def _amp_ctx(self):
        """Get appropriate autocast context for device type with safety"""
        if self.device.type == "cuda" and AUTOCAST_AVAILABLE:
            return autocast("cuda", dtype=self.amp_dtype, enabled=self.config.use_mixed_precision)
        elif self.device.type == "cpu" and AUTOCAST_AVAILABLE:
            # CPU autocast requires PyTorch 1.11+ for bfloat16 support
            allow = self.config.use_mixed_precision and Version(torch.__version__) >= Version("1.11")
            if self.config.use_mixed_precision and not allow and not hasattr(self, '_cpu_autocast_warned'):
                logger.info("CPU autocast disabled: PyTorch < 1.11 doesn't support bfloat16 autocast on CPU. "
                            "Upgrade to PyTorch 1.11+ for CPU mixed precision support.")
                self._cpu_autocast_warned = True
            return autocast("cpu", dtype=torch.bfloat16, enabled=allow)
        return nullcontext()

    def _compile_step_function(self):
        """Compile step function that's actually used."""
        @compile_function_fixed(
            mode=self.config.compile_mode,
            dynamic=self.config.use_dynamic_compilation,
            max_cache_size=self.config.max_compiled_shapes,
            max_cache_size_mb=self.config.global_cache_size_mb,
            use_global_cache=self.config.global_compile_cache,
            enable_cpu_compile=self.config.enable_cpu_compile,
            compile_timeout=self.config.compile_timeout
        )
        def _sample_step_compiled(x, score, alpha_bar_t, dt, kernel_fft=None, freq_weights=None):
            """Actually used fused sampling step - fully tensor-native

            POLISH: Renamed alpha_t  alpha_bar_t for clarity
            """
            drift = -0.5 * (1 - alpha_bar_t) * score * dt
            drift = drift.to(x.dtype)

            if x.ndim == 4 and x.shape[2] >= 64 and kernel_fft is not None:
                x_next = self.fft_grid_transport_inline(x, drift, alpha_bar_t, kernel_fft, freq_weights)
            else:
                x_next = self.compute_drift_and_transport_inline(x, drift, alpha_bar_t)

            return x_next

        self._sample_step = _sample_step_compiled

    def _warmup_cuda(self):
        """Warmup CUDA kernels with gated FP32 time"""
        if self.device.type != 'cuda':
            return

        size = (2, 3, 64, 64)
        x = torch.randn(size, device=self.device, dtype=self.amp_dtype)

        # Only use channels_last on Ampere+
        if hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
            if any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in self.score_model.modules()):
                try:
                    x = x.to(memory_format=torch.channels_last)
                except TypeError:
                    pass

        # Handle discrete vs continuous timesteps for warmup
        if self.config.discrete_timesteps:
            # Use mid-step integer index for warmup
            idx = self.config.num_timesteps // 2
            t = torch.tensor([idx, idx], device=self.device, dtype=torch.long)
        else:
            dtype_t = torch.float32 if self.config.use_fp32_time else self.amp_dtype
            t = torch.tensor([0.5, 0.5], device=self.device, dtype=dtype_t)

        with torch.no_grad():
            with self._amp_ctx():
                _ = self.score_model(x, t)

        if self.config.use_momentum_transport:
            self.momentum_transport.reset_velocity()

        if os.environ.get('FASTSBOT_WARMUP_SYNC', '0') == '1':
            torch.cuda.synchronize()

    def _get_cached_noise_schedule(self, t: float) -> float:
        """Cache noise schedule values with 6-decimal precision"""
        t_key = round(float(t), 6)
        if t_key not in self.noise_schedule_cache:
            target_size = 800
            while len(self.noise_schedule_cache) > 1000:
                self.noise_schedule_cache.pop(next(iter(self.noise_schedule_cache)))
                if len(self.noise_schedule_cache) <= target_size:
                    break

            self.noise_schedule_cache[t_key] = float(self.noise_schedule(t_key))

        return self.noise_schedule_cache[t_key]

    def _normalize_cache_key(self, key_base: str, shape: Tuple[int, ...],
                            device: torch.device, dtype: torch.dtype) -> str:
        """Exact cache keys for score correctness"""
        shape_str = 'x'.join(map(str, shape))
        return f"{key_base}_shape_{shape_str}_{str(device)}_{dtype}"

    def _convert_model_output_to_score(self, model_output: torch.Tensor, alpha_bar_t: float) -> torch.Tensor:
        """Convert raw model output to a score tensor when necessary."""
        if self._model_outputs_score:
            return model_output

        sigma_t = math.sqrt(max(1e-8, 1.0 - float(alpha_bar_t)))
        return -model_output / sigma_t

    @torch.no_grad()
    def compute_score_cached(self, x: torch.Tensor, t: float, cache_key: Optional[str] = None) -> torch.Tensor:
        """Compute score with improved noise-to-score conversion for enhanced sampling

        POLISH: Suppresses strict mode warning when called from sample_improved
        """
        # POLISH: Smarter strict mode warning - don't warn from sample_improved
        if cache_key is None:
            if (os.environ.get("FASTSBOT_STRICT", "0") == "1" and
                os.environ.get("FASTSBOT_FORCE_SCORE_CACHE", "0") == "1"):
                logger.warning("compute_score_cached called without cache_key - score will be recomputed. "
                               "Provide a unique cache_key for better performance.")

        if cache_key is not None:
            if isinstance(t, torch.Tensor):
                t_val = float(t.item())
            else:
                t_val = float(t)
            t_key = f"{t_val:.6f}"
            cache_key = self._normalize_cache_key(
                f"{cache_key}_t{t_key}", x.shape, x.device, x.dtype
            )

            clone_cache = os.environ.get("FASTSBOT_CLONE_CACHE", "1") == "1"  # Default to True for safety
            cached = self.score_cache.get(cache_key, clone=clone_cache)
            if cached is not None:
                if cached.shape != x.shape:
                    cached = None
                else:
                    if x.dim() == 4 and hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
                        try:
                            cached = cached.contiguous(memory_format=torch.channels_last)
                        except TypeError:
                            cached = cached.contiguous()
                    return cached

        if x.dim() == 4 and hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
            try:
                if not x.is_contiguous(memory_format=torch.channels_last):
                    x = x.contiguous(memory_format=torch.channels_last)
            except (TypeError, AttributeError):
                if not x.is_contiguous():
                    x = x.contiguous()

        # Improved timestep handling
        dtype_t = torch.float32 if self.config.use_fp32_time else x.dtype

        # Handle discrete vs continuous timesteps
        if self.config.discrete_timesteps:
            # Model expects discrete timesteps [0, num_timesteps-1]
            idx = int(round(float(t) * (self.config.num_timesteps - 1)))
            idx = max(0, min(self.config.num_timesteps - 1, idx))
            t_tensor = torch.full((x.shape[0],), idx, device=x.device, dtype=torch.long)
        else:
            # Model expects normalized timesteps [0, 1]
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=dtype_t)

        with self._amp_ctx():
            model_output = self.score_model(x, t_tensor)

        alpha_bar_t = self._get_cached_noise_schedule(t)

        score = self._convert_model_output_to_score(model_output, alpha_bar_t)

        # Optional: Apply score correction for better quality
        if self.config.use_score_correction and alpha_bar_t < 0.9:
            correction = self.compute_score_correction(x, score, t, alpha_bar_t)
            score = score + correction

        if cache_key is not None:
            if x.dim() == 4 and hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
                try:
                    score = score.contiguous(memory_format=torch.channels_last)
                except TypeError:
                    pass
            self.score_cache.put(cache_key, score)

        return score

    def compute_score_correction(self, x: torch.Tensor, score: torch.Tensor, t: float, alpha_bar_t: Optional[float] = None) -> torch.Tensor:
        """Apply score correction for improved sampling quality

        POLISH: Using alpha_bar_t for clarity
        """
        with torch.no_grad():
            # Estimate local curvature for adaptive correction
            eps = 1e-4
            gen = getattr(self.config, 'generator', None)
            noise = _randn_like_compat(x, gen) * eps
            x_perturbed = x + noise

            # Get perturbed score (without correction to avoid recursion)
            original_correction = self.config.use_score_correction
            self.config.use_score_correction = False
            score_perturbed = self.compute_score_cached(x_perturbed, t)
            self.config.use_score_correction = original_correction

            # Estimate curvature
            curvature = (score_perturbed - score) / (eps + 1e-8)

            # Apply adaptive correction (stronger at later timesteps)
            if alpha_bar_t is None:
                alpha_bar_t = self._get_cached_noise_schedule(t)
            correction_strength = 0.1 * (1.0 - alpha_bar_t)
            correction = -correction_strength * curvature

        return correction

    def compute_score_patches(self, x: torch.Tensor, t: float, run_uid: Optional[int] = None) -> torch.Tensor:
        """Clean API name for patch-based score computation"""
        return self.compute_score_patches_fixed(x, t, run_uid)

    def compute_score_patches_fixed(self, x: torch.Tensor, t: float, run_uid: Optional[int] = None) -> torch.Tensor:
        """Patch processing with TIME in cache key"""
        B, C, H, W = x.shape

        if H * W < 256 * 256:
            if isinstance(t, torch.Tensor):
                t_val = float(t.item())
            else:
                t_val = float(t)
            t_key = f"{t_val:.6f}"

            # Add fingerprint to avoid pointer aliasing
            fp = x.reshape(-1)
            if fp.numel() >= 4:
                sample = torch.stack([fp[0], fp[fp.numel()//3], fp[2*fp.numel()//3], fp[-1]]).float()
                chk = float(sample.sum().item())
            else:
                chk = float(fp.float().sum().item())

            if run_uid is not None:
                cache_key = f"run{run_uid}_t{t_key}_small_ptr{int(x.data_ptr())}_chk{chk:.6e}"
            else:
                cache_key = f"t{t_key}_small_ptr{int(x.data_ptr())}_chk{chk:.6e}"
            return self.compute_score_cached(x, t, cache_key=cache_key)

        if isinstance(t, torch.Tensor):
            t_val = float(t.item())
        else:
            t_val = float(t)
        t_key = f"{t_val:.6f}"
        if run_uid is not None:
            input_cache_key = f"patches_run{run_uid}_t{t_key}_input_ptr{int(x.data_ptr())}"
        else:
            input_cache_key = f"patches_t{t_key}_input_ptr{int(x.data_ptr())}"

        input_cache_key = self._normalize_cache_key(
            input_cache_key, x.shape, x.device, x.dtype
        )

        cached = self.score_cache.get(input_cache_key, clone=False)
        if cached is not None:
            if cached.shape == x.shape:
                return cached

        if self.config.adaptive_patch_size and self.device.type == 'cuda':
            if hasattr(torch.cuda, 'mem_get_info'):
                free_memory = torch.cuda.mem_get_info()[0]
                bytes_per_element = 4 if self.amp_dtype == torch.float32 else 2
                patch_memory = B * C * self.config.max_patch_size**2 * bytes_per_element * 4

                if patch_memory > free_memory * 0.5:
                    patch_size = int(math.sqrt(free_memory * 0.5 / (B * C * bytes_per_element * 4)))
                    patch_size = max(32, min(patch_size, self.config.max_patch_size))
                else:
                    patch_size = self.config.max_patch_size
            else:
                patch_size = self.config.max_patch_size
        else:
            patch_size = min(self.config.max_patch_size, max(H, W) // 2)

        # Apply training patch size override if specified
        if self.config.training_max_patch_size is not None:
            patch_size = min(patch_size, int(self.config.training_max_patch_size))

        patch_size = max(32, patch_size)

        # Ensure patch size doesn't exceed image dimensions
        patch_size = min(patch_size, H, W)

        overlap = int(patch_size * self.config.patch_overlap_ratio)
        stride = max(1, patch_size - overlap)  # Ensure stride is at least 1

        pad_h = (stride - (H - patch_size) % stride) % stride
        pad_w = (stride - (W - patch_size) % stride) % stride

        # Handle reflect padding edge cases
        # Reflect mode requires input size >= 2 and kernel size >= 3
        if min(H, W) >= 2 and patch_size >= 3:
            pad_mode = 'reflect'
        else:
            pad_mode = 'replicate'

        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        H_pad, W_pad = x_padded.shape[2:]

        patches = F.unfold(x_padded, kernel_size=patch_size, stride=stride, padding=0)
        n_patches_h = (H_pad - patch_size) // stride + 1
        n_patches_w = (W_pad - patch_size) // stride + 1
        n_patches = n_patches_h * n_patches_w

        patches = patches.transpose(1, 2).reshape(B * n_patches, C, patch_size, patch_size)

        if hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
            try:
                patches = patches.contiguous(memory_format=torch.channels_last)
            except TypeError:
                patches = patches.contiguous()

        dtype_t = torch.float32 if self.config.use_fp32_time else patches.dtype

        # Handle timestep format
        if self.config.discrete_timesteps:
            idx = int(round(float(t) * (self.config.num_timesteps - 1)))
            idx = max(0, min(self.config.num_timesteps - 1, idx))
            t_tensor = torch.full((patches.shape[0],), idx, device=x.device, dtype=torch.long)
        else:
            t_tensor = torch.full((patches.shape[0],), float(t), device=x.device, dtype=dtype_t)

        with torch.no_grad():
            with self._amp_ctx():
                patch_output = self.score_model(patches, t_tensor)

        # POLISH: Using clearer variable names
        alpha_bar_t = self._get_cached_noise_schedule(t)
        patch_scores = self._convert_model_output_to_score(patch_output, alpha_bar_t)

        patch_scores = patch_scores.reshape(B, n_patches, C * patch_size * patch_size).transpose(1, 2)

        scores_padded = F.fold(
            patch_scores,
            output_size=(H_pad, W_pad),
            kernel_size=patch_size,
            stride=stride
        )

        # Use cached overlap mask instead of recomputing
        overlap_count = self._get_overlap_mask(H_pad, W_pad, patch_size, stride, x.device)
        # Shape: (1, 1, H_pad, W_pad), broadcast over B
        # Convert to FP32 for safe division
        orig_dtype = scores_padded.dtype
        eps = torch.finfo(torch.float32).eps * 16
        scores_padded = scores_padded.float() / (overlap_count.float() + eps)
        scores_padded = scores_padded.to(orig_dtype)

        scores = scores_padded[:, :, :H, :W]

        self.score_cache.put(input_cache_key, scores)

        return scores

    # ENHANCED SAMPLING METHODS

    def dynamic_threshold(self, x0_pred: torch.Tensor, percentile: Optional[float] = None) -> torch.Tensor:
        """Dynamic thresholding for stable sampling

        POLISH: Added adaptive floor option
        """
        if percentile is None:
            percentile = self.config.dynamic_thresholding_percentile

        # Cast to FP32 for stable quantile computation
        batch_size = x0_pred.shape[0]
        original_dtype = x0_pred.dtype
        x0_pred_flat = x0_pred.float().reshape(batch_size, -1)

        # Compute percentile for each sample
        try:
            s = torch.quantile(
                x0_pred_flat.abs(),
                percentile,
                dim=1,
                keepdim=True
            )
        except AttributeError:
            # Fallback for old PyTorch without quantile
            if NUMPY_AVAILABLE:
                s_np = np.quantile(x0_pred_flat.abs().cpu().numpy(), percentile, axis=1, keepdims=True)
                s = torch.tensor(s_np, device=x0_pred.device, dtype=torch.float32)
            else:
                # Ultra-fallback: use kthvalue
                k = max(1, int(percentile * x0_pred_flat.shape[1]))
                s = x0_pred_flat.abs().kthvalue(k, dim=1, keepdim=True).values

        # POLISH: Adaptive floor based on content
        if self.config.dynamic_thresholding_adaptive_floor:
            # Use adaptive floor based on mean absolute value
            content_scale = x0_pred_flat.abs().mean(dim=1, keepdim=True)
            floor = torch.maximum(torch.ones_like(s), content_scale * 0.5)
            s = torch.maximum(s, floor)
        else:
            # Original fixed floor = 1.0
            s = torch.maximum(s, torch.ones_like(s))

        s = s.view(-1, *([1] * (x0_pred.dim() - 1)))

        # Threshold and rescale (cast back to original dtype)
        x0_pred = torch.clamp(x0_pred, -s.to(original_dtype), s.to(original_dtype)) / s.to(original_dtype)

        return x0_pred

    def ddim_step(self, x_t: torch.Tensor, noise_pred: torch.Tensor,
                  t_curr: float, t_next: float, eta: float = 0.0) -> torch.Tensor:
        """DDIM sampling step with improved stability

        POLISH: Using clearer variable names (alpha_bar instead of alpha)
        """
        alpha_bar_t = self._get_cached_noise_schedule(t_curr)
        alpha_bar_next = self._get_cached_noise_schedule(t_next)

        # Predict x0
        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(1.0 - alpha_bar_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

        # Clip predictions for stability
        if self.config.use_dynamic_thresholding:
            x0_pred = self.dynamic_threshold(x0_pred)
        else:
            x0_pred = torch.clamp(x0_pred, -1, 1)

        # Validate ratio instead of clamping - fail fast on bad schedules
        ratio = alpha_bar_t / max(alpha_bar_next, 1e-12)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"Invalid noise schedule: alpha_bar_t/alpha_bar_next={ratio:.6f} at t_curr={t_curr:.4f}, t_next={t_next:.4f}. "
                f"Schedule must be monotonically decreasing (alpha_bar_t={alpha_bar_t:.6f}, alpha_bar_next={alpha_bar_next:.6f})."
            )

        # Compute variance
        sigma_t = eta * math.sqrt(max(0.0, (1 - alpha_bar_next) / max(1 - alpha_bar_t, 1e-12))) * math.sqrt(1 - ratio)

        # Compute mean (with protection against negative values under sqrt)
        sqrt_alpha_bar_next = math.sqrt(alpha_bar_next)
        under = max(0.0, 1.0 - alpha_bar_next - sigma_t**2)
        pred_sample_direction = math.sqrt(under) * noise_pred

        # Add noise if eta > 0
        x_next = sqrt_alpha_bar_next * x0_pred + pred_sample_direction
        if eta > 0 and t_next > 0:
            gen = getattr(self.config, 'generator', None)
            noise = _randn_like_compat(x_t, gen)
            x_next = x_next + sigma_t * noise

        return x_next

    def ddpm_step_improved(self, x_t: torch.Tensor, noise_pred: torch.Tensor,
                          t_curr: float, t_next: float) -> torch.Tensor:
        """Improved DDPM step with better variance handling

        POLISH: Using clearer variable names throughout
        """
        # Early shape validation for learned variance
        if self.config.use_learned_variance:
            expected_channels = 2 * x_t.shape[1]
            if noise_pred.shape[1] != x_t.shape[1] and noise_pred.shape[1] != expected_channels:
                raise ValueError(
                    f"With use_learned_variance=True, noise_pred must have either C={x_t.shape[1]} "
                    f"(fixed variance) or 2C={expected_channels} channels ([||logvar]). "
                    f"Got {noise_pred.shape[1]} channels. Check your model output."
                )

        alpha_bar_t = self._get_cached_noise_schedule(t_curr)
        alpha_bar_next = self._get_cached_noise_schedule(t_next)

        # Use the actual next sampling time as the "previous"  to match the chosen discretization
        alpha_bar_prev = alpha_bar_next if t_next > 0 else 1.0

        # Numerically safe _t
        beta_t = 1.0 - alpha_bar_t / max(alpha_bar_prev, 1e-12)
        beta_t = max(beta_t, 1e-20)

        # Track if learned variance path is used
        learned_variance_used = False

        # Improved variance computation
        if self.config.use_learned_variance and noise_pred.shape[1] == 2 * x_t.shape[1]:
            # Model predicts both mean and variance
            learned_variance_used = True
            c = noise_pred.shape[1]
            xc = x_t.shape[1]
            if c != 2 * xc:
                raise ValueError(f"Expected [||logvar] with {2*xc} channels, got {c}")
            # Split noise_pred if model predicts both mean and variance
            noise_pred, log_variance = torch.chunk(noise_pred, 2, dim=1)
            # Interpolate between minimum and maximum variance
            min_log = math.log(max(1e-20, beta_t))
            # Guard division in variance computation
            denominator = max(1 - alpha_bar_t, 1e-12)
            max_log = math.log(max(1e-20, beta_t * (1 - alpha_bar_prev) / denominator))
            frac = (log_variance + 1) / 2  # Assume model outputs in [-1, 1]
            model_log_variance = frac * max_log + (1 - frac) * min_log
            variance = torch.exp(model_log_variance).clamp_min(0.0)  # Explicit clamp
        else:
            # Use fixed variance schedule
            # Guard against tiny denominator
            denominator = max(1 - alpha_bar_t, 1e-12)
            variance = beta_t * (1 - alpha_bar_next) / denominator

            # One-time warning if learned variance configured but not used
            if self.config.use_learned_variance and not hasattr(self, '_learned_variance_warned'):
                logger.info("Learned variance configured but model output shape doesn't match. "
                            f"Expected {2 * x_t.shape[1]} channels for [||logvar], got {noise_pred.shape[1]}. "
                            "Using fixed variance schedule.")
                self._learned_variance_warned = True

        # Compute mean with guards for tiny denominators
        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(max(1.0 - alpha_bar_t, 1e-12))

        # Predict x0 and clip
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / max(sqrt_alpha_bar_t, 1e-12)
        if self.config.use_dynamic_thresholding:
            x0_pred = self.dynamic_threshold(x0_pred)
        else:
            x0_pred = torch.clamp(x0_pred, -1, 1)

        # Correct posterior mean coefficients
        # The posterior mean is:  = _{t-1} * _t/(1-_t) * x0 + _t * (1-_{t-1})/(1-_t) * x_t
        denominator = max(1 - alpha_bar_t, 1e-12)
        alpha_step = max(alpha_bar_t / max(alpha_bar_next, 1e-12), 0.0)  # per-step alpha = alpha_bar_t/alpha_bar_next ( 1)
        mean_coef1 = math.sqrt(alpha_bar_next) * beta_t / denominator
        mean_coef2 = math.sqrt(alpha_step) * (1 - alpha_bar_next) / denominator
        mean = mean_coef1 * x0_pred + mean_coef2 * x_t

        # Add noise
        if t_next > 0:
            gen = getattr(self.config, 'generator', None)
            noise = _randn_like_compat(x_t, gen)
            if torch.is_tensor(variance):
                # Handle tensor variance (from learned variance models)
                std_dev = torch.clamp(variance, min=0.0).sqrt()
                # If model predicted per-channel/voxel variance, ensure proper shape
                while std_dev.dim() < x_t.dim():
                    std_dev = std_dev.unsqueeze(-1)
                if std_dev.shape != x_t.shape:
                    std_dev = std_dev.expand_as(x_t)
            else:
                # Handle scalar variance
                std_dev = math.sqrt(max(0.0, variance))
            x_next = mean + std_dev * noise
        else:
            x_next = mean

        return x_next

    def sample_improved(
        self,
        shape: Tuple[int, int, int, int],
        timesteps: List[float],
        verbose: bool = True,
        guidance_scale: float = None,
        use_ddim: bool = None,
        eta: float = None,
        init_samples: Optional[torch.Tensor] = None,
        callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Improved sampling with better score matching and DDIM support

        POLISH: Fixed guidance to scale score direction, not just noise
        """
        # Ensure no autograd for memory efficiency
        no_grad_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with no_grad_ctx():
            # Use config defaults if not specified
            if guidance_scale is None:
                guidance_scale = self.config.guidance_scale
            if use_ddim is None:
                use_ddim = self.config.use_ddim_sampling
            if eta is None:
                eta = self.config.ddim_eta

            # Validate timesteps
            if len(timesteps) < 2:
                raise ValueError("timesteps must have length  2 (descending t in [1,0]).")

            # Proper discrete timestep conversion
            if self.config.discrete_timesteps:
                N = self.config.num_timesteps
                snapped = []
                for t in timesteps:
                    # Accept ints or floats; snap floats to grid i/(N-1)
                    if isinstance(t, int):
                        i = max(0, min(N-1, t))
                    else:
                        i = int(round(float(t) * (N - 1)))
                        i = max(0, min(N-1, i))
                    snapped.append(i / (N - 1))
                timesteps = snapped

            # Ensure timesteps are sorted in descending order
            timesteps = sorted(timesteps, reverse=True)

            # Validate ALL timesteps are in range [0, 1], not just endpoints
            if any((t < 0.0) or (t > 1.0) for t in timesteps):
                invalid = [t for t in timesteps if t < 0.0 or t > 1.0]
                raise ValueError(f"All timesteps must be in range [0, 1]. Found invalid values: {invalid}")

            # Remove duplicates while preserving order
            timesteps_unique = []
            prev = None
            for t in timesteps:
                if prev is None or abs(t - prev) > 1e-10:
                    timesteps_unique.append(t)
                    prev = t
            timesteps = timesteps_unique

            if len(timesteps) < 2:
                raise ValueError("After removing duplicates, timesteps must have length  2")

            # Explicit monotonicity check for clearer errors
            if any(timesteps[i] <= timesteps[i+1] for i in range(len(timesteps)-1)):
                raise ValueError(f"Timesteps must be strictly descending (from 1  0). "
                                 f"Got: {timesteps[:5]}... Check your timestep ordering.")

            # Initialize samples
            if init_samples is not None:
                x_t = init_samples.to(self.device, non_blocking=True)
            else:
                gen = getattr(self.config, 'generator', None)
                x_t = _randn_like_compat(torch.zeros(shape, device=self.device, dtype=self.amp_dtype), gen)

            if x_t.dim() == 4 and hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
                try:
                    x_t = x_t.contiguous(memory_format=torch.channels_last)
                except TypeError:
                    x_t = x_t.contiguous()

            # Reset momentum if using momentum transport
            if self.config.use_momentum_transport:
                self.momentum_transport.reset_velocity()

            # Create iterator
            iterator = tqdm(range(len(timesteps)-1), desc="Sampling (Improved)") if verbose and TQDM_AVAILABLE else range(len(timesteps)-1)

            # Sample with improved algorithm
            for i in iterator:
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]

                # Get noise schedule values (using clearer names)
                alpha_bar_t = self._get_cached_noise_schedule(t_curr)

                # Get score prediction
                score = self.compute_score_cached(x_t, t_curr)

                # Convert score to noise for DDPM/DDIM sampling
                sigma_t = math.sqrt(1.0 - alpha_bar_t)
                noise_pred = -score * sigma_t

                # POLISH: Improved guidance that scales score direction
                if guidance_scale != 1.0 and guidance_scale > 0:
                    if self.config.guidance_mode == "score":
                        # Scale the score direction (better approach)
                        score_guided = guidance_scale * score
                        guided_noise = -score_guided * sigma_t
                    else:
                        # Original noise scaling (less principled but sometimes works)
                        guided_noise = guidance_scale * noise_pred
                else:
                    guided_noise = noise_pred

                if use_ddim:
                    # DDIM sampling (deterministic or with controlled noise)
                    x_t = self.ddim_step(x_t, guided_noise, t_curr, t_next, eta)
                else:
                    # Improved DDPM sampling with variance schedule
                    x_t = self.ddpm_step_improved(x_t, guided_noise, t_curr, t_next)

                # No-NaN invariant check for debugging
                if not torch.isfinite(x_t).all():
                    raise RuntimeError(
                        f"NaN/Inf detected in sampling at step transition "
                        f"t_curr={t_curr:.4f} -> t_next={t_next:.4f}. "
                        f"Check your noise schedule and model outputs."
                    )

                # Apply callback if provided
                if callback is not None:
                    callback(t_next, x_t)

                # Update progress bar if verbose
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(
                        t=f"{t_next:.4f}",
                        mean=f"{x_t.mean().item():.3f}",
                        std=f"{x_t.std().item():.3f}"
                    )

            # Final clamping
            x_t = torch.clamp(x_t, -1, 1)

            return x_t

    def apply_guidance(self, x_t: torch.Tensor, score: torch.Tensor,
                      guidance_scale: float, alpha_bar_t: float,
                      t_curr: float, t_next: float) -> torch.Tensor:
        """Apply simple score scaling guidance (NOT classifier-free guidance).

        DEPRECATED: Use sample_improved() with guidance_scale parameter instead.

        POLISH: Using clearer variable names (alpha_bar_t)
        """
        # Warn about deprecated API
        if not hasattr(self, '_guidance_deprecated_warned'):
            logger.info("apply_guidance() is deprecated. The improved sampling path in sample_improved() "
                        "now applies guidance before each step. Consider using sample_improved() instead.")
            self._guidance_deprecated_warned = True

        # Simple guidance by scaling the score
        if guidance_scale != 1.0:
            # Apply guidance (simplified: guidance just scales the score)
            guided_score = guidance_scale * score

            # Compute guided update with actual timestep difference
            sigma_t = math.sqrt(max(1e-8, 1.0 - alpha_bar_t))
            dt = max(1e-6, float(t_curr) - float(t_next))
            x_t = x_t - dt * sigma_t * guided_score

        return x_t

    def compute_controlled_drift(self, x: torch.Tensor, score: torch.Tensor, alpha_bar_t: float, dt: torch.Tensor) -> torch.Tensor:
        """Enhanced drift with control variate (dt already in FP32)

        Args:
            x: current state
            score: score estimate
            alpha_bar_t: (t) at current time (NOT the time index t)
            dt: step size (float tensor, ideally FP32)
        """
        drift = -0.5 * (1 - alpha_bar_t) * score * dt
        drift = drift.to(x.dtype)

        if self.config.control_variate_strength > 0:
            score_norm = torch.norm(score.reshape(x.shape[0], -1), dim=1, keepdim=True)
            std = score_norm.std(unbiased=False)
            mean = score_norm.mean()
            # Renamed misleading variable - it's a coefficient, not variance
            coeff = std / (mean + 1e-8)

            # Use the "noise level" proxy (1 - ) rather than misusing t
            control_strength = self.config.control_variate_strength * coeff * (1.0 - alpha_bar_t)
            drift = drift * (1 + control_strength)

        return drift

    def compute_fisher_transport(self, x: torch.Tensor, score: torch.Tensor, alpha_bar_t: float, dt: torch.Tensor) -> torch.Tensor:
        """Transport using Fisher information geometry (dt in FP32)

        Args:
            x: current state
            score: score estimate
            alpha_bar_t: (t) at current time (NOT the time index t)
            dt: step size (float tensor, ideally FP32)
        """
        fisher_diag = self.kernel_module.estimate_fisher_diagonal(x, score, t=0.0, alpha=alpha_bar_t)

        if self.config.use_fp32_fisher and score.dtype in [torch.float16, torch.bfloat16]:
            score_fp32 = score.float()
            fisher_fp32 = fisher_diag.float()
            natural_grad = score_fp32 / (fisher_fp32 + 1e-6)
            natural_grad = natural_grad.to(score.dtype)
        else:
            natural_grad = score / (fisher_diag + 1e-6)

        curvature = fisher_diag.mean(dim=tuple(range(1, fisher_diag.dim())), keepdim=True)
        step_size = dt / (1 + curvature)

        one_minus_alpha_bar = x.new_tensor(1 - alpha_bar_t)
        half = x.new_tensor(0.5)
        transport = x - step_size * natural_grad * half * one_minus_alpha_bar

        return transport.to(x.dtype)

    def compute_drift_and_transport_inline(self, x: torch.Tensor, drift: torch.Tensor,
                                          alpha_bar: Union[float, torch.Tensor]) -> torch.Tensor:
        """Inline drift and transport with improved bounds

        Args:
            x: current state
            drift: proposed drift update
            alpha_bar: (t) at current time (scalar/tensor)
        """
        legacy_fp32 = (
            getattr(self.config, "legacy_transport_mode", False)
            and x.dtype in (torch.float16, torch.bfloat16)
        )
        if legacy_fp32:
            x_work = x.float()
            drift_work = drift.float()
        else:
            x_work = x
            drift_work = drift

        if not torch.is_tensor(alpha_bar):
            alpha_bar = x_work.new_tensor(alpha_bar)

        if TRITON_AVAILABLE and self.config.use_triton_kernels and x_work.is_cuda and x_work.numel() > 65536:
            x_flat = x_work.reshape(-1).contiguous()
            drift_flat = drift_work.reshape(-1).contiguous()
            out = torch.empty_like(x_flat)

            # Triton scalar as 1-elem tensor for future-proofing
            scale_val = (5.0 * (1 - alpha_bar)).float().mean().clamp_(0.1, 10.0)
            scale_buf = torch.tensor([float(scale_val)], device=x_work.device, dtype=x_work.dtype)

            n_elements = x_flat.numel()
            launch_triton_kernel_safe(
                fused_drift_transport_kernel_fixed,
                x_flat, drift_flat, out, scale_buf,
                n_elements=n_elements,
                kernel_type="default"
            )

            result = out.reshape(x_work.shape)
            return result.to(x.dtype) if legacy_fp32 else result

        if x_work.dim() == 4:
            drift_norms = torch.norm(drift_work.reshape(x_work.shape[0], x_work.shape[1], -1), dim=2, p=2)
            drift_norms = drift_norms.mean(dim=1, keepdim=True)
        else:
            drift_norms = torch.norm(drift_work.reshape(x_work.shape[0], -1), dim=1, keepdim=True)

        mean_norm = drift_norms.float().mean().clamp_(min=1e-12, max=1e6).to(drift_norms.dtype)

        norm_ratio = drift_norms / mean_norm
        norm_ratio_clamped = torch.clamp(norm_ratio, min=0.25, max=4.0)
        weights = norm_ratio_clamped / (norm_ratio_clamped + 1.0)

        weights = weights.clamp_min_(self.config.transport_weight_min).clamp_max_(self.config.transport_weight_max)

        for _ in range(x.dim() - len(weights.shape)):
            weights = weights.unsqueeze(-1)

        transported = x_work + weights * drift_work
        return transported.to(x.dtype) if legacy_fp32 else transported

    def fft_grid_transport_inline(self, x: torch.Tensor, drift: torch.Tensor,
                                  alpha_bar: Union[float, torch.Tensor],
                                  kernel_fft: torch.Tensor,
                                  freq_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Inline FFT-based transport - fully tensor-native with precomputed kernels

        POLISH: Added FP32 guards for FFT with FP16/BF16 inputs
        """
        B, C = x.shape[:2]
        spatial_dims = x.shape[2:]

        y_pred = x + drift

        # POLISH: FP32 guards for FFT stability with half precision
        original_dtype = x.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
            y_pred = y_pred.float()

        x_fft = torch.fft.rfftn(x, dim=tuple(range(2, x.dim())))
        y_fft = torch.fft.rfftn(y_pred, dim=tuple(range(2, y_pred.dim())))

        kernel = kernel_fft.unsqueeze(0).unsqueeze(0).to(y_fft.real.dtype)
        kernel = torch.maximum(kernel, kernel.new_tensor(1e-6))
        y_r, y_i = y_fft.real, y_fft.imag
        smoothed_fft = torch.complex(y_r * kernel, y_i * kernel)

        if self.config.freq_weighting and freq_weights is not None:
            freq_weights_expanded = freq_weights.unsqueeze(0).unsqueeze(0)

            diff_real = (y_fft.real - x_fft.real).abs()
            diff_imag = (y_fft.imag - x_fft.imag).abs()
            diff_mag = torch.sqrt(diff_real**2 + diff_imag**2)

            weighted_diff = diff_mag * freq_weights_expanded

            diff_norm = weighted_diff.reshape(B, C, -1).mean(dim=(1, 2))
            weights = torch.sigmoid(diff_norm * 3.0).view(B, 1, 1, 1)
            weights = weights.to(x_fft.real.dtype)
        else:
            weights = x_fft.real.new_tensor(0.5)

        result_fft = (1 - weights) * x_fft + weights * smoothed_fft

        if result_fft.dtype != torch.complex64 and result_fft.dtype != torch.complex128:
            result_fft = result_fft.to(smoothed_fft.dtype)

        result = torch.fft.irfftn(result_fft, s=spatial_dims, dim=tuple(range(2, x.dim())))

        # Cast back to original dtype
        result = result.to(original_dtype)

        if hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
            try:
                if x.is_contiguous(memory_format=torch.channels_last):
                    result = result.contiguous(memory_format=torch.channels_last)
            except (TypeError, AttributeError):
                pass

        return result

    def iterative_transport_vectorized(
        self,
        x: torch.Tensor,
        score: torch.Tensor,
        dt: Union[float, torch.Tensor],
        alpha_bar_t: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Clean API name for iterative transport

        POLISH: Using clearer parameter name
        """
        return self.iterative_transport_vectorized_fixed(x, score, dt, alpha_bar_t)

    def iterative_transport_vectorized_fixed(
        self,
        x: torch.Tensor,
        score: torch.Tensor,
        dt: Union[float, torch.Tensor],
        alpha_bar_t: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Fully vectorized iterative transport with GPU-optimized ops

        POLISH: Using clearer parameter names
        """
        # Ensure dt is FP32 tensor
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, dtype=torch.float32, device=x.device)

        # Get alpha_bar as scalar for computation
        if torch.is_tensor(alpha_bar_t):
            alpha_bar_val = float(alpha_bar_t.item())
        else:
            alpha_bar_val = float(alpha_bar_t)

        # Use Fisher transport for high precision
        if self.config.use_fisher_geometry:
            x_new = self.compute_fisher_transport(x, score, alpha_bar_val, dt)
        else:
            drift = self.compute_controlled_drift(x, score, alpha_bar_val, dt)
            x_new = self.compute_drift_and_transport_inline(x, drift, alpha_bar_val)

        # Hierarchical bridge transport
        if self.config.use_hierarchical_bridge and x.dim() == 4:
            drift = -0.5 * (1 - alpha_bar_val) * score * dt
            drift = drift.to(x.dtype)
            x_new = self.hierarchical_bridge.compute_multiscale_transport(x, drift, alpha_bar_val)

        # Momentum transport
        if self.config.use_momentum_transport:
            drift = -0.5 * (1 - alpha_bar_val) * score * dt
            drift = drift.to(x.dtype)
            x_new = self.momentum_transport.apply_transport(x_new, drift, alpha_bar_val)

        return x_new

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        timesteps: List[float],
        verbose: bool = True,
        run_uid: Optional[int] = None,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Main sampling method with enhanced transport and improved timestep handling

        Args:
            shape: Target shape (B, C, H, W)
            timesteps: List of timesteps from 1 to 0
            verbose: Show progress bar
            run_uid: Optional unique ID for caching
            return_trajectory: Return intermediate states

        Returns:
            Final samples or (samples, trajectory) if return_trajectory=True
        """
        # Use no_grad for memory efficiency
        no_grad_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with no_grad_ctx():
            if len(timesteps) < 2:
                raise ValueError("timesteps must have length  2")

            # Proper discrete timestep conversion
            if self.config.discrete_timesteps:
                N = self.config.num_timesteps
                snapped = []
                for t in timesteps:
                    if isinstance(t, int):
                        i = max(0, min(N-1, t))
                    else:
                        i = int(round(float(t) * (N - 1)))
                        i = max(0, min(N-1, i))
                    snapped.append(i / (N - 1))
                timesteps = snapped

            # Validate timesteps
            timesteps = sorted(timesteps, reverse=True)
            if timesteps[0] > 1.0 or timesteps[-1] < 0.0:
                raise ValueError(f"Timesteps must be in range [0, 1]. Got [{timesteps[-1]}, {timesteps[0]}]")

            # Pre-cache schedule values
            for t in timesteps:
                _ = self._get_cached_noise_schedule(t)

            # Initialize
            gen = getattr(self.config, 'generator', None)
            x_t = _randn_like_compat(torch.zeros(shape, device=self.device, dtype=self.amp_dtype), gen)

            if x_t.dim() == 4 and hasattr(self.config, 'use_channels_last') and self.config.use_channels_last:
                try:
                    x_t = x_t.contiguous(memory_format=torch.channels_last)
                except TypeError:
                    x_t = x_t.contiguous()

            trajectory = [x_t.clone()] if return_trajectory else []

            # Precompute FFT kernels if needed
            if x_t.dim() == 4 and x_t.shape[-1] >= 64:
                sigma = 0.25 * max(x_t.shape[-2:])
                kernel_fft = self.kernel_module.compute_gaussian_kernel_fft(
                    x_t.shape[2:], sigma, x_t.device
                )
                freq_weights = self.kernel_module.get_frequency_weights(
                    x_t.shape[2:], str(x_t.device)
                ) if self.config.freq_weighting else None
            else:
                kernel_fft = None
                freq_weights = None

            # Reset momentum
            if self.config.use_momentum_transport:
                self.momentum_transport.reset_velocity()

            # Main sampling loop
            iterator = tqdm(range(len(timesteps)-1), desc="Sampling") if verbose and TQDM_AVAILABLE else range(len(timesteps)-1)

            for i in iterator:
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = max(1e-6, float(t_curr) - float(t_next))

                # Ensure dt is FP32
                if self.config.use_fp32_time:
                    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=x_t.device)
                else:
                    dt_tensor = torch.tensor(dt, dtype=x_t.dtype, device=x_t.device)

                # Get score
                if x_t.shape[-1] > self.config.max_patch_size:
                    score = self.compute_score_patches_fixed(x_t, t_curr, run_uid)
                else:
                    cache_key = f"run{run_uid}_t{t_curr:.6f}" if run_uid else f"t{t_curr:.6f}"
                    score = self.compute_score_cached(x_t, t_curr, cache_key)

                # Get alpha_bar
                alpha_bar_t = self._get_cached_noise_schedule(t_curr)

                # Transport
                x_t = self._sample_step(x_t, score, alpha_bar_t, dt_tensor, kernel_fft, freq_weights)

                # Corrector steps
                for _ in range(self.config.corrector_steps):
                    gen = getattr(self.config, 'generator', None)
                    noise = _randn_like_compat(x_t, gen) * math.sqrt(2 * dt * self.config.corrector_snr)
                    x_t = x_t + noise

                    score_corr = self.compute_score_cached(x_t, t_next)
                    x_t = x_t + dt_tensor * self.config.corrector_snr * score_corr

                if return_trajectory:
                    trajectory.append(x_t.clone())

                # Update progress
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(
                        t=f"{t_next:.4f}",
                        mean=f"{x_t.mean().item():.3f}",
                        std=f"{x_t.std().item():.3f}"
                    )

            # Final clamp
            x_t = torch.clamp(x_t, -1, 1)

            if return_trajectory:
                trajectory.append(x_t)
                return x_t, trajectory

            return x_t

    def sample_batch_ot(
        self,
        source_batch: torch.Tensor,
        target_batch: torch.Tensor,
        timesteps: List[float],
        verbose: bool = True
    ) -> torch.Tensor:
        """Sample using optimal transport between batches

        Args:
            source_batch: Source samples
            target_batch: Target samples
            timesteps: Sampling timesteps
            verbose: Show progress

        Returns:
            Transported samples
        """
        if source_batch.shape != target_batch.shape:
            raise ValueError(f"Shape mismatch: source {source_batch.shape} != target {target_batch.shape}")

        # Apply adaptive eps based on timestep
        alpha_bar_init = self._get_cached_noise_schedule(timesteps[0])
        eps = self.config.ot_eps_min * (1.0 + 9.0 * (1.0 - alpha_bar_init))

        # Choose transport method based on memory
        method = self.config.batch_ot_method

        if method == "sliced":
            n_proj = self.config.sliced_ot_projections
            x_t = self.transport_module.sliced_ot.transport(
                source_batch, target_batch, eps, n_proj
            )
        else:
            # Full OT
            x_t = self.transport_module.sliced_ot._full_ot(
                source_batch.unsqueeze(2),
                target_batch.unsqueeze(2),
                eps
            ).squeeze(2)

        # Continue regular sampling
        return self.sample(
            x_t.shape,
            timesteps[1:],
            verbose=verbose
        )

    def create_optimal_timesteps(
        self,
        num_steps: int,
        schedule_type: str = "linear",
        schedule_power: float = 1.0
    ) -> List[float]:
        """Create optimal timestep schedule for sampling

        Args:
            num_steps: Number of sampling steps
            schedule_type: "linear", "quadratic", "cosine", "uniform_alpha_bar", or "log_snr"
            schedule_power: Power for polynomial schedules

        Returns:
            List of timesteps from 1 to 0
        """
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")

        if schedule_type == "linear":
            # Linear spacing in t
            timesteps = torch.linspace(1.0, 0.0, num_steps).tolist()

        elif schedule_type == "quadratic":
            # Quadratic spacing (more steps near t=0)
            t = torch.linspace(0, 1, num_steps) ** 2
            timesteps = (1.0 - t).tolist()

        elif schedule_type == "cosine":
            # Cosine schedule
            t = torch.linspace(0, 1, num_steps)
            timesteps = (0.5 * (1 + torch.cos(math.pi * t))).tolist()

        elif schedule_type == "uniform_alpha_bar":
            # Uniform spacing in alpha_bar space
            alpha_bars = torch.linspace(1.0, 0.0, num_steps)
            timesteps = []
            for alpha_bar_target in alpha_bars:
                # Find t that gives this alpha_bar
                # Binary search since noise_schedule is monotonic
                left, right = 0.0, 1.0
                for _ in range(32):  # 32 iterations for precision
                    mid = (left + right) / 2
                    alpha_bar_mid = self._get_cached_noise_schedule(mid)
                    if alpha_bar_mid > alpha_bar_target:
                        left = mid
                    else:
                        right = mid
                timesteps.append((left + right) / 2)

        elif schedule_type == "log_snr":
            # Log-SNR schedule (POLISH: fixed mapping)
            # Classical log-SNR: log(alpha_bar/(1-alpha_bar))
            # Sample uniformly in log-SNR space
            log_snr_max = 4.0  # log(~0.98/0.02)
            log_snr_min = -4.0  # log(~0.02/0.98)
            log_snrs = torch.linspace(log_snr_max, log_snr_min, num_steps)

            timesteps = []
            for log_snr_target in log_snrs:
                # Find t that gives this log-SNR
                # alpha_bar/(1-alpha_bar) = exp(log_snr)
                # alpha_bar = exp(log_snr)/(1 + exp(log_snr)) = sigmoid(log_snr)
                alpha_bar_target = torch.sigmoid(log_snr_target).item()

                # Binary search for t
                left, right = 0.0, 1.0
                for _ in range(32):
                    mid = (left + right) / 2
                    alpha_bar_mid = self._get_cached_noise_schedule(mid)
                    if alpha_bar_mid > alpha_bar_target:
                        left = mid
                    else:
                        right = mid
                timesteps.append((left + right) / 2)

        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        # Ensure endpoints
        if abs(timesteps[0] - 1.0) > 1e-3:
            timesteps[0] = 1.0
        if abs(timesteps[-1]) > 1e-3:
            timesteps[-1] = 0.0

        # Remove duplicates
        unique_timesteps = []
        prev = None
        for t in timesteps:
            if prev is None or abs(t - prev) > 1e-6:
                unique_timesteps.append(t)
                prev = t

        return unique_timesteps

    def reset_caches(self):
        """Reset all caches"""
        self.score_cache.reset()
        self.kernel_module.kernel_cache.reset()
        self.kernel_module.fisher_cache.reset()
        self._overlap_cache.clear()

        if hasattr(self, 'momentum_transport'):
            self.momentum_transport.reset_velocity()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {
            "score_cache_mb": self.score_cache.current_size / (1024 * 1024),
            "kernel_cache_mb": self.kernel_module.kernel_cache.current_size / (1024 * 1024),
            "fisher_cache_mb": self.kernel_module.fisher_cache.current_size / (1024 * 1024),
        }

        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)

        return stats





def make_schedule(schedule_type: str = "linear",
                  beta_start: float = 0.0001,
                  beta_end: float = 0.02,
                  num_timesteps: int = 1000) -> Callable[[float], float]:
    """Create noise schedule function returning _bar(t) for t[0,1]

    POLISH: Clearer documentation about continuous-time _bar

    Args:
        schedule_type: "linear", "cosine", "quadratic", or "sigmoid"
        beta_start: Starting beta value
        beta_end: Ending beta value
        num_timesteps: Number of discretization steps (for reference)

    Returns:
        Function mapping t[0,1] to _bar(t)[0,1]

    Note: This returns the continuous-time _bar(t) = exp(-(s)ds),
    not the discrete product form. For discrete DDPM schedules,
    use your own schedule function.
    """

    def linear_schedule(t: float) -> float:
        """Linear beta schedule  exponential _bar"""
        # Continuous integral form: _bar(t) = exp(-(s)ds)
        beta_t = beta_start + t * (beta_end - beta_start)
        # Integral of linear: (a + bs)ds = as + bs2/2
        integral = beta_start * t + 0.5 * (beta_end - beta_start) * t**2
        return math.exp(-integral)

    def cosine_schedule(t: float) -> float:
        """Cosine schedule from improved DDPM"""
        s = 0.008
        f_t = math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0

    def quadratic_schedule(t: float) -> float:
        """Quadratic beta schedule"""
        beta_t = beta_start + (beta_end - beta_start) * t**2
        # Integral: (a + bs2)ds = as + bs3/3
        integral = beta_start * t + (beta_end - beta_start) * t**3 / 3
        return math.exp(-integral)

    def sigmoid_schedule(t: float) -> float:
        """Sigmoid-based schedule"""
        beta_max = beta_end
        beta_min = beta_start
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (beta_max - beta_min) + beta_min

        # Compute alpha_bar products
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Interpolate
        idx = min(int(t * (num_timesteps - 1)), num_timesteps - 2)
        w = t * (num_timesteps - 1) - idx
        return (1 - w) * alpha_bars[idx].item() + w * alpha_bars[idx + 1].item()

    schedules = {
        "linear": linear_schedule,
        "cosine": cosine_schedule,
        "quadratic": quadratic_schedule,
        "sigmoid": sigmoid_schedule
    }

    if schedule_type not in schedules:
        raise ValueError(f"Unknown schedule_type: {schedule_type}. Choose from {list(schedules.keys())}")

    return schedules[schedule_type]



def example_usage():
    """Example of how to use the FastSB-OT solver with CFG compatibility"""

    # Mock score model
    class MockScoreModel(nn.Module):
        def forward(self, x, t):
            return torch.randn_like(x)

    # For CFG compatibility, wrap your model:
    class CFGWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, t, condition=None):
            if condition is not None:
                # Get conditional and unconditional scores
                score_cond = self.model(x, t, condition)
                score_uncond = self.model(x, t, None)
                # Return the difference for CFG
                return score_uncond + self.cfg_scale * (score_cond - score_uncond)
            return self.model(x, t)

    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MockScoreModel()

    # Create noise schedule (_bar(t) from 10)
    noise_schedule = make_schedule("cosine", num_timesteps=1000)

    # Configure solver
    config = FastSBOTConfig(
        quality="balanced",
        use_mixed_precision=True,
        use_ddim_sampling=True,
        ddim_eta=0.0,  # Deterministic
        guidance_scale=1.5,  # Score scaling
        guidance_mode="score",  # Use score-based guidance
        use_dynamic_thresholding=True,
        seed=42
    )

    # Create solver
    solver = FastSBOTSolver(model, noise_schedule, config, device)

    # Generate samples
    shape = (4, 3, 256, 256)
    timesteps = solver.create_optimal_timesteps(50, "uniform_alpha_bar")

    # Use improved sampling
    samples = solver.sample_improved(
        shape=shape,
        timesteps=timesteps,
        guidance_scale=1.5,
        use_ddim=True,
        eta=0.0,
        verbose=True
    )

    print(f"Generated samples: {samples.shape}")
    print(f"Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")

    # Check memory usage
    memory_stats = solver.get_memory_usage()
    print(f"Memory usage: {memory_stats}")

    return samples




