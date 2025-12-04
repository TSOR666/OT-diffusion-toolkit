"""
FastSB-OT: Production Solver - FINAL POLISHED+
Core solver implementation with all critical patches, production hardening, and final polish.

Author: Thierry Silvio Claude Soreze
Status: Production FINAL POLISHED+
Build: 2025.01.07.020  # Production final polished+ build

Key Highlights:
- GENERATOR: Device-correct seeded RNG (ensures deterministic CUDA/MPS sampling when seed is set)
- DRIFT/FISHER API: Fixed misuse of t vs  (alpha_bar) in drift/transport helpers
- FISHER CACHE:  included in cache key to avoid incorrect reuse across noise levels
- RNG: Safer fallback if a mismatched-device generator is supplied
- DOCS: Clarified parameter names where  (alpha_bar) was previously referred to as alpha/t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import sys
import json
import random
import inspect
import threading
import logging
import tempfile
from typing import Callable, Tuple, List, Optional, Dict, Union, Any, Literal
from collections import OrderedDict
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum

# CORRECTNESS FIX: Fallback for packaging import
try:
    from packaging.version import Version
except Exception:
    # Conservative fallback if packaging isn't installed
    # FIX: Parse version string to enable proper comparisons
    class Version:
        """Minimal version comparison (fallback when packaging unavailable)"""
        def __init__(self, version_str):
            self.version = version_str
<<<<<<< Updated upstream
            # Parse major.minor.patch from version string
            try:
                parts = version_str.split('.')
                self.major = int(parts[0]) if len(parts) > 0 else 0
                self.minor = int(parts[1]) if len(parts) > 1 else 0
                self.patch = int(parts[2].split('+')[0].split('a')[0].split('b')[0].split('rc')[0]) if len(parts) > 2 else 0
            except (ValueError, AttributeError):
                self.major = self.minor = self.patch = 0

        def __ge__(self, other):
            # Proper version comparison instead of always returning False
            if not isinstance(other, Version):
                return False
            return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)
=======
            try:
                # Parse version string (handles formats like "2.1.0", "2.1.0rc1", etc.)
                parts = version_str.split('.')
                self.major = int(parts[0]) if len(parts) > 0 else 0
                self.minor = int(parts[1]) if len(parts) > 1 else 0
                # Remove suffixes (rc, alpha, beta, etc.)
                patch_str = parts[2] if len(parts) > 2 else "0"
                self.patch = int(patch_str.split('+')[0].split('a')[0].split('b')[0].split('rc')[0])
            except (ValueError, AttributeError, IndexError):
                self.major = self.minor = self.patch = 0

        def _cmp_tuple(self):
            return (self.major, self.minor, self.patch)

        def __ge__(self, other):
            if not isinstance(other, Version):
                return NotImplemented
            return self._cmp_tuple() >= other._cmp_tuple()

        def __gt__(self, other):
            if not isinstance(other, Version):
                return NotImplemented
            return self._cmp_tuple() > other._cmp_tuple()

        def __le__(self, other):
            if not isinstance(other, Version):
                return NotImplemented
            return self._cmp_tuple() <= other._cmp_tuple()

        def __lt__(self, other):
            if not isinstance(other, Version):
                return NotImplemented
            return self._cmp_tuple() < other._cmp_tuple()

        def __eq__(self, other):
            if not isinstance(other, Version):
                return NotImplemented
            return self._cmp_tuple() == other._cmp_tuple()
>>>>>>> Stashed changes

# Try to import numpy (optional for stats)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    # Fallback to Python's statistics module
    import statistics

# Module logger with NullHandler
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Build hash for cache versioning
BUILD_HASH = "2025.01.07.020"  # Production final polished+ build

# Try to import autocast and nullcontext at top level
try:
    from torch import autocast
    AUTOCAST_AVAILABLE = True
except ImportError:
    AUTOCAST_AVAILABLE = False
    def autocast(device_type, dtype=None, enabled=True):
        from contextlib import contextmanager
        @contextmanager
        def noop():
            yield
        return noop()

from contextlib import nullcontext

# Try to import tqdm gracefully
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# PyTorch cross-version RNG compatibility helper
def _randn_like_compat(x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Generate random tensor like x with optional generator (cross-version compatible)

    WARNING: If generator device mismatches x.device, falls back to non-deterministic sampling
    """
    if generator is not None:
        try:
            return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        except RuntimeError as e:
            # Device mismatch - cannot maintain determinism
            logger.warning(
                f"Generator device mismatch with tensor device {x.device}. "
                f"Falling back to non-deterministic sampling. Error: {e}"
            )
            return torch.randn_like(x)
    else:
        return torch.randn_like(x)

# Opt-in CUDA optimizations with env var
def setup_cuda_optimizations():
    """Safe CUDA setup with opt-in TF32 configuration"""
    if not torch.cuda.is_available():
        return

    # Only set TF32 if explicitly requested
    if os.environ.get('FASTSBOT_SET_TF32', '0') == '1':
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True

        if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True

        if hasattr(torch, 'set_float32_matmul_precision'):
            try:
                if Version(torch.__version__) >= Version("2.1.0"):
                    torch.set_float32_matmul_precision("medium")
                else:
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        logger.info("TF32 optimizations enabled via FASTSBOT_SET_TF32=1")

    # Only configure memory allocation if explicitly requested
    if os.environ.get('FASTSBOT_SET_ALLOC_CONF', '0') == '1':
        # Warn if allocator already initialized
        if hasattr(torch.cuda, "is_initialized") and torch.cuda.is_initialized():
            logger.warning("PYTORCH_CUDA_ALLOC_CONF set after CUDA init; it may not take effect in this process. "
                           "Import this module before any CUDA operations for allocator settings to apply.")

        current_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
        wanted = {
            'max_split_size_mb': '512',
            'garbage_collection_threshold': '0.8',
            'expandable_segments': 'True'
        }

        if current_conf:
            try:
                parts = dict(kv.split(':', 1) for kv in current_conf.split(',') if kv and ':' in kv)
            except Exception:
                parts = {}
        else:
            parts = {}

        parts.update({k: v for k, v in wanted.items() if k not in parts})
        final_conf = ','.join(f'{k}:{v}' for k, v in parts.items())
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = final_conf

        logger.info("Memory allocation config set via FASTSBOT_SET_ALLOC_CONF=1")
        logger.debug(f"PYTORCH_CUDA_ALLOC_CONF set to: {final_conf}")

setup_cuda_optimizations()

# Triton availability check with sentinel
_TRITON_AVAILABLE = None
_TRITON_IMPORTS = None

def check_triton_availability():
    """Check Triton availability once and cache result"""
    global _TRITON_AVAILABLE, _TRITON_IMPORTS

    if _TRITON_AVAILABLE is not None:
        return _TRITON_AVAILABLE, _TRITON_IMPORTS

    try:
        import triton
        import triton.language as tl
        from triton import next_power_of_two

        # Define sigmoid for Triton (local helper)
        def tl_sigmoid(x):
            return 1 / (1 + tl.exp(-x))

        _TRITON_AVAILABLE = True
        _TRITON_IMPORTS = (triton, tl, next_power_of_two, tl_sigmoid)

    except ImportError:
        _TRITON_AVAILABLE = False
        _TRITON_IMPORTS = None

    return _TRITON_AVAILABLE, _TRITON_IMPORTS

TRITON_AVAILABLE, _ = check_triton_availability()

PYTORCH_VERSION = Version(torch.__version__).release[:2]
COMPILE_AVAILABLE = hasattr(torch, 'compile')

_TORCH_COMPILE_ACCEPTS_DYNAMIC = None
if COMPILE_AVAILABLE:
    _compile_sig = inspect.signature(torch.compile)
    _TORCH_COMPILE_ACCEPTS_DYNAMIC = 'dynamic' in _compile_sig.parameters

_CACHED_DEVICE_PROPERTIES = {}

# Cache pi tensor to avoid repeated allocations
_PI_TENSOR_CACHE = {}

def get_pi_tensor(device, dtype):
    """Get cached pi tensor for device/dtype"""
    key = (str(device), str(dtype))
    if key not in _PI_TENSOR_CACHE:
        _PI_TENSOR_CACHE[key] = torch.tensor(math.pi, device=device, dtype=dtype)
    return _PI_TENSOR_CACHE[key]

# Global compilation cache with build hash versioning
_GLOBAL_COMPILE_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE_BYTES = 0
_CACHE_VERSION = (3, 14, torch.__version__, BUILD_HASH)

# Global inflight tracking
_GLOBAL_INFLIGHT = {}
_INFLIGHT_LOCK = threading.Lock()

# Track which signatures have already warned about timeout
_TIMEOUT_WARNED = set()

def clear_global_compile_cache():
    """Expose API to clear global compile cache"""
    global _GLOBAL_COMPILE_CACHE, _CACHE_SIZE_BYTES
    with _CACHE_LOCK:
        _GLOBAL_COMPILE_CACHE.clear()
        _CACHE_SIZE_BYTES = 0
    logger.info("Global compile cache cleared")

def _estimate_cache_size(obj) -> int:
    """Better cache size estimation with multi-GPU awareness"""
    try:
        if hasattr(obj, '__code__') and hasattr(obj.__code__, 'co_filename'):
            filename = obj.__code__.co_filename
            if os.path.exists(filename):
                return os.path.getsize(filename)
    except Exception:
        pass

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx).lower()
        if 'radeon' in device_name or 'amd' in device_name:
            return 12 * 1024 * 1024
        else:
            return 5 * 1024 * 1024
    else:
        return 1 * 1024 * 1024

def _make_stable_signature(obj) -> str:
    """Create stable signature for non-tensor objects"""
    try:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return json.dumps(obj)
        elif isinstance(obj, (list, tuple)):
            return json.dumps([_make_stable_signature(x) for x in obj])
        elif isinstance(obj, dict):
            return json.dumps({k: _make_stable_signature(v) for k, v in sorted(obj.items())})
        else:
            return repr(type(obj))
    except Exception:
        return str(type(obj))

def compile_function_fixed(mode="reduce-overhead", dynamic=True, max_cache_size=256,
                          max_cache_size_mb=1024, use_global_cache=True, enable_cpu_compile=None,
                          compile_timeout=30.0):
    """Thread-safe compilation with timeout and guaranteed event release."""
    def decorator(func):
        if not COMPILE_AVAILABLE:
            return func

        if enable_cpu_compile is None:
            cpu_compile = os.environ.get('FASTSBOT_CPU_COMPILE', 'false').lower() == 'true'
        else:
            cpu_compile = enable_cpu_compile

        if not (torch.cuda.is_available() or cpu_compile):
            return func

        if use_global_cache:
            cache = _GLOBAL_COMPILE_CACHE
            cache_lock = _CACHE_LOCK
            inflight = _GLOBAL_INFLIGHT
            inflight_lock = _INFLIGHT_LOCK
            base_func_key = f"{func.__module__}.{func.__qualname__}"
        else:
            if not hasattr(func, '_compiled_cache'):
                func._compiled_cache = OrderedDict()
                func._cache_lock = threading.Lock()
                func._cache_size_bytes = 0
                func._inflight = {}
                func._inflight_lock = threading.Lock()
            cache = func._compiled_cache
            cache_lock = func._cache_lock
            inflight = func._inflight
            inflight_lock = func._inflight_lock
            base_func_key = "local"

        @wraps(func)
        def wrapper(*args, **kwargs):
            global _CACHE_SIZE_BYTES
            compiled_fn = None
            need_compile = False
            evt = None

            def make_tensor_sig(arg):
                # Signature based on tensor metadata, not values (intentional for compile cache)
                if isinstance(arg, torch.Tensor):
                    try:
                        memory_format = "channels_last" if arg.is_contiguous(memory_format=torch.channels_last) else "default"
                    except (TypeError, AttributeError):
                        memory_format = "default"
                    return (arg.shape, arg.dtype, str(arg.device), memory_format, arg.requires_grad)
                return None

            args_tensor_sig = tuple(make_tensor_sig(arg) for arg in args)
            args_nontensor_sig = tuple(
                _make_stable_signature(arg) if not isinstance(arg, torch.Tensor) else None
                for arg in args
            )

            tensor_kwargs_sig = tuple(
                (k, make_tensor_sig(kwargs[k])) for k in sorted(kwargs)
                if isinstance(kwargs[k], torch.Tensor)
            )
            nontensor_kwargs_sig = tuple(
                (k, _make_stable_signature(kwargs[k])) for k in sorted(kwargs)
                if not isinstance(kwargs[k], torch.Tensor)
            )

            sig_core = (_CACHE_VERSION, base_func_key, args_tensor_sig, args_nontensor_sig,
                       tensor_kwargs_sig, nontensor_kwargs_sig)
            shape_sig_compiled = ("c",) + sig_core
            shape_sig_eager = ("e",) + sig_core

            with cache_lock:
                entry = cache.get(shape_sig_compiled) or cache.get(shape_sig_eager)
                if entry is not None:
                    key_to_use = shape_sig_compiled if shape_sig_compiled in cache else shape_sig_eager
                    compiled_fn, size = cache.pop(key_to_use)
                    cache[key_to_use] = (compiled_fn, size)

            if compiled_fn is not None:
                return compiled_fn(*args, **kwargs)

            with inflight_lock:
                evt = inflight.get(sig_core)
                if evt is None:
                    evt = threading.Event()
                    inflight[sig_core] = evt
                    need_compile = True

            if need_compile:
                estimated_size = _estimate_cache_size(func)
                compile_kwargs = {"mode": mode, "fullgraph": False}
                if _TORCH_COMPILE_ACCEPTS_DYNAMIC:
                    compile_kwargs["dynamic"] = dynamic

                # FIX: Initialize is_eager BEFORE try block to ensure it's defined in finally
                is_eager = True  # Assume eager mode by default

                # Wrap compilation in try/finally to guarantee event release
                try:
                    compiled_local = torch.compile(func, **compile_kwargs)
                    is_eager = False  # Only set to False if compilation succeeds
                except Exception as e:
                    logger.warning(f"torch.compile failed, falling back to eager mode: {e}")
                    compiled_local = func
                    is_eager = True
                    estimated_size = 0
                finally:
                    # Always update cache and release event
                    final_key = shape_sig_eager if is_eager else shape_sig_compiled
                    with cache_lock:
                        if final_key not in cache:
                            if not is_eager:
                                max_size_bytes = max_cache_size_mb * 1024 * 1024
                                if use_global_cache:
                                    while ((_CACHE_SIZE_BYTES + estimated_size > max_size_bytes) or
                                           (len(cache) >= max_cache_size)) and cache:
                                        _, old_size = cache.popitem(last=False)[1]
                                        _CACHE_SIZE_BYTES -= old_size
                                    _CACHE_SIZE_BYTES = max(0, _CACHE_SIZE_BYTES) + estimated_size
                                else:
                                    while ((func._cache_size_bytes + estimated_size > max_size_bytes) or
                                           (len(cache) >= max_cache_size)) and cache:
                                        _, old_size = cache.popitem(last=False)[1]
                                        func._cache_size_bytes -= old_size
                                    func._cache_size_bytes = max(0, func._cache_size_bytes) + estimated_size
                            cache[final_key] = (compiled_local, estimated_size)
                        compiled_fn = cache[final_key][0]

                    # Always release event
                    with inflight_lock:
                        inflight.pop(sig_core, None)
                    evt.set()
            else:
                # Add timeout to prevent indefinite blocking
                if not evt.wait(timeout=compile_timeout):
                    # Warn once per signature about timeout (thread-safe check)
                    global _TIMEOUT_WARNED
                    should_warn = False
                    with inflight_lock:  # Protect _TIMEOUT_WARNED access
                        if sig_core not in _TIMEOUT_WARNED:
                            _TIMEOUT_WARNED.add(sig_core)
                            should_warn = True

                    if should_warn:
                        logger.warning(f"Compilation timeout after {compile_timeout}s for signature {base_func_key}, "
                                       f"using eager mode. This warning will only appear once per signature.")

                    compiled_fn = func

                    # Cache eager stub to avoid re-spinning on timeout
                    with cache_lock:
                        cache[shape_sig_eager] = (func, 0)

                    # Atomically remove and signal event
                    with inflight_lock:
                        evt_to_signal = inflight.pop(sig_core, None)
                        if evt_to_signal is not None:
                            try:
                                evt_to_signal.set()
                            except Exception:
                                pass
                else:
                    with cache_lock:
                        entry = cache.get(shape_sig_compiled) or cache.get(shape_sig_eager)
                        if entry is None:
                            compiled_fn = func
                        else:
                            key_to_use = shape_sig_compiled if shape_sig_compiled in cache else shape_sig_eager
                            compiled_fn, size = cache.pop(key_to_use)
                            cache[key_to_use] = (compiled_fn, size)

            return compiled_fn(*args, **kwargs)

        return wrapper
    return decorator

def fastsbot_next_power_of_two(n):
    """Helper to find next power of two (renamed to avoid confusion with Triton's)"""
    return 1 << (n - 1).bit_length()

def get_optimal_block_size(n_elements, device_capability=None):
    """Hardware-aware block size selection with cached device properties"""
    if device_capability is None and torch.cuda.is_available():
        device_idx = torch.cuda.current_device()

        # Use cache to avoid repeated device queries
        if device_idx not in _CACHED_DEVICE_PROPERTIES:
            _CACHED_DEVICE_PROPERTIES[device_idx] = torch.cuda.get_device_capability(device_idx)
        device_capability = _CACHED_DEVICE_PROPERTIES[device_idx]

    if device_capability and device_capability[0] >= 9:
        max_block_size = 1024
    else:
        max_block_size = 512

    block_size = min(max_block_size, max(32, fastsbot_next_power_of_two(n_elements) // 4))
    return block_size

# Triton kernels with proper imports and caching
if TRITON_AVAILABLE:
    triton, tl, triton_next_power_of_two, tl_sigmoid = check_triton_availability()[1]

    @triton.jit
    def fused_drift_transport_kernel_fixed(
        x_ptr, drift_ptr, out_ptr,
        scale: tl.float32,  # FIX: Pass as scalar value, not pointer
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """Proper bounds checking prevents race conditions

        CRITICAL FIX: scale is now a scalar value, not a pointer.
        This prevents segfaults when passing Python floats or scalars.

        MATHEMATICAL NOTE: Sigmoid gating logic reconsidered.
        The sigmoid of |drift| always outputs [0.5, 1.0] after clamping,
        meaning we always apply at least 50% of the drift.
        This is intentional for stable transport - see documentation.
        """
        # Local helpers to avoid monkey-patching
        wmin = lambda a, b: tl.where(a < b, a, b)
        wmax = lambda a, b: tl.where(a > b, a, b)
        wabs = lambda x: tl.where(x >= 0, x, -x)

        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mask = offsets < N

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        drift = tl.load(drift_ptr + offsets, mask=mask, other=0.0)

        # FIX: Use scale directly as a value (no tl.load needed)
        scale_clamped = wmin(wmax(scale, 0.1), 10.0)
        drift_abs = wabs(drift)

        # Sigmoid gating: weight in [0.5, 0.95] after clamping
        # This provides adaptive transport based on drift magnitude
        weight = 1.0 / (1.0 + tl.exp(-drift_abs * scale_clamped))
        weight = wmin(wmax(weight, 0.05), 0.95)

        out = x + weight * drift

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def fisher_diagonal_kernel_fixed(
        score_ptr, out_ptr,
<<<<<<< Updated upstream
        alpha_value: tl.float32,  # runtime scalar (alpha_bar value)
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """Proper masking for Fisher diagonal computation

        MATHEMATICAL FIX: Fisher information diagonal is computed as score^2, not |score|.
        - Empirical Fisher: E[∇log p · ∇log p^T] ≈ (∇log p)^2 for diagonal
        - Using L1 norm (|score|) changes gradient scaling and is non-standard
        - L2 norm (score^2) matches standard Fisher preconditioning

        Adaptive epsilon:
        - High noise (alpha→0): score is unstable, need larger epsilon
        - Low noise (alpha→1): score is stable, smaller epsilon suffices
        - Formula: eps = 1e-4 + 1e-3 * (1 - alpha) implements this correctly
=======
        alpha_value,  # runtime scalar, not constexpr (we pass α_bar)
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """Fisher diagonal computation with proper score^2 (not |score|)

        MATHEMATICAL FIX: Fisher information diagonal is E[∇log p · ∇log p^T] ≈ (∇log p)^2
        - Using score^2 (element-wise squaring) is correct for empirical Fisher diagonal
        - Using |score| was non-standard and gave incorrect gradient scaling
>>>>>>> Stashed changes
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mask = offsets < N

        score = tl.load(score_ptr + offsets, mask=mask, other=0.0)
<<<<<<< Updated upstream

        # FIX: Use score^2 instead of |score| for proper Fisher Information
        # This matches the mathematical definition: F_ii = E[(∂log p/∂θ_i)^2]
        fisher_diag = score * score

        # Adaptive regularization based on noise level (alpha_bar)
        adaptive_eps = 1e-4 + 1e-3 * (1.0 - alpha_value)
        fisher = fisher_diag + adaptive_eps
=======
        # Adaptive regularization: ε = ε_base + ε_scale * (1 - α_bar)
        # - ε_base = 1e-4: minimum stability for well-conditioned regions
        # - ε_scale = 1e-3: empirically tuned for typical score ranges
        # - (1 - α_bar): noise level proxy (high noise → larger ε needed)
        adaptive_eps = 1e-4 + 1e-3 * (1.0 - alpha_value)
        fisher = score * score + adaptive_eps  # FIXED: Use score^2, not |score|
>>>>>>> Stashed changes

        tl.store(out_ptr + offsets, fisher, mask=mask)

    def launch_triton_kernel_safe(kernel, *args, n_elements, kernel_type="default",
                                 eps=1e-3, alpha=None):
        """Hardware-aware kernel launch with kernel type"""
        BLOCK_SIZE = min(512, triton_next_power_of_two(n_elements))

        device_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        if device_capability and device_capability[0] >= 8:
            num_warps = 8
            num_stages = 3
        else:
            num_warps = 4
            num_stages = 2

        grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

        if kernel_type == "fisher":
            alpha_value = float(alpha) if torch.is_tensor(alpha) else alpha
            kernel[(grid_size,)](
                *args, alpha_value=alpha_value, N=n_elements, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps, num_stages=num_stages
            )
        else:
            kernel[(grid_size,)](
                *args, N=n_elements, BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps, num_stages=num_stages
            )

def log_sum_exp_stabilized(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Use built-in numerically stable log-sum-exp"""
    return torch.logsumexp(x, dim=dim)


