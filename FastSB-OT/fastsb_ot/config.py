"""Configuration primitives for the FastSB-OT solver."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Literal

import torch

from . import common

logger = common.logger
Version = common.Version
NUMPY_AVAILABLE = common.NUMPY_AVAILABLE
_CACHED_DEVICE_PROPERTIES = common._CACHED_DEVICE_PROPERTIES
np = getattr(common, "np", None)

__all__ = ["QualityPreset", "FastSBOTConfig"]


class QualityPreset(Enum):
    """Quality presets for different speed/quality tradeoffs"""
    DRAFT = "draft"
    BALANCED = "balanced"
    ULTRA = "ultra"
    EXTREME = "extreme"


@dataclass
class FastSBOTConfig:
    """Configuration for the FastSB-OT Solver with enhanced sampling controls.

    Key parameters:
        eps: OT regularization parameter
        guidance_scale: Score scaling factor (1.0-2.0 recommended, NOT CFG)
        use_ddim_sampling: Use DDIM instead of DDPM
        ddim_eta: Stochasticity for DDIM (0=deterministic, 1=DDPM)
        discrete_timesteps: Whether model expects discrete timestep indices
        use_score_correction: Apply curvature-based score correction
        use_dynamic_thresholding: Apply dynamic thresholding to predictions
    """
    # Basic parameters
    eps: float = 0.01
    adaptive_eps: bool = True

    # Resolution handling
    progressive_generation: bool = True
    patch_based: bool = True
    max_patch_size: int = 512
    patch_overlap_ratio: float = 0.25
    adaptive_patch_size: bool = True

    # Training patch size override
    training_max_patch_size: Optional[int] = None

    # Computational methods
    quality: str = "balanced"
    method: Literal["compiled", "iterative", "hybrid", "fisher"] = "hybrid"

    # Memory optimization
    memory_efficient: bool = True
    use_mixed_precision: bool = True
    use_bfloat16: bool = False
    max_batch_per_device: Optional[int] = None
    cache_size_mb: int = 1024
    max_cache_entries: int = 50
    memory_limit_ot_mb: int = 100
    legacy_transport_mode: bool = False

    # Quality enhancements
    corrector_steps: int = field(init=False)
    corrector_snr: float = 0.25
    freq_weighting: bool = True

    # Mathematical enhancements
    use_momentum_transport: bool = True
    momentum_beta: float = 0.9
    use_hierarchical_bridge: bool = True
    hierarchical_scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    use_fisher_geometry: bool = True
    use_adaptive_langevin: bool = True
    control_variate_strength: float = 0.1

    # Critical compilation and precision settings
    use_fp32_fisher: bool = True
    use_fp32_time: bool = True
    batch_ot_method: str = "sliced"
    sliced_ot_projections: int = 100

    # Enhanced compilation settings
    seed: Optional[int] = None
    warmup: bool = True
    compile_mode: str = "reduce-overhead"
    use_triton_kernels: bool = True
    enable_cpu_compile: bool = False
    use_dynamic_compilation: bool = True
    max_compiled_shapes: int = 256
    compile_timeout: float = 30.0

    # Cache management
    cuda_cache_flush_watermark: float = 0.8
    global_compile_cache: bool = True
    global_cache_size_mb: int = 1024
    cuda_cache_flush_threshold_mb: int = 32

    # Advanced parameters
    landmark_ratio: float = 0.1
    sinkhorn_tolerance: float = 1e-6
    critical_alpha_thresholds: List[float] = field(default_factory=lambda: [0.9, 0.5, 0.1])
    sliced_ot_projection_fn: Optional[Callable[[int, int], int]] = None

    # Transport parameters
    transport_gamma: float = 0.5
    transport_weight_min: float = 0.2
    transport_weight_max: float = 0.8
    ot_eps_min: float = 5e-3

    # Operational settings
    enable_profiling: bool = False
    profiler_trace_path: Optional[str] = None
    log_level: str = "INFO"

    # Deterministic RNG hook
    generator: Optional[torch.Generator] = None

    # ENHANCED SAMPLING PARAMETERS
    # Sampling improvements
    variance_preserving_sampling: bool = True
    use_score_correction: bool = False
    use_learned_variance: bool = False
    use_dynamic_thresholding: bool = True
    dynamic_thresholding_percentile: float = 0.995
    dynamic_thresholding_adaptive_floor: bool = True  # POLISH: Added adaptive floor option
    discrete_timesteps: bool = False  # Set True if model expects discrete timesteps
    num_timesteps: int = 1000  # Total number of training timesteps

    # DDIM sampling
    use_ddim_sampling: bool = True
    ddim_eta: float = 0.0  # 0 for deterministic, 1 for DDPM

    # Guidance
    guidance_scale: float = 1.0  # Score direction scaling (NOT CFG), use 1.0-2.0 for best results
    guidance_mode: str = "score"  # POLISH: "score" or "noise" mode

    # Improved noise schedule
    use_cosine_schedule: bool = False
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Internal fields
    _sinkhorn_iterations: int = field(init=False, default=50)

    def __post_init__(self):
        """Set parameters based on quality preset and apply seed"""
        self._apply_quality_preset()
        self._apply_hardware_config()
        self._apply_seed()

    def _apply_quality_preset(self):
        """Apply quality preset settings"""
        presets = {
            "draft": (0, 10),
            "balanced": (0, 50),
            "ultra": (1, 50),
            "extreme": (2, 100)
        }

        if self.quality in presets:
            self.corrector_steps, sinkhorn_iterations = presets[self.quality]
        else:
            self.corrector_steps, sinkhorn_iterations = presets["balanced"]

        self._sinkhorn_iterations = sinkhorn_iterations

    def _apply_hardware_config(self):
        """Apply hardware-specific adjustments with memory limits"""
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            if device_id not in _CACHED_DEVICE_PROPERTIES:
                _CACHED_DEVICE_PROPERTIES[device_id] = torch.cuda.get_device_properties(device_id)

            props = _CACHED_DEVICE_PROPERTIES[device_id]
            capability = props.major, props.minor
            total_memory = props.total_memory

            # Only use channels_last on Ampere+ (capability >= 8.0)
            self.use_channels_last = capability[0] >= 8

            if capability[0] >= 8 and self.quality in ["ultra", "extreme"]:
                self.use_bfloat16 = True

            legacy_gpu = capability[0] < 7
            if legacy_gpu:
                # Triton kernels require Volta+ for stable performance
                self.use_triton_kernels = False
                # Keep hierarchical bridge available but force safer precision later
                self.legacy_transport_mode = True
            else:
                self.legacy_transport_mode = False

            if self.adaptive_patch_size:
                if total_memory < 8e9:
                    self.max_patch_size = min(self.max_patch_size, 256)
                    self.memory_limit_ot_mb = 50
                elif total_memory < 16e9:
                    self.max_patch_size = min(self.max_patch_size, 384)
                    self.memory_limit_ot_mb = 100
                else:
                    self.memory_limit_ot_mb = 200
        else:
            self.use_channels_last = False
            self.legacy_transport_mode = True

    def _apply_seed(self):
        """Apply seed for reproducibility with optional deterministic RNG"""
        if self.seed is not None:
            random.seed(self.seed)
            # Only seed NumPy if it's available
            if NUMPY_AVAILABLE:
                np.random.seed(self.seed)
            torch.manual_seed(self.seed)

            if self.generator is None:
                try:
                    # Create CPU generator by default; solver will migrate to device as needed
                    self.generator = torch.Generator()
                except Exception:
                    self.generator = None
            if self.generator is not None:
                try:
                    self.generator.manual_seed(self.seed)
                except Exception:
                    pass

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

                deterministic = os.environ.get("FASTSBOT_DETERMINISTIC", "0") == "1"
                torch.backends.cudnn.deterministic = deterministic
                torch.backends.cudnn.benchmark = not deterministic

                # Safer deterministic algorithms - wrap with try/except
                if deterministic and hasattr(torch, 'use_deterministic_algorithms'):
                    try:
                        torch.use_deterministic_algorithms(True)
                    except Exception as e:
                        logger.warning(f"Deterministic algorithms requested but not fully available: {e}. "
                                       f"Some operations (FFT, Triton) may not be deterministic.")


