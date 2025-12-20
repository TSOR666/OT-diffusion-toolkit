"""Configuration objects for the SPOT solver."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

__all__ = ["SolverConfig"]


@dataclass
class SolverConfig:
    """Configuration with sensible defaults."""

    eps: float = 0.1  # Base entropic regularization; used as the floor/start for adaptive scaling
    adaptive_eps: bool = True  # If True, scale eps dynamically; uses eps as the baseline
    adaptive_eps_scale: str = "sigma"  # 'sigma' | 'data' | 'none'
    sinkhorn_iterations: int = 20  # Increased from default 10 for better convergence
    dpm_solver_order: int = 3  # Increased from default 2 for higher accuracy
    error_tolerance: float = 1e-5  # Tightened from 1e-4 for better precision
    use_pot_library: bool = True
    use_mixed_precision: bool = True
    richardson_extrapolation: bool = True
    richardson_threshold: float = 0.05  # Lowered from 0.1 for more careful extrapolation
    richardson_max_overhead: float = 0.5
    max_tensor_size_elements: int = 50_000_000  # ~200MB per tensor (conservative for 12GB GPUs)
    max_dense_matrix_elements: int = 50_000_000  # ~200MB cost matrix (forces blockwise for N×M > 50M)
    use_patch_based_ot: bool = True
    patch_size: int = 32  # Reduced from 64; monitor for tiling artifacts and consider 64 for more global context
    highres_patch_size: int = 64  # Reduced from 96 for better patch coverage on high-res images
    ultrares_patch_size: int = 96  # Reduced from 128 for ultra-high-res images
    auto_tune_highres: bool = True
    force_per_pixel_b1: bool = False
    timestep_shape_b1: bool = False
    blockwise_threshold: int = 1_000_000
    highres_blockwise_threshold: int = 4_000_000
    ultrares_blockwise_threshold: int = 9_000_000
    deterministic: bool = False
    deterministic_cdist_cpu: bool = False
    enable_tf32: bool = False
    max_fallbacks: int = 10000
    profile_memory: bool = False

    # GPU Acceleration options
    use_triton: bool = True  # Use Triton kernels for GPU acceleration when available
    use_cuda_graphs: bool = False  # Experimental: Use CUDA graphs to reduce kernel launch overhead
    use_channels_last: bool = True  # Prefer NHWC memory format for large images

    # Advanced integrator options
    integrator: str = "dpm_solver++"  # 'dpm_solver++' | 'heun' | 'ddim' | 'adaptive' | 'exponential'
    heun_enabled: bool = False  # Deprecated; use integrator='heun' instead
    ddim_eta: float = 0.0  # DDIM eta parameter (0=deterministic, 1=DDPM-like)
    adaptive_atol: float = 1e-5  # Absolute tolerance for adaptive integrator
    adaptive_rtol: float = 1e-3  # Relative tolerance for adaptive integrator

    # Corrector options
    use_corrector: bool = False  # Enable predictor-corrector sampling
    corrector_type: str = "langevin"  # 'langevin' | 'tweedie' | 'adaptive'
    corrector_steps: int = 1  # Number of corrector steps per predictor step
    langevin_snr: float = 0.16  # SNR for Langevin corrector
    tweedie_mixing: float = 0.5  # Mixing coefficient for Tweedie corrector
    adaptive_corrector_threshold: float = 0.1  # Error threshold for adaptive corrector

    # Precision / compilation controls
    autocast_precision: str = "bf16"  # 'bf16' | 'fp16' | 'fp32' | 'none'
    compile_score_model: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    compile_warmup: bool = True

    def __post_init__(self) -> None:
        if self.heun_enabled:
            warnings.warn(
                "heun_enabled is deprecated; set integrator='heun' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def validate(self) -> None:
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        if self.sinkhorn_iterations <= 0:
            raise ValueError(f"sinkhorn_iterations must be positive, got {self.sinkhorn_iterations}")
        if self.error_tolerance <= 0:
            raise ValueError("error_tolerance must be positive")
        if self.dpm_solver_order not in [1, 2, 3]:
            raise ValueError("DPM-Solver order must be 1-3")
        if self.blockwise_threshold <= 0:
            raise ValueError("blockwise_threshold must be positive")
        if self.highres_blockwise_threshold <= 0:
            raise ValueError("highres_blockwise_threshold must be positive")
        if self.ultrares_blockwise_threshold <= 0:
            raise ValueError("ultrares_blockwise_threshold must be positive")
        if not 0 <= self.richardson_max_overhead <= 1:
            raise ValueError("richardson_max_overhead must be in [0, 1]")
        if self.adaptive_eps_scale not in ["sigma", "data", "none"]:
            raise ValueError("adaptive_eps_scale must be 'sigma', 'data', or 'none'")
        if self.autocast_precision not in ["bf16", "fp16", "fp32", "none"]:
            raise ValueError("autocast_precision must be 'bf16', 'fp16', 'fp32', or 'none'")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.highres_patch_size <= 0:
            raise ValueError("highres_patch_size must be positive")
        if self.ultrares_patch_size <= 0:
            raise ValueError("ultrares_patch_size must be positive")
        if self.max_fallbacks < 0:
            raise ValueError("max_fallbacks must be non-negative")
        if self.max_tensor_size_elements <= 0:
            raise ValueError("max_tensor_size_elements must be positive")
        if self.max_dense_matrix_elements <= 0:
            raise ValueError("max_dense_matrix_elements must be positive")
        if self.richardson_threshold < 0:
            raise ValueError("richardson_threshold must be non-negative")

        # Validate new options
        valid_integrators = ["dpm_solver++", "heun", "ddim", "adaptive", "exponential"]
        if self.integrator not in valid_integrators:
            raise ValueError(f"integrator must be one of {valid_integrators}")
        if self.heun_enabled and self.integrator != "heun":
            raise ValueError(
                f"Conflicting config: heun_enabled=True but integrator='{self.integrator}'. "
                "Set integrator='heun' instead."
            )

        valid_correctors = ["langevin", "tweedie", "adaptive"]
        if self.corrector_type not in valid_correctors:
            raise ValueError(f"corrector_type must be one of {valid_correctors}")

        if not 0 <= self.ddim_eta <= 1:
            raise ValueError("ddim_eta must be in [0, 1]")

        if self.corrector_steps < 0:
            raise ValueError("corrector_steps must be non-negative")

        if self.langevin_snr <= 0:
            raise ValueError("langevin_snr must be positive")

        if not 0 <= self.tweedie_mixing <= 1:
            raise ValueError("tweedie_mixing must be in [0, 1]")
