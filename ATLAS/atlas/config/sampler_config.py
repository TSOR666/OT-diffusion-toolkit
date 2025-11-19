from dataclasses import dataclass, field, replace
from typing import List, Optional


@dataclass
class SamplerConfig:
    """Configuration for the diffusion sampler pipeline."""

    # General sampling parameters
    sb_iterations: int = 3
    error_tolerance: float = 1e-4
    use_linear_solver: bool = True
    use_mixed_precision: bool = True
    memory_efficient: bool = True
    verbose_logging: bool = False
    guidance_scale: float = 1.0
    denoise_final: bool = True

    # Hierarchical sampling parameters
    hierarchical_sampling: bool = True
    dynamic_weighting: str = "adaptive"
    interpolation_order: int = 3
    critical_thresholds: List[float] = field(
        default_factory=lambda: [0.95, 0.8, 0.6, 0.4, 0.2, 0.05]
    )
    patch_based_processing: bool = True

    # Auto-tuning parameters
    auto_tuning: bool = False
    max_cached_batch_size: int = 16

    # Memory management
    memory_threshold_mb: float = 8192.0

    # High resolution helpers
    enable_cuda_graphs: bool = False
    cuda_graph_warmup_iters: int = 2
    tile_size: Optional[int] = None
    tile_stride: Optional[int] = None
    tile_overlap: float = 0.125
    tile_blending: str = "hann"

    # Reproducibility
    seed: Optional[int] = None
    cg_relative_tolerance: float = 1e-5
    cg_absolute_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.sb_iterations <= 0:
            raise ValueError(f"sb_iterations must be positive, got {self.sb_iterations}")
        if self.error_tolerance <= 0:
            raise ValueError(f"error_tolerance must be positive, got {self.error_tolerance}")
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale must be non-negative, got {self.guidance_scale}")
        if self.interpolation_order < 0:
            raise ValueError(f"interpolation_order must be non-negative, got {self.interpolation_order}")
        if self.max_cached_batch_size <= 0:
            raise ValueError(f"max_cached_batch_size must be positive, got {self.max_cached_batch_size}")
        if self.memory_threshold_mb <= 0:
            raise ValueError(f"memory_threshold_mb must be positive, got {self.memory_threshold_mb}")
        if self.dynamic_weighting not in ["uniform", "adaptive", "exponential"]:
            raise ValueError(f"dynamic_weighting must be 'uniform', 'adaptive', or 'exponential', got {self.dynamic_weighting}")
        if not all(0 <= t <= 1 for t in self.critical_thresholds):
            raise ValueError(f"All critical_thresholds must be in [0, 1], got {self.critical_thresholds}")
        if self.critical_thresholds and self.critical_thresholds != sorted(self.critical_thresholds, reverse=True):
            raise ValueError("critical_thresholds must be in descending order.")
        if self.cuda_graph_warmup_iters < 0:
            raise ValueError("cuda_graph_warmup_iters must be non-negative")
        if self.tile_size is not None and self.tile_size <= 0:
            raise ValueError("tile_size must be positive when set")
        if self.tile_stride is not None and self.tile_stride <= 0:
            raise ValueError("tile_stride must be positive when set")
        if not (0.0 <= self.tile_overlap < 1.0):
            raise ValueError("tile_overlap must be in [0, 1)")
        if self.tile_blending != "none" and self.tile_overlap <= 0:
            raise ValueError(
                "tile_overlap must be positive when tile_blending is enabled."
            )
        if self.tile_blending not in {"hann", "linear", "none"}:
            raise ValueError("tile_blending must be 'hann', 'linear', or 'none'")
        if self.cg_relative_tolerance <= 0:
            raise ValueError("cg_relative_tolerance must be positive")
        if self.cg_absolute_tolerance < 0:
            raise ValueError("cg_absolute_tolerance must be non-negative")

    def with_overrides(self, **kwargs):
        """Return a copy with specific fields overridden."""
        return replace(self, **kwargs)
