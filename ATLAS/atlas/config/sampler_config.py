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

    # Reproducibility
    seed: Optional[int] = None

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

    def with_overrides(self, **kwargs):
        """Return a copy with specific fields overridden."""
        return replace(self, **kwargs)
