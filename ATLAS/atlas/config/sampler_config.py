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

    def with_overrides(self, **kwargs):
        """Return a copy with specific fields overridden."""

        return replace(self, **kwargs)
