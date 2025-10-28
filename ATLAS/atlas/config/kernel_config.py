from dataclasses import dataclass, field, replace
from typing import List


@dataclass
class KernelConfig:
    """Configuration options for RKHS kernel operators."""

    kernel_type: str = "gaussian"
    epsilon: float = 0.01
    adaptive_epsilon: bool = True
    solver_type: str = "auto"
    rff_features: int = 2048
    n_landmarks: int = 100
    orthogonal: bool = True
    multi_scale: bool = True
    scale_factors: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    max_kernel_cache_size: int = 16

    def with_overrides(self, **kwargs):
        """Return a copy with specific fields overridden."""

        return replace(self, **kwargs)
