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

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.rff_features <= 0:
            raise ValueError(f"rff_features must be positive, got {self.rff_features}")
        if self.n_landmarks <= 0:
            raise ValueError(f"n_landmarks must be positive, got {self.n_landmarks}")
        if self.max_kernel_cache_size <= 0:
            raise ValueError(f"max_kernel_cache_size must be positive, got {self.max_kernel_cache_size}")
        if self.kernel_type not in ["gaussian", "laplacian", "cauchy"]:
            raise ValueError(f"kernel_type must be 'gaussian', 'laplacian', or 'cauchy', got {self.kernel_type}")
        if self.solver_type not in ["auto", "direct", "rff", "nystrom", "fft"]:
            raise ValueError(f"solver_type must be 'auto', 'direct', 'rff', 'nystrom', or 'fft', got {self.solver_type}")
        if not all(sf > 0 for sf in self.scale_factors):
            raise ValueError(f"All scale_factors must be positive, got {self.scale_factors}")

    def with_overrides(self, **kwargs):
        """Return a copy with specific fields overridden."""
        return replace(self, **kwargs)
