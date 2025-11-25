from dataclasses import dataclass, field, replace
from typing import List
import warnings


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
        if self.epsilon < 1e-6:
            warnings.warn(
                f"Very small epsilon ({self.epsilon}) may cause numerical instability",
                UserWarning,
                stacklevel=2,
            )
        if self.epsilon > 100:
            warnings.warn(
                f"Very large epsilon ({self.epsilon}) may over-smooth results",
                UserWarning,
                stacklevel=2,
            )
        if self.rff_features <= 0:
            raise ValueError(f"rff_features must be positive, got {self.rff_features}")
        if self.rff_features > 32768:
            warnings.warn(
                f"Very large rff_features ({self.rff_features}) may cause memory issues",
                ResourceWarning,
                stacklevel=2,
            )
        if self.n_landmarks <= 0:
            raise ValueError(f"n_landmarks must be positive, got {self.n_landmarks}")
        if self.n_landmarks > 10000:
            warnings.warn(
                f"Very large n_landmarks ({self.n_landmarks}) may be slow or memory intensive",
                ResourceWarning,
                stacklevel=2,
            )
        if self.max_kernel_cache_size <= 0:
            raise ValueError(f"max_kernel_cache_size must be positive, got {self.max_kernel_cache_size}")
        if self.kernel_type not in ["gaussian", "laplacian", "cauchy"]:
            raise ValueError(f"kernel_type must be 'gaussian', 'laplacian', or 'cauchy', got {self.kernel_type}")
        if self.solver_type not in ["auto", "direct", "rff", "nystrom", "fft"]:
            raise ValueError(f"solver_type must be 'auto', 'direct', 'rff', 'nystrom', or 'fft', got {self.solver_type}")
        if not self.scale_factors:
            raise ValueError("scale_factors cannot be empty.")
        if not all(sf > 0 for sf in self.scale_factors):
            raise ValueError(f"All scale_factors must be positive, got {self.scale_factors}")
        if len(self.scale_factors) != len(set(self.scale_factors)):
            warnings.warn(
                "scale_factors contains duplicates, which is inefficient.",
                UserWarning,
                stacklevel=2,
            )
        if self.multi_scale:
            total_features = self.rff_features * len(self.scale_factors)
            if total_features > 65536:
                warnings.warn(
                    f"Multi-scale RFF uses {total_features} total features "
                    f"({self.rff_features} x {len(self.scale_factors)}); this may cause memory issues.",
                    ResourceWarning,
                    stacklevel=2,
                )
        if self.orthogonal and self.solver_type not in ["auto", "rff"]:
            warnings.warn(
                f"orthogonal=True only applies to RFF solver; solver_type={self.solver_type} will ignore it.",
                UserWarning,
                stacklevel=2,
            )

    def with_overrides(self, **kwargs):
        """Return a copy with specific fields overridden."""
        new_cfg = replace(self, **kwargs)
        new_cfg.validate()
        return new_cfg
