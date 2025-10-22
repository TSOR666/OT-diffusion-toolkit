from abc import ABC, abstractmethod

import torch


class KernelOperator(ABC):
    """Abstract base class for kernel operators in the RKHS formulation."""

    def __init__(self, epsilon: float, device: torch.device) -> None:
        self.epsilon = epsilon
        self.device = device

    @abstractmethod
    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply the kernel operator to input vector(s)."""

    @abstractmethod
    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply the transpose of the kernel operator to input vector(s)."""

    @abstractmethod
    def get_error_bound(self, n_samples: int) -> float:
        """Return an estimate of the approximation error for a given sample count."""

    def clear_cache(self) -> None:  # pragma: no cover - default no-op
        """Clear any cached computations to free memory."""
        return None
