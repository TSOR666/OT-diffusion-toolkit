from abc import ABC, abstractmethod

import torch


class KernelOperator(ABC):
    """Abstract base class for kernel operators in the RKHS formulation."""

    def __init__(self, epsilon: float, device: torch.device, is_symmetric: bool = True) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = float(epsilon)
        self.device = device
        self.is_symmetric = bool(is_symmetric)

    @abstractmethod
    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply the kernel operator (e.g., K @ v).

        Args:
            x: Input locations, shape [n, ...].
            v: Vectors to transform, shape [n, *].

        Returns:
            Transformed vectors with leading batch dimension n.
        """

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose of the kernel operator.

        For symmetric kernels (default), this is identical to ``apply``.
        Override for asymmetric approximations.
        """
        if self.is_symmetric:
            return self.apply(x, v)
        raise NotImplementedError(f"{self.__class__.__name__} must override apply_transpose for asymmetric kernels.")

    @abstractmethod
    def get_error_bound(self, n_samples: int) -> float:
        """Return an estimate of the approximation error for a given sample count."""

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # pragma: no cover - optional override
        """Compute pairwise kernel evaluations K(x, y)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement pairwise kernel evaluation."
        )

    def clear_cache(self) -> None:  # pragma: no cover - default no-op
        """Clear any cached computations to free memory."""
        return None

    def _validate_device(self, *tensors: torch.Tensor) -> None:
        """Ensure all provided tensors reside on the operator device."""
        for idx, tensor in enumerate(tensors):
            if tensor.device != self.device:
                raise RuntimeError(
                    f"Tensor {idx} is on {tensor.device}, expected {self.device} for {self.__class__.__name__}."
                )
