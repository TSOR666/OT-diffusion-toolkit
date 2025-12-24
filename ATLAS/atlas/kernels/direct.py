from typing import Optional

import torch

from .base import KernelOperator


class DirectKernelOperator(KernelOperator):
    """
    Direct implementation of a kernel operator using the explicit kernel matrix.

    Memory complexity O(n^2) and compute O(n^2), best suited for small batches.
    """

    def __init__(
        self,
        kernel_type: str = "gaussian",
        epsilon: float = 0.01,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(epsilon, device)
        self.kernel_type = kernel_type
        self.kernel_matrix: Optional[torch.Tensor] = None
        self.last_x_shape: Optional[torch.Size] = None

    def _squared_distance(self, x_flat: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
        """Compute squared Euclidean distance ||x - y||^2 with clamping for stability."""
        x_norm = (x_flat ** 2).sum(1, keepdim=True)
        y_norm = (y_flat ** 2).sum(1, keepdim=True)
        xy = x_flat @ y_flat.t()  # (n, d) @ (d, m) -> (n, m)
        dist_sq = x_norm + y_norm.t() - 2 * xy
        return torch.clamp(dist_sq, min=0.0)

    def _compute_kernel_matrix(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        if y is None:
            y = x
        else:
            y = y.to(self.device)

        if self.kernel_type == "gaussian":
            x_flat = x.reshape(x.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)

            dist_sq = self._squared_distance(x_flat, y_flat)
            sigma_sq = self.epsilon ** 2
            # Two-sided clamp: max=0 ensures no positive exponents from numerical errors
            exponent = torch.clamp(-dist_sq / (2 * sigma_sq), min=-50.0, max=0.0)
            return torch.exp(exponent)

        if self.kernel_type == "laplacian":
            x_flat = x.reshape(x.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)
            dist = torch.cdist(x_flat, y_flat, p=2)
            # Two-sided clamp: max=0 ensures no positive exponents from numerical errors
            exponent = torch.clamp(-dist / self.epsilon, min=-50.0, max=0.0)
            return torch.exp(exponent)

        if self.kernel_type == "cauchy":
            x_flat = x.reshape(x.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)

            dist_sq = self._squared_distance(x_flat, y_flat)
            denom = 1.0 + dist_sq / (self.epsilon ** 2)
            denom = torch.clamp(denom, min=1e-12)
            return 1.0 / denom

        raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def setup(self, x: torch.Tensor) -> None:
        needs_setup = (
            self.kernel_matrix is None
            or x.shape != self.last_x_shape
            or (self.kernel_matrix is not None and self.kernel_matrix.device != self.device)
        )
        if (
            needs_setup
        ):
            self.kernel_matrix = self._compute_kernel_matrix(x)
            self.last_x_shape = x.shape
            if __debug__:
                asym = (self.kernel_matrix - self.kernel_matrix.t()).abs().max()
                if asym > 1e-5:
                    raise RuntimeError(f"Kernel matrix not symmetric (max asymmetry={asym.item():.3e}).")

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.setup(x)

        if self.kernel_matrix is None:
            raise RuntimeError("Kernel matrix not initialized after setup")
        v = v.to(self.device)
        if v.shape[0] != self.kernel_matrix.shape[0]:
            raise ValueError(
                f"v batch size {v.shape[0]} does not match kernel size {self.kernel_matrix.shape[0]}"
            )
        if v.dim() == 1:
            return self.kernel_matrix @ v
        v_shape = v.shape
        v_flat = v.reshape(v.shape[0], -1)
        result_flat = self.kernel_matrix @ v_flat
        return result_flat.reshape(v_shape)

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.apply(x, v)

    def get_error_bound(self, n_samples: int) -> float:
        return 0.0

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return explicit kernel matrix between x and y."""
        return self._compute_kernel_matrix(x, y)

    def clear_cache(self) -> None:
        self.kernel_matrix = None
        self.last_x_shape = None
