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

    def _compute_kernel_matrix(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y is None:
            y = x

        if self.kernel_type == "gaussian":
            x_flat = x.reshape(x.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)

            x_norm = (x_flat ** 2).sum(1).view(-1, 1)
            y_norm = (y_flat ** 2).sum(1).view(1, -1)
            xy = x_flat @ y_flat.t()
            dist_sq = x_norm + y_norm - 2 * xy
            dist_sq = torch.clamp(dist_sq, min=0.0)
            return torch.exp(-dist_sq / (2 * self.epsilon))

        if self.kernel_type == "laplacian":
            x_flat = x.reshape(x.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)
            dist = torch.cdist(x_flat, y_flat, p=2)
            return torch.exp(-dist / self.epsilon)

        if self.kernel_type == "cauchy":
            x_flat = x.reshape(x.size(0), -1)
            y_flat = y.reshape(y.size(0), -1)

            x_norm = (x_flat ** 2).sum(1).view(-1, 1)
            y_norm = (y_flat ** 2).sum(1).view(1, -1)
            xy = x_flat @ y_flat.t()
            dist_sq = x_norm + y_norm - 2 * xy
            dist_sq = torch.clamp(dist_sq, min=0.0)

            return 1.0 / (1.0 + dist_sq / self.epsilon ** 2)

        raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def setup(self, x: torch.Tensor) -> None:
        if self.kernel_matrix is None or x.shape != self.last_x_shape:
            self.kernel_matrix = self._compute_kernel_matrix(x)
            self.last_x_shape = x.shape

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.kernel_matrix is None or x.shape != self.last_x_shape:
            self.setup(x)

        if self.kernel_matrix is None:
            raise RuntimeError("Kernel matrix not initialized after setup")
        if v.dim() > 1 and v.shape[0] == self.kernel_matrix.shape[0]:
            v_shape = v.shape
            v_flat = v.reshape(v.shape[0], -1)
            result_flat = self.kernel_matrix @ v_flat
            return result_flat.reshape(v_shape)
        return self.kernel_matrix @ v

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.kernel_matrix is None or x.shape != self.last_x_shape:
            self.setup(x)

        if self.kernel_matrix is None:
            raise RuntimeError("Kernel matrix not initialized after setup")
        if v.dim() > 1 and v.shape[0] == self.kernel_matrix.shape[0]:
            v_shape = v.shape
            v_flat = v.reshape(v.shape[0], -1)
            result_flat = self.kernel_matrix.t() @ v_flat
            return result_flat.reshape(v_shape)
        return self.kernel_matrix.t() @ v

    def get_error_bound(self, n_samples: int) -> float:
        return 1e-10

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return explicit kernel matrix between x and y."""
        return self._compute_kernel_matrix(x, y)

    def clear_cache(self) -> None:
        self.kernel_matrix = None
        self.last_x_shape = None
