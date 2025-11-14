"""Random Fourier Feature helpers for SBDS."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch


class KernelDerivativeRFF:
    """Random Fourier Feature approximation supporting kernel derivatives."""

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 1024,
        sigma: float = 1.0,
        kernel_type: str = "gaussian",
        seed: Optional[int] = None,
        device: torch.device | None = None,
        rademacher: bool = False,
        orthogonal: bool = True,
        derivative_order: int = 1,
    ) -> None:
        # Input validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if derivative_order < 0:
            raise ValueError(f"derivative_order must be non-negative, got {derivative_order}")
        if kernel_type not in ["gaussian", "laplacian", "cauchy"]:
            raise ValueError(
                f"kernel_type must be one of ['gaussian', 'laplacian', 'cauchy'], "
                f"got '{kernel_type}'"
            )

        # RFF implementation only supports Gaussian kernels correctly
        # For Laplacian and Cauchy kernels, the spectral measure differs and requires
        # different sampling distributions and derivative formulas
        if kernel_type != "gaussian":
            raise NotImplementedError(
                f"Random Fourier Features currently only supports 'gaussian' kernel. "
                f"Kernel type '{kernel_type}' requires different spectral sampling: "
                f"Laplacian kernel needs Cauchy-distributed features, "
                f"Cauchy kernel needs Laplace-distributed features. "
                f"The derivative formulas also differ from the Gaussian case."
            )

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.sigma = sigma
        self.kernel_type = kernel_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rademacher = rademacher
        self.orthogonal = orthogonal
        self.derivative_order = derivative_order

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._initialize_random_features()
        self.error_bound_factor = math.sqrt(
            math.log(max(input_dim * 100, 2)) / feature_dim
        )

    def _initialize_random_features(self) -> None:
        # Gaussian kernel: spectral measure is also Gaussian, scale by 1/
        scale = 1.0 / self.sigma

        if self.orthogonal:
            num_blocks = self.feature_dim // self.input_dim
            remainder = self.feature_dim % self.input_dim
            blocks = []
            for _ in range(num_blocks):
                block = torch.randn(self.input_dim, self.input_dim, device=self.device)
                q, _ = torch.linalg.qr(block)
                blocks.append(q)
            if remainder > 0:
                block = torch.randn(self.input_dim, remainder, device=self.device)
                q, _ = torch.linalg.qr(block, mode="reduced")
                blocks.append(q)
            weights = torch.cat(blocks, dim=1) * scale
        else:
            if self.rademacher:
                weights = torch.randint(
                    0, 2, (self.input_dim, self.feature_dim), device=self.device
                )
                weights = (weights.float() * 2 - 1) * scale
            else:
                weights = torch.randn(self.input_dim, self.feature_dim, device=self.device) * scale

        offset = torch.rand(self.feature_dim, device=self.device) * 2 * math.pi
        self.weights = weights
        self.offset = offset

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)

        if x.size(1) != self.input_dim:
            raise ValueError(
                f"Input dimension {x.size(1)} does not match expected {self.input_dim}"
            )

        projection = x @ self.weights + self.offset
        return torch.cos(projection) * math.sqrt(2.0 / self.feature_dim)

    def compute_kernel(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if y is not None and y.dim() > 2:
            y = y.reshape(y.size(0), -1)

        x_features = self.compute_features(x)
        if y is None:
            return x_features @ x_features.T

        y_features = self.compute_features(y)
        return x_features @ y_features.T

    def compute_kernel_derivative(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        order: int = 1,
        coordinate: Optional[int | tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute kernel derivatives using Gaussian kernel derivative formulas.
        For order=1: k(x,y)/x_i = k(x,y) * (y_i - x_i) / 2
        For order=2: 2k(x,y)/x_ix_j = k(x,y) * (y_i - x_i)(y_j - x_j) / 4 - _ij * k(x,y) / 2
        """
        if order > self.derivative_order:
            raise ValueError(
                f"Requested derivative order {order} > supported order {self.derivative_order}"
            )

        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if y is not None and y.dim() > 2:
            y = y.reshape(y.size(0), -1)
        else:
            y = x

        if order == 1:
            kernel = self.compute_kernel(x, y)
            if coordinate is not None:
                diff = y[:, coordinate].unsqueeze(0) - x[:, coordinate].unsqueeze(1)
                return kernel * diff / (self.sigma**2)

            derivatives = []
            for idx in range(self.input_dim):
                diff_i = y[:, idx].unsqueeze(0) - x[:, idx].unsqueeze(1)
                derivatives.append(kernel * diff_i / (self.sigma**2))
            return torch.stack(derivatives, dim=0)

        if order == 2:
            kernel = self.compute_kernel(x, y)
            if coordinate is not None:
                if isinstance(coordinate, tuple):
                    i_idx, j_idx = coordinate
                else:
                    i_idx = j_idx = coordinate
                diff_i = y[:, i_idx].unsqueeze(0) - x[:, i_idx].unsqueeze(1)
                diff_j = y[:, j_idx].unsqueeze(0) - x[:, j_idx].unsqueeze(1)
                hessian = kernel * (diff_i * diff_j / (self.sigma**4))
                if i_idx == j_idx:
                    hessian = hessian - kernel / (self.sigma**2)
                return hessian

            hessian = torch.zeros(
                self.input_dim,
                self.input_dim,
                x.size(0),
                y.size(0),
                device=self.device,
            )
            for i_idx in range(self.input_dim):
                for j_idx in range(self.input_dim):
                    diff_i = y[:, i_idx].unsqueeze(0) - x[:, i_idx].unsqueeze(1)
                    diff_j = y[:, j_idx].unsqueeze(0) - x[:, j_idx].unsqueeze(1)
                    hessian[i_idx, j_idx] = kernel * (diff_i * diff_j / (self.sigma**4))
                    if i_idx == j_idx:
                        hessian[i_idx, j_idx] = hessian[i_idx, j_idx] - kernel / (self.sigma**2)
            return hessian

        raise NotImplementedError(f"Derivatives of order {order} not implemented")

    def compute_score_approximation(
        self, x: torch.Tensor, y: torch.Tensor, weights_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kernel = self.compute_kernel(x, y)
        derivatives = self.compute_kernel_derivative(x, y, order=1)

        if weights_y is not None:
            kernel = kernel * weights_y.unsqueeze(0)
            derivatives = derivatives * weights_y.unsqueeze(0).unsqueeze(0)

        kernel_sum = kernel.sum(dim=1, keepdim=True) + 1e-10
        score = torch.zeros(x.size(0), self.input_dim, device=self.device)
        for idx in range(self.input_dim):
            score[:, idx] = derivatives[idx].sum(dim=1) / kernel_sum.squeeze(1)
        return score

    def estimate_error_bound(self, n_samples: int) -> Dict[str, float]:
        """
        Estimate theoretical error bounds for the random Fourier feature approximation.

        The deviation between the exact kernel matrix and its RFF surrogate concentrates
        at a rate proportional to sqrt(log(n) / D) with high probability, where D is the
        number of random features. This helper reports the leading-order constants for
        the kernel and its first two derivatives.

        Args:
            n_samples: Number of samples (used for concentration bound)

        Returns:
            Dictionary with error bounds for kernel and derivatives

        Raises:
            ValueError: If n_samples or feature_dim is invalid
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

        # Rahimi & Recht (2008) provide a high-probability deviation bound of the form
        # O(sqrt(log(1/delta) / D)). We expose a conservative version that folds the
        # sample-dependent union bound into the logarithmic term.
        log_term = math.log(max(2, n_samples)) + math.log(2.0)
        base_error = math.sqrt(2.0 * log_term / self.feature_dim)
        return {
            "kernel": base_error,
            "first_derivative": base_error / self.sigma,
            "second_derivative": base_error / (self.sigma**2),
        }

