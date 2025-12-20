"""Random Fourier Feature helpers for SBDS."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class KernelDerivativeRFF(nn.Module):
    """Random Fourier Feature approximation supporting kernel derivatives."""

    # Type hints for registered buffers
    weights: torch.Tensor
    offset: torch.Tensor

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
        super().__init__()
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
        self.rademacher = rademacher
        self.orthogonal = orthogonal
        self.derivative_order = derivative_order

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        init_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights, offset = self._initialize_random_features(init_device)
        self.register_buffer("weights", weights)
        self.register_buffer("offset", offset)
        self.error_bound_factor = math.sqrt(
            math.log(max(input_dim * 100, 2)) / feature_dim
        )

    @property
    def device(self) -> torch.device:
        """Expose the current device (follows buffers when .to()/.cuda() are used)."""
        return self.weights.device

    def _initialize_random_features(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random Fourier features and phase offsets."""
        scale = 1.0 / self.sigma

        if self.orthogonal:
            num_blocks = self.feature_dim // self.input_dim
            remainder = self.feature_dim % self.input_dim
            blocks = []
            for _ in range(num_blocks):
                block = torch.randn(self.input_dim, self.input_dim, device=device)
                q, _ = torch.linalg.qr(block)
                blocks.append(q)
            if remainder > 0:
                block = torch.randn(self.input_dim, remainder, device=device)
                q, _ = torch.linalg.qr(block, mode="reduced")
                blocks.append(q)
            weights = torch.cat(blocks, dim=1) * scale
        else:
            if self.rademacher:
                weights = torch.randint(
                    0, 2, (self.input_dim, self.feature_dim), device=device
                )
                weights = (weights.float() * 2 - 1) * scale
            else:
                weights = torch.randn(self.input_dim, self.feature_dim, device=device) * scale

        offset = torch.rand(self.feature_dim, device=device) * 2 * math.pi
        return weights, offset

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return x.to(device=self.device, dtype=self.weights.dtype)

    def _feature_projections(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = self._flatten_input(x)
        projection = x_flat @ self.weights + self.offset
        return x_flat, projection

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = self._flatten_input(x)

        if x_flat.size(1) != self.input_dim:
            raise ValueError(
                f"Input dimension {x_flat.size(1)} does not match expected {self.input_dim}"
            )

        projection = x_flat @ self.weights + self.offset
        feature_scale = math.sqrt(2.0 / self.feature_dim)
        return torch.cos(projection) * feature_scale

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
        Compute kernel derivatives using RFF feature-space gradients.

        For φ(x) = sqrt(2/D) * cos(Wx + b):
        - First derivative: (∂φ(x)/∂x)^T φ(y)
        - Second derivative: Hessian of φ(x) contracted with φ(y)
        """
        if order < 1 or order > 2:
            raise NotImplementedError(f"Derivatives of order {order} not implemented")
        if order > self.derivative_order:
            raise ValueError(
                f"Requested derivative order {order} > supported order {self.derivative_order}"
            )

        x_flat, x_proj = self._feature_projections(x)
        if x_flat.size(1) != self.input_dim:
            raise ValueError(
                f"Input dimension {x_flat.size(1)} does not match expected {self.input_dim}"
            )

        feature_scale = math.sqrt(2.0 / self.feature_dim)
        x_cos = torch.cos(x_proj)
        x_sin = torch.sin(x_proj)

        if y is None:
            y_features = x_cos * feature_scale
        else:
            _, y_proj = self._feature_projections(y)
            y_features = torch.cos(y_proj) * feature_scale

        weights_t = self.weights.t()  # (D, d)

        if order == 1:
            grad_phi_x = -feature_scale * x_sin.unsqueeze(-1) * weights_t.unsqueeze(0)
            derivatives = torch.einsum("bjd,kj->dbk", grad_phi_x, y_features)

            if coordinate is not None:
                if isinstance(coordinate, tuple):
                    raise ValueError("coordinate for first-order derivative must be an int index")
                return derivatives[coordinate]

            return derivatives

        weight_outer = weights_t.unsqueeze(-1) * weights_t.unsqueeze(-2)  # (D, d, d)
        hess_phi_x = -feature_scale * x_cos.unsqueeze(-1).unsqueeze(-1) * weight_outer.unsqueeze(0)
        hessian = torch.einsum("bjdq,kj->dqbk", hess_phi_x, y_features)

        if coordinate is not None:
            if isinstance(coordinate, tuple):
                i_idx, j_idx = coordinate
            else:
                i_idx = j_idx = coordinate
            return hessian[i_idx, j_idx]

        return hessian

    def compute_score_approximation(
        self, x: torch.Tensor, y: torch.Tensor, weights_y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        feature_scale = math.sqrt(2.0 / self.feature_dim)

        x_flat, x_proj = self._feature_projections(x)
        _, y_proj = self._feature_projections(y)

        x_cos = torch.cos(x_proj)
        x_sin = torch.sin(x_proj)
        y_features = torch.cos(y_proj) * feature_scale

        if weights_y is not None:
            weights = weights_y.view(-1, 1).to(y_features.dtype)
            weight_sum = torch.clamp(weights.sum(), min=1e-8)
            phi_y_sum = (y_features * weights).sum(dim=0) / weight_sum
        else:
            phi_y_sum = y_features.mean(dim=0)

        phi_x = x_cos * feature_scale

        kernel_sum = torch.einsum("bd,d->b", phi_x, phi_y_sum).unsqueeze(-1) + 1e-10

        grad_phi_x = -feature_scale * x_sin.unsqueeze(-1) * self.weights.t().unsqueeze(0)
        numerator = torch.einsum("bjd,j->bd", grad_phi_x, phi_y_sum)

        return numerator / kernel_sum

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

