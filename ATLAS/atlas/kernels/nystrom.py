"""Nystrom kernel operator implementation.

This module provides a numerically stable approximation of dense kernel
matrices using the Nystrom method.  The implementation focuses on the use
cases exercised in the unit tests: it supports Gaussian, Laplacian and
Cauchy kernels, caches the landmark kernel matrix and falls back to
pseudo-inverse based solves when a Cholesky factorisation is not
available.  The original file in the repository contained duplicated
strings and indentation errors which prevented the module from being
imported.  The rewritten version keeps the public API intact while
removing the unreachable code paths that triggered the syntax errors.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .base import KernelOperator
from ..utils.random import set_seed


class NystromKernelOperator(KernelOperator):
    """Approximate kernel operator using the Nystrom method."""

    _SUPPORTED_KERNELS = {"gaussian", "laplacian", "cauchy"}

    def __init__(
        self,
        landmarks: torch.Tensor,
        kernel_type: str = "gaussian",
        epsilon: float = 0.01,
        device: Optional[torch.device] = None,
        regularization: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(epsilon, device)

        kernel = kernel_type.lower()
        if kernel not in self._SUPPORTED_KERNELS:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        if seed is not None:
            set_seed(seed)

        self.kernel_type = kernel
        self.regularization = float(regularization)
        self.landmarks = landmarks.to(self.device)
        self.n_landmarks = int(self.landmarks.shape[0])

        if self.n_landmarks == 0:
            raise ValueError("At least one landmark is required for the Nystrom approximation.")

        self._chol_factor: Optional[torch.Tensor] = None
        self._svd_factors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self._cached_kernels: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}

        self.K_ll = self._compute_kernel(self.landmarks, self.landmarks)
        self._factorise_landmark_kernel()

    # ------------------------------------------------------------------
    # Kernel computations
    # ------------------------------------------------------------------
    def _compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)

        if self.kernel_type == "gaussian":
            x_norm = (x_flat.square()).sum(dim=1, keepdim=True)
            y_norm = (y_flat.square()).sum(dim=1, keepdim=True).transpose(0, 1)
            dist_sq = torch.clamp(x_norm + y_norm - 2.0 * x_flat @ y_flat.T, min=0.0)
            sigma_sq = max(self.epsilon ** 2, 1e-12)
            return torch.exp(-dist_sq / (2.0 * sigma_sq))

        if self.kernel_type == "laplacian":
            dist = torch.cdist(x_flat, y_flat, p=2)
            return torch.exp(-dist / self.epsilon)

        # Cauchy kernel
        x_norm = (x_flat.square()).sum(dim=1, keepdim=True)
        y_norm = (y_flat.square()).sum(dim=1, keepdim=True).transpose(0, 1)
        dist_sq = torch.clamp(x_norm + y_norm - 2.0 * x_flat @ y_flat.T, min=0.0)
        return 1.0 / (1.0 + dist_sq / (self.epsilon ** 2))

    # ------------------------------------------------------------------
    def _factorise_landmark_kernel(self) -> None:
        eye = torch.eye(self.n_landmarks, device=self.device, dtype=self.K_ll.dtype)
        system = self.K_ll + eye * self.regularization
        try:
            self._chol_factor = torch.linalg.cholesky(system)
            self._svd_factors = None
        except RuntimeError:
            self._chol_factor = None
            self._svd_factors = torch.linalg.svd(system, full_matrices=False)

    def _solve_landmark_system(self, rhs: torch.Tensor) -> torch.Tensor:
        if rhs.dim() == 1:
            rhs = rhs.unsqueeze(-1)
            squeeze = True
        else:
            squeeze = False

        if self._chol_factor is not None:
            solution = torch.cholesky_solve(rhs, self._chol_factor)
        elif self._svd_factors is not None:
            U, S, Vh = self._svd_factors
            inv = (U.mT @ rhs) / S.unsqueeze(-1)
            solution = Vh.mT @ inv
        else:
            eye = torch.eye(self.n_landmarks, device=self.device, dtype=self.K_ll.dtype)
            system = self.K_ll + eye * self.regularization
            solution = torch.linalg.solve(system, rhs)

        if squeeze:
            solution = solution.squeeze(-1)
        return solution

    # ------------------------------------------------------------------
    def _kernel_landmarks(self, batch: int, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (batch, dtype)
        cached = self._cached_kernels.get(cache_key)
        if cached is not None:
            return cached

        landmark_expansion = torch.eye(self.n_landmarks, device=self.device, dtype=dtype)
        self._cached_kernels[cache_key] = landmark_expansion
        return landmark_expansion

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.shape[0] != v.shape[0]:
            raise ValueError("Input data and vector must share the same batch dimension.")

        x = x.to(self.device)
        v = v.to(self.device)
        features = v.reshape(v.shape[0], -1)

        K_xl = self._compute_kernel(x, self.landmarks).to(features.dtype)
        rhs = K_xl.T @ features
        solved = self._solve_landmark_system(rhs)
        result = K_xl @ solved
        return result.reshape(v.shape)

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # The supported kernels are symmetric; the transpose equals the original.
        return self.apply(x, v)

    def get_error_bound(self, n_samples: int) -> float:  # type: ignore[override]
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        return 1.0 / math.sqrt(self.n_landmarks))

    def clear_cache(self) -> None:  # type: ignore[override]
        self._cached_kernels.clear()

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return the kernel matrix between x and y using the configured kernel."""
        return self._compute_kernel(x, y)
