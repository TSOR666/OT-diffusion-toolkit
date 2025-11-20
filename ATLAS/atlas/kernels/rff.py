"""Random Fourier Features kernel operator."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch

from .base import KernelOperator
from ..utils.random import set_seed


class RFFKernelOperator(KernelOperator):
    """Approximate kernels with random Fourier features."""

    _SUPPORTED_KERNELS = {"gaussian", "laplacian"}

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 2048,
        kernel_type: str = "gaussian",
        epsilon: float = 0.01,
        device: Optional[torch.device] = None,
        orthogonal: bool = True,
        multi_scale: bool = True,
        scale_factors: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        max_cached_batch_size: int = 8,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(epsilon, device)

        kernel = kernel_type.lower()
        if kernel not in self._SUPPORTED_KERNELS:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.kernel_type = kernel
        self.orthogonal = orthogonal
        self.multi_scale = multi_scale
        self.scale_factors = list(scale_factors) if scale_factors is not None else [1.0]
        self.max_cached_batch_size = int(max_cached_batch_size)

        self.seed = seed
        if seed is not None:
            set_seed(seed)

        self.weights: List[torch.Tensor] = []
        self.offsets: List[torch.Tensor] = []
        self._feature_cache: "OrderedDict[Tuple[int, torch.dtype], torch.Tensor]" = OrderedDict()

        self._initialise_features()

    # ------------------------------------------------------------------
    def _base_scale(self) -> float:
        if self.kernel_type == "gaussian":
            return 1.0 / max(self.epsilon, 1e-12)
        if self.kernel_type in {"laplacian", "cauchy"}:
            return 1.0 / max(self.epsilon, 1e-12)
        raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def _sample_weights(self, num_features: int, scale: float) -> torch.Tensor:
        if self.orthogonal:
            blocks: List[torch.Tensor] = []
            remaining = num_features
            while remaining > 0:
                block = torch.randn(self.input_dim, self.input_dim, device=self.device)
                q, _ = torch.linalg.qr(block)
                take = min(remaining, self.input_dim)
                blocks.append(q[:, :take])
                remaining -= take
            weights = torch.cat(blocks, dim=1)
        else:
            weights = torch.randn(self.input_dim, num_features, device=self.device)
        return weights * scale

    def _sample_offsets(self, num_features: int) -> torch.Tensor:
        return torch.rand(num_features, device=self.device) * 2.0 * math.pi

    def _initialise_features(self) -> None:
        self.weights.clear()
        self.offsets.clear()

        base_scale = self._base_scale()
        factors = self.scale_factors if self.multi_scale else [1.0]
        splits = self._split_features(len(factors))

        for idx, (factor, count) in enumerate(zip(factors, splits)):
            if self.seed is not None:
                set_seed(self.seed + idx)
            weights = self._sample_weights(count, base_scale * factor)
            offsets = self._sample_offsets(count)
            self.weights.append(weights)
            self.offsets.append(offsets)

    def _split_features(self, num_groups: int) -> List[int]:
        base = self.feature_dim // num_groups
        remainder = self.feature_dim % num_groups
        return [base + (1 if i < remainder else 0) for i in range(num_groups)]

    # ------------------------------------------------------------------
    def _clear_stale_cache(self) -> None:
        while len(self._feature_cache) > self.max_cached_batch_size:
            self._feature_cache.popitem(last=False)

    def _cache_key(self, x: torch.Tensor) -> Optional[Tuple]:
        if x.size(0) > self.max_cached_batch_size:
            return None
        return (
            tuple(x.shape),
            x.dtype,
            x.device.type,
            x.device.index if x.device.type == 'cuda' else None,
            int(x.data_ptr()),
        )

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        # Ensure the tensor has a well-defined storage layout for caching.
        x = x.contiguous()
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.shape[1]}"
            )

        cache_key = self._cache_key(x)
        if cache_key is not None and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        projections = []
        norm_factor = math.sqrt(2.0 / self.feature_dim)
        for weights, offsets in zip(self.weights, self.offsets):
            proj = x @ weights + offsets
            if self.kernel_type == "gaussian":
                features = torch.cos(proj)
            elif self.kernel_type == "laplacian":
                features = torch.cos(proj) + torch.sin(proj)
            else:  
                raise ValueError(f"Unsupported kernel: {self.kernel_type}")
            projections.append(features * norm_factor)

        phi = torch.cat(projections, dim=1)

        if cache_key is not None:
            self._feature_cache[cache_key] = phi
            self._clear_stale_cache()
        return phi

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.shape[0] != v.shape[0]:
            raise ValueError("Input data and vector must share the same batch dimension.")

        features = self.compute_features(x)
        target_shape = v.shape
        v_flat = v.reshape(v.shape[0], -1).to(features.dtype)

        transformed = features.T @ v_flat
        result = features @ transformed
        return result.reshape(target_shape)

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.apply(x, v)

    def get_error_bound(self, n_samples: int) -> float:  # type: ignore[override]
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        return math.sqrt(math.log(max(n_samples, 2)) / self.feature_dim)

    def clear_cache(self) -> None:  # type: ignore[override]
        self._feature_cache.clear()

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute approximate kernel matrix via shared feature space."""
        phi_x = self.compute_features(x)
        phi_y = self.compute_features(y)
        return phi_x @ phi_y.T
