"""Random Fourier Features kernel operator."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import List, Optional, Sequence

import torch

from .base import KernelOperator


class RFFKernelOperator(KernelOperator):
    """Approximate kernels with random Fourier features."""

    _SUPPORTED_KERNELS = {"gaussian", "laplacian", "cauchy"}

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
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if scale_factors is not None and (len(scale_factors) == 0 or any(s <= 0 for s in scale_factors)):
            raise ValueError(f"scale_factors must be positive when provided, got {scale_factors}")

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
        self._cache_fingerprint_size = 256

        self.seed = seed
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)

        self.weights: List[torch.Tensor] = []
        self.offsets: List[torch.Tensor] = []
        self._feature_cache: "OrderedDict[tuple[object, ...], torch.Tensor]" = OrderedDict()

        self._initialise_features()

    # ------------------------------------------------------------------
    def _ensure_device_consistency(self) -> None:
        """Ensure generator and cached parameters stay on the configured device."""
        if self.rng.device != self.device:
            self.rng = torch.Generator(device=self.device)
            if self.seed is not None:
                self.rng.manual_seed(self.seed)

        if any(weight.device != self.device for weight in self.weights):
            self.weights = [weight.to(self.device) for weight in self.weights]
        if any(offset.device != self.device for offset in self.offsets):
            self.offsets = [offset.to(self.device) for offset in self.offsets]

    def _base_scale(self) -> float:
        if self.kernel_type == "gaussian":
            return 1.0 / self.epsilon
        if self.kernel_type in {"laplacian", "cauchy"}:
            return 1.0 / self.epsilon
        raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def _sample_weights(self, num_features: int, scale: float) -> torch.Tensor:
        if self.orthogonal:
            if self.kernel_type != "gaussian":
                raise ValueError("Orthogonal RFF sampling is only supported for Gaussian kernels.")
            blocks: List[torch.Tensor] = []
            remaining = num_features
            while remaining > 0:
                block = torch.randn(self.input_dim, self.input_dim, device=self.device, generator=self.rng)
                q, _ = torch.linalg.qr(block)
                take = min(remaining, self.input_dim)
                blocks.append(q[:, :take])
                remaining -= take
            weights = torch.cat(blocks, dim=1)
        else:
            if self.kernel_type == "gaussian":
                weights = torch.randn(self.input_dim, num_features, device=self.device, generator=self.rng)
            elif self.kernel_type == "laplacian":
                u = torch.rand(self.input_dim, num_features, device=self.device, generator=self.rng)
                # Clamp away from 0 and 1 to prevent tan() from producing ±inf
                u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
                weights = torch.tan(math.pi * (u - 0.5))
            elif self.kernel_type == "cauchy":
                # Cauchy kernel RFF: sample from standard Cauchy distribution
                # Cauchy(0,1) = Normal(0,1) / Normal(0,1) (ratio of independent normals)
                numer = torch.randn(self.input_dim, num_features, device=self.device, generator=self.rng)
                denom = torch.randn(self.input_dim, num_features, device=self.device, generator=self.rng)
                # Clamp denominator away from zero to prevent inf
                denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-7)
                weights = numer / denom
            else:
                raise ValueError(f"Unsupported kernel type for sampling: {self.kernel_type}")
        return weights * scale

    def _sample_offsets(self, num_features: int) -> torch.Tensor:
        return torch.rand(num_features, device=self.device, generator=self.rng) * 2.0 * math.pi

    def _initialise_features(self) -> None:
        self.weights.clear()
        self.offsets.clear()

        base_scale = self._base_scale()
        factors = self.scale_factors if self.multi_scale else [1.0]
        splits = self._split_features(len(factors))

        for idx, (factor, count) in enumerate(zip(factors, splits)):
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

    def _cache_key(self, x: torch.Tensor) -> Optional[tuple[object, ...]]:
        """
        Build a lightweight, content-based cache key.

        The tensor is sampled (not fully materialised) to avoid large host transfers.
        """
        if x.requires_grad:
            return None

        try:
            flat = x.reshape(-1)
            if flat.numel() == 0:
                sample = flat
            elif flat.numel() <= self._cache_fingerprint_size:
                sample = flat
            else:
                idx = torch.linspace(
                    0, flat.numel() - 1, steps=self._cache_fingerprint_size, device=x.device
                ).long()
                sample = flat.index_select(0, idx)

            sample_bytes = sample.detach().cpu().numpy().tobytes()
            digest = hashlib.sha1(sample_bytes).hexdigest()
            device_index = x.device.index if x.device.index is not None else -1
            return (
                tuple(x.shape),
                tuple(x.stride()),
                str(x.dtype),
                x.device.type,
                device_index,
                digest,
            )
        except Exception:
            return None

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_device_consistency()
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
            if self.kernel_type in {"gaussian", "laplacian", "cauchy"}:
                features = torch.cos(proj)
            else:
                raise ValueError(f"Unsupported kernel: {self.kernel_type}")
            projections.append(features * norm_factor)

        phi = torch.cat(projections, dim=1)

        if cache_key is not None:
            self._feature_cache[cache_key] = phi
            self._clear_stale_cache()
        return phi

    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != v.shape[0]:
            raise ValueError("Input data and vector must share the same batch dimension.")

        features = self.compute_features(x)
        target_shape = v.shape
        v_flat = v.reshape(v.shape[0], -1).to(features.dtype)

        transformed = features.T @ v_flat  # (f, n) @ (n, k) -> (f, k)
        result = features @ transformed  # (n, f) @ (f, k) -> (n, k)
        return result.reshape(target_shape)

    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.apply(x, v)

    def get_error_bound(self, n_samples: int, confidence: float = 0.95) -> float:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        delta = (1.0 - confidence) / (n_samples ** 2)
        return math.sqrt(2.0 * math.log(2.0 / delta) / self.feature_dim)

    def clear_cache(self) -> None:
        self._feature_cache.clear()

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute approximate kernel matrix via shared feature space."""
        phi_x = self.compute_features(x)
        phi_y = self.compute_features(y)
        return phi_x @ phi_y.T  # (n, f) @ (f, m) -> (n, m)
