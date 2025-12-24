
from __future__ import annotations

import hashlib
import math
from typing import Any, Optional, Sequence, Tuple, cast

import torch

from .base import KernelOperator


class FFTKernelOperator(KernelOperator):
    """
    FFT-based implementation of kernel operator for grid-structured data.
    
    Memory: O(r^d)
    Computation: O(r^d log r)
    Best for: Grid-structured data like images
    
    Improvements:
    - Multi-scale kernel support for better detail preservation
    - Channel-wise processing for optimized performance
    - Extended kernel type support
    - Memory optimizations for high-resolution images
    """

    _SUPPORTED_KERNELS = {"gaussian", "laplacian", "cauchy"}
    _COORD_FINGERPRINT_SIZE = 256

    def __init__(
        self, 
        grid_shape: Sequence[int],
        kernel_type: str = 'gaussian', 
        epsilon: float = 0.01,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        multi_scale: bool = True,
        scale_factors: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Initialize FFT kernel operator.

        Args:
            grid_shape: Shape of the grid [dim1, dim2, ...]
            kernel_type: Type of kernel ('gaussian', 'laplacian', 'cauchy')
            epsilon: Regularization parameter
            device: Computation device
            multi_scale: Whether to use multiscale approach
            scale_factors: Scale factors for multi-scale kernel
        """
        super().__init__(epsilon, device)

        # Validate grid_shape
        if not grid_shape or any(dim <= 0 for dim in grid_shape):
            raise ValueError(f"grid_shape must contain only positive dimensions, got {grid_shape}")

        self.grid_shape = tuple(int(dim) for dim in grid_shape)
        kernel = kernel_type.lower()
        if kernel not in self._SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel type: {kernel_type}. "
                f"Supported types: {sorted(self._SUPPORTED_KERNELS)}"
            )

        self.kernel_type = kernel
        self.multi_scale = multi_scale
        if scale_factors is None:
            self.scale_factors = [0.5, 1.0, 2.0]
        else:
            self.scale_factors = list(scale_factors)
        if any(factor <= 0 for factor in self.scale_factors):
            raise ValueError(f"scale_factors must be positive, got {self.scale_factors}")
        if self.multi_scale and len(self.scale_factors) == 0:
            raise ValueError("scale_factors must contain at least one value when multi_scale is enabled.")

        self._cleared = False
        self._coord_perm_key: Optional[Tuple[Any, ...]] = None
        self._coord_perm: Optional[torch.Tensor] = None
        self._coord_perm_inv: Optional[torch.Tensor] = None
        
        # Precompute FFT of kernel
        if self.multi_scale:
            self.kernel_ffts = [self._compute_kernel_fft(self.grid_shape, epsilon * factor) 
                               for factor in self.scale_factors]
            weight_values = torch.tensor(self.scale_factors, dtype=torch.float32, device=self.device)
            self.weights = weight_values / weight_values.sum()
        else:
            self.kernel_fft = self._compute_kernel_fft(self.grid_shape, epsilon)
    
    def _compute_kernel_fft(self, shape: Sequence[int], epsilon: float) -> torch.Tensor:
        """
        Compute the FFT of the convolution kernel.
        
        Args:
            shape: Shape of the grid
            epsilon: Regularization parameter
            
        Returns:
            FFT of the kernel
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"shape must contain only positive dimensions, got {shape}")

        # Create coordinate grid
        coords = []
        for i, size in enumerate(shape):
            coord = torch.arange(size, device=self.device, dtype=torch.float32) - size // 2
            coord = coord.reshape([1 if j != i else size for j in range(len(shape))])
            coords.append(coord)
            
        # Compute squared distances
        dist_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for coord in coords:
            dist_sq = dist_sq + coord.square()
        
        # Compute kernel
        if self.kernel_type == 'gaussian':
            sigma_sq = epsilon ** 2
            # Two-sided clamp: max=0 ensures no positive exponents from numerical errors
            exponent = torch.clamp(-dist_sq / (2 * sigma_sq), min=-50.0, max=0.0)
            kernel = torch.exp(exponent)
        elif self.kernel_type == 'laplacian':
            # Two-sided clamp: max=0 ensures no positive exponents from numerical errors
            exponent = torch.clamp(-torch.sqrt(dist_sq + 1e-10) / epsilon, min=-50.0, max=0.0)
            kernel = torch.exp(exponent)
        elif self.kernel_type == 'cauchy':
            denom = 1.0 + dist_sq / (epsilon ** 2)
            denom = torch.clamp(denom, min=1e-12)
            kernel = 1.0 / denom
        else:
            raise ValueError(
                f"Unsupported kernel type: {self.kernel_type}. "
                f"Supported types: 'gaussian', 'laplacian', 'cauchy'"
            )
            
        # Normalize kernel
        kernel_sum = kernel.sum()
        if kernel_sum <= 0:
            raise ValueError(
                f"Kernel normalization failed: kernel sum is {kernel_sum:.6e}. "
                f"This may indicate invalid epsilon ({epsilon}) or numerical issues."
            )
        kernel = kernel / kernel_sum

        kernel = torch.fft.fftshift(kernel, dim=tuple(range(len(shape))))

        # Compute FFT of kernel
        kernel_fft = torch.fft.rfftn(kernel)

        return cast(torch.Tensor, kernel_fft)

    def _coord_cache_key(self, x: torch.Tensor) -> Optional[Tuple[Any, ...]]:
        if x.requires_grad:
            return None

        try:
            flat = x.reshape(-1)
            if flat.numel() == 0:
                sample = flat
            elif flat.numel() <= self._COORD_FINGERPRINT_SIZE:
                sample = flat
            else:
                idx = torch.linspace(
                    0,
                    flat.numel() - 1,
                    steps=self._COORD_FINGERPRINT_SIZE,
                    device=x.device,
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

    def _build_coordinate_permutation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_dims = len(self.grid_shape)
        grid_numel = math.prod(self.grid_shape)

        if x.dim() != 2 or x.shape[0] != grid_numel or x.shape[1] != spatial_dims:
            raise ValueError(
                f"Expected coordinate grid of shape ({grid_numel}, {spatial_dims}), got {tuple(x.shape)}"
            )

        x = x.detach()
        strides = [1] * spatial_dims
        for d in range(spatial_dims - 2, -1, -1):
            strides[d] = strides[d + 1] * self.grid_shape[d + 1]

        flat_index = torch.zeros(grid_numel, device=x.device, dtype=torch.long)
        for dim, (size, stride) in enumerate(zip(self.grid_shape, strides)):
            values = x[:, dim].contiguous()
            unique_values = torch.unique(values).sort().values
            if unique_values.numel() != size:
                raise ValueError(
                    "Coordinate grid does not match grid_shape: expected "
                    f"{size} unique values for dim {dim}, got {unique_values.numel()}"
                )
            idx = torch.searchsorted(unique_values, values)
            if not torch.all(unique_values[idx] == values):
                raise ValueError("Coordinate grid contains values outside inferred axis values.")
            flat_index = flat_index + idx.to(torch.long) * int(stride)

        sorted_flat = flat_index.sort().values
        expected = torch.arange(grid_numel, device=x.device, dtype=sorted_flat.dtype)
        if not torch.equal(sorted_flat, expected):
            raise ValueError(
                "Coordinate grid does not cover a full regular grid compatible with grid_shape."
            )

        perm = torch.argsort(flat_index)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(grid_numel, device=perm.device, dtype=perm.dtype)
        return perm, inv

    def _maybe_build_permutation(
        self, x: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if x is None:
            return None, None

        x = x.to(self.device)
        cache_key = self._coord_cache_key(x)
        if (
            cache_key is not None
            and cache_key == self._coord_perm_key
            and self._coord_perm is not None
            and self._coord_perm_inv is not None
        ):
            return self._coord_perm, self._coord_perm_inv

        perm, inv = self._build_coordinate_permutation(x)
        if cache_key is not None:
            self._coord_perm_key = cache_key
            self._coord_perm = perm
            self._coord_perm_inv = inv
        return perm, inv

    def _apply_kernel_fft(self, u: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel using FFT convolution.
        
        Args:
            u: Input grid data
            kernel_fft: Pre-computed FFT of the kernel
            
        Returns:
            Result of kernel convolution
        """
        spatial_dims = len(self.grid_shape)
        if spatial_dims == 0:
            raise ValueError("grid_shape must contain at least one spatial dimension")
        if tuple(u.shape[-spatial_dims:]) != tuple(self.grid_shape):
            raise ValueError(
                f"Input grid shape {tuple(u.shape)} does not match configured grid_shape {tuple(self.grid_shape)}"
            )

        dims = tuple(range(-spatial_dims, 0))
        u_fft = torch.fft.rfftn(u, dim=dims)
        result_fft = u_fft * kernel_fft
        out = torch.fft.irfftn(result_fft, s=self.grid_shape, dim=dims)
        return cast(torch.Tensor, out)

    def _apply_kernel(self, u: torch.Tensor) -> torch.Tensor:
        if self._cleared:
            raise RuntimeError(
                "FFTKernelOperator cache cleared; reinitialize before calling apply()."
            )

        spatial_dims = len(self.grid_shape)
        dims = tuple(range(-spatial_dims, 0))
        if self.multi_scale:
            u_fft = torch.fft.rfftn(u, dim=dims)
            result_fft = torch.zeros_like(u_fft)
            for weight, kernel_fft in zip(self.weights, self.kernel_ffts):
                result_fft = result_fft + (weight * (u_fft * kernel_fft))
            out = torch.fft.irfftn(result_fft, s=self.grid_shape, dim=dims)
            return cast(torch.Tensor, out)
        return self._apply_kernel_fft(u, self.kernel_fft)
    
    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel operator using FFT convolution.

        Args:
            x: Input data (not used, here for API consistency)
            v: Input vector [batch_size]

        Returns:
            Result [batch_size]
        """
        if self._cleared:
            raise RuntimeError(
                "FFTKernelOperator cache cleared; reinitialize before calling apply()."
            )

        spatial_dims = len(self.grid_shape)
        if spatial_dims == 0:
            raise ValueError("grid_shape must contain at least one spatial dimension")

        if v.numel() == 0:
            return v

        grid_numel = math.prod(self.grid_shape)
        v_in = v.to(self.device)
        compute_dtype = v_in.dtype
        if compute_dtype in {torch.float16, torch.bfloat16}:
            compute_dtype = torch.float32
        v_work = v_in.to(compute_dtype)

        # Case 1: tensor already stored on a spatial grid (..., *grid_shape)
        if v_work.dim() >= spatial_dims and v_work.shape[-spatial_dims:] == self.grid_shape:
            batch_shape = v_work.shape[:-spatial_dims]
            v_batches = v_work.reshape(-1, *self.grid_shape)
            out_batches = self._apply_kernel(v_batches)
            return out_batches.reshape(*batch_shape, *self.grid_shape)

        # Case 2: points axis first (grid_numel, ...)
        if v_work.dim() >= 1 and v_work.shape[0] == grid_numel:
            perm, inv = self._maybe_build_permutation(x)
            target_shape = v_work.shape
            v_flat = v_work.reshape(grid_numel, -1)
            if perm is not None:
                v_flat = v_flat.index_select(0, perm)

            feature_dim = int(v_flat.shape[1])
            v_batches = (
                v_flat.transpose(0, 1)
                .contiguous()
                .reshape(feature_dim, *self.grid_shape)
            )
            out_batches = self._apply_kernel(v_batches)
            out_flat = (
                out_batches.reshape(feature_dim, grid_numel)
                .transpose(0, 1)
                .contiguous()
            )
            if inv is not None:
                out_flat = out_flat.index_select(0, inv)
            return out_flat.reshape(target_shape)

        # Case 3: points axis last (..., grid_numel)
        if v_work.dim() >= 1 and v_work.shape[-1] == grid_numel:
            perm, inv = self._maybe_build_permutation(x)
            target_shape = v_work.shape
            v_batch = v_work.reshape(-1, grid_numel)
            if perm is not None:
                v_batch = v_batch.index_select(1, perm)
            v_grid = v_batch.reshape(-1, *self.grid_shape)
            out_grid = self._apply_kernel(v_grid)
            out_batch = out_grid.reshape(-1, grid_numel)
            if inv is not None:
                out_batch = out_batch.index_select(1, inv)
            return out_batch.reshape(target_shape)

        raise ValueError(
            "Input tensor must either end with grid_shape, have grid_numel as the first dimension, "
            "or have grid_numel as the last dimension."
        )
    
    def apply_transpose(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply transpose kernel operator using FFT convolution.
        
        For symmetric kernels, this is the same as apply().
        
        Args:
            x: Input data (not used, here for API consistency)
            v: Input vector [batch_size]
            
        Returns:
            Result [batch_size]
        """
        # For symmetric kernels, K^T = K
        return self.apply(x, v)
    
    def get_error_bound(self, n_samples: int) -> float:
        """
        Get theoretical error bound for FFT approximation.
        
        Error bound: O(1/resolution) where resolution is grid size.
        
        Args:
            n_samples: Number of samples (not used)
            
        Returns:
            Theoretical error bound
        """
        # Error bound based on grid resolution
        min_res = min(self.grid_shape)
        return 1.0 / min_res

    def clear_cache(self) -> None:
        """Clear any cached computations to free memory.

        Note: After clearing, the operator must be reinitialized before use.
        """
        self._cleared = True
        self._coord_perm_key = None
        self._coord_perm = None
        self._coord_perm_inv = None
        if self.multi_scale:
            self.kernel_ffts.clear()
            self.weights = torch.empty(0, device=self.device)
        else:
            self.kernel_fft = torch.empty(0, device=self.device)
