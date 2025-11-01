
import math
from typing import List, Optional, Sequence

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
    def __init__(
        self, 
        grid_shape: List[int],
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
            kernel_type: Type of kernel ('gaussian', 'laplacian', 'multiquadric', 'cauchy')
            epsilon: Regularization parameter
            device: Computation device
            multi_scale: Whether to use multiscale approach
            scale_factors: Scale factors for multi-scale kernel
        """
        super().__init__(epsilon, device)

        # Validate grid_shape
        if not grid_shape or any(dim <= 0 for dim in grid_shape):
            raise ValueError(f"grid_shape must contain only positive dimensions, got {grid_shape}")

        self.grid_shape = grid_shape
        self.kernel_type = kernel_type
        self.multi_scale = multi_scale
        self.scale_factors = [float(factor) for factor in (scale_factors or [0.5, 1.0, 2.0])]
        
        # Precompute FFT of kernel
        if self.multi_scale:
            if not self.scale_factors:
                raise ValueError("scale_factors must contain at least one value when multi_scale is enabled")
            self.kernel_ffts = [
                self._compute_kernel_fft(grid_shape, epsilon * factor)
                for factor in self.scale_factors
            ]
            if scale_factors is None and len(self.scale_factors) == 3:
                base_weights = torch.tensor(
                    [1.0, 2.0, 1.0], device=self.device, dtype=torch.float32
                )
            else:
                base_weights = torch.ones(
                    len(self.scale_factors), device=self.device, dtype=torch.float32
                )
            self.weights = base_weights / base_weights.sum()
        else:
            self.kernel_fft = self._compute_kernel_fft(grid_shape, epsilon)
    
    def _compute_kernel_fft(self, shape: List[int], epsilon: float) -> torch.Tensor:
        """
        Compute the FFT of the convolution kernel.
        
        Args:
            shape: Shape of the grid
            epsilon: Regularization parameter
            
        Returns:
            FFT of the kernel
        """
        # Create coordinate grid
        coords = []
        for i, size in enumerate(shape):
            coord = torch.arange(size, device=self.device) - size // 2
            coord = coord.reshape([1 if j != i else size for j in range(len(shape))])
            coords.append(coord)
            
        # Compute squared distances
        dist_sq = sum(c**2 for c in coords)
        
        # Compute kernel
        if self.kernel_type == 'gaussian':
            kernel = torch.exp(-dist_sq / (2 * epsilon))
        elif self.kernel_type == 'laplacian':
            kernel = torch.exp(-torch.sqrt(dist_sq + 1e-10) / epsilon)
        elif self.kernel_type == 'multiquadric':
            kernel = torch.sqrt(dist_sq + epsilon**2)
        elif self.kernel_type == 'cauchy':
            kernel = 1.0 / (1.0 + dist_sq / epsilon**2)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
            
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Compute FFT of kernel
        kernel_fft = torch.fft.rfftn(kernel)
        
        return kernel_fft
    
    def _apply_kernel_fft(self, u: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel using FFT convolution.
        
        Args:
            u: Input grid data
            kernel_fft: Pre-computed FFT of the kernel
            
        Returns:
            Result of kernel convolution
        """
        # Compute FFT of u
        u_fft = torch.fft.rfftn(u)
        
        # Apply kernel in Fourier domain (pointwise multiplication)
        result_fft = u_fft * kernel_fft
        
        # Transform back to spatial domain
        result = torch.fft.irfftn(result_fft, s=u.shape)
        
        return result
    
    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel operator using FFT convolution.

        Args:
            x: Input data (not used, here for API consistency)
            v: Input vector [batch_size]

        Returns:
            Result [batch_size]
        """
        spatial_dims = len(self.grid_shape)
        if spatial_dims == 0:
            raise ValueError("grid_shape must contain at least one spatial dimension")

        if v.numel() == 0:
            return v

        original_shape = v.shape
        needs_reduction = False
        grid_numel = math.prod(self.grid_shape)

        if v.dim() >= spatial_dims and list(v.shape[-spatial_dims:]) == self.grid_shape:
            reshaped = v
        elif v.dim() >= 1 and v.shape[-1] == grid_numel:
            try:
                reshaped = v.reshape(*v.shape[:-1], *self.grid_shape)
            except RuntimeError as exc:  # pragma: no cover - defensive programming
                raise ValueError(
                    "Input tensor cannot be reshaped to match provided grid_shape"
                ) from exc
        elif x is not None and x.shape[0] == v.shape[0]:
            # Handle vectors defined per-sample (e.g. [batch]) by broadcasting to the grid
            batch_dim = v.shape[0]
            view_shape = (batch_dim,) + (1,) * spatial_dims
            expanded = v.reshape(view_shape).expand(batch_dim, *self.grid_shape)
            reshaped = expanded.contiguous()
            needs_reduction = True
        else:
            raise ValueError(
                "Input tensor cannot be reshaped or broadcast to match provided grid_shape"
            )

        batch_shape = reshaped.shape[:-spatial_dims]
        v_batches = reshaped.reshape(-1, *self.grid_shape)

        results = []
        for sample in v_batches:
            if self.multi_scale:
                result_grid = torch.zeros_like(sample)
                for weight, kernel_fft in zip(self.weights, self.kernel_ffts):
                    scale_result = self._apply_kernel_fft(sample, kernel_fft)
                    result_grid += weight * scale_result
            else:
                result_grid = self._apply_kernel_fft(sample, self.kernel_fft)
            results.append(result_grid)

        stacked = torch.stack(results, dim=0)
        stacked = stacked.reshape(*batch_shape, *self.grid_shape)

        if needs_reduction:
            reduce_dims = tuple(range(-spatial_dims, 0))
            reduced = stacked.mean(dim=reduce_dims)
            return reduced.reshape(original_shape)

        return stacked
    
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
        """Clear any cached computations to free memory."""
        # FFT kernels are precomputed and stored, no dynamic cache to clear
        pass
