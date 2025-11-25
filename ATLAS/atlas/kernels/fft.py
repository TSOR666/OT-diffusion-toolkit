
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

        self.grid_shape = grid_shape
        self.kernel_type = kernel_type
        self.multi_scale = multi_scale
        if scale_factors is None:
            self.scale_factors = [0.5, 1.0, 2.0]
        else:
            self.scale_factors = list(scale_factors)
        if self.multi_scale and len(self.scale_factors) == 0:
            raise ValueError("scale_factors must contain at least one value when multi_scale is enabled.")
        
        # Precompute FFT of kernel
        if self.multi_scale:
            self.kernel_ffts = [self._compute_kernel_fft(grid_shape, epsilon * factor) 
                               for factor in self.scale_factors]
            weight_values = torch.tensor(self.scale_factors, dtype=torch.float32, device=self.device)
            self.weights = weight_values / weight_values.sum()
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
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"shape must contain only positive dimensions, got {shape}")

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
            sigma_sq = epsilon ** 2
            exponent = torch.clamp(-dist_sq / (2 * sigma_sq), min=-50.0)
            kernel = torch.exp(exponent)
        elif self.kernel_type == 'laplacian':
            exponent = torch.clamp(-torch.sqrt(dist_sq + 1e-10) / epsilon, min=-50.0)
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
        if list(u.shape) != self.grid_shape:
            raise ValueError(f"Input grid shape {tuple(u.shape)} does not match configured grid_shape {tuple(self.grid_shape)}")
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
        # Clear precomputed FFT kernels to free GPU memory
        if self.multi_scale:
            self.kernel_ffts.clear()
            if hasattr(self, "weights"):
                del self.weights
                self.weights = None
        else:
            if hasattr(self, "kernel_fft"):
                del self.kernel_fft
                self.kernel_fft = None
