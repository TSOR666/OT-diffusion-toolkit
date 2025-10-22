
import math
from typing import List, Optional, Tuple

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
        scale_factors: List[float] = None
    ):
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
        self.grid_shape = grid_shape
        self.kernel_type = kernel_type
        self.multi_scale = multi_scale
        self.scale_factors = scale_factors if scale_factors else [0.5, 1.0, 2.0]
        
        # Precompute FFT of kernel
        if self.multi_scale:
            self.kernel_ffts = [self._compute_kernel_fft(grid_shape, epsilon * factor) 
                               for factor in self.scale_factors]
            self.weights = torch.softmax(torch.tensor([1.0, 2.0, 1.0], device=self.device), dim=0)
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
    
    def _reshape_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape flattened vector to grid format.
        
        Args:
            x: Flattened vector [batch_size]
            
        Returns:
            Grid-shaped tensor matching grid_shape
        """
        return x.reshape(self.grid_shape)
    
    def _reshape_from_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape grid tensor to flattened vector.
        
        Args:
            x: Grid tensor matching grid_shape
            
        Returns:
            Flattened vector [batch_size]
        """
        return x.reshape(-1)
    
    def apply(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel operator using FFT convolution.
        
        Args:
            x: Input data (not used, here for API consistency)
            v: Input vector [batch_size]
            
        Returns:
            Result [batch_size]
        """
        # For multi-channel data, process each channel separately
        if len(self.grid_shape) > 2 and self.grid_shape[0] > 1:
            # Assume channel is first dimension
            channels = self.grid_shape[0]
            spatial_shape = self.grid_shape[1:]
            
            v_reshaped = v.reshape(channels, -1)
            results = []
            
            for c in range(channels):
                channel_v = v_reshaped[c]
                channel_v_grid = channel_v.reshape(spatial_shape)
                
                if self.multi_scale:
                    # Apply multiple kernels and combine results
                    result_grid = torch.zeros_like(channel_v_grid)
                    for i, kernel_fft in enumerate(self.kernel_ffts):
                        scale_result = self._apply_kernel_fft(channel_v_grid, kernel_fft)
                        result_grid += self.weights[i] * scale_result
                else:
                    result_grid = self._apply_kernel_fft(channel_v_grid, self.kernel_fft)
                
                results.append(result_grid.reshape(-1))
            
            return torch.cat(results).reshape(v.shape)
        else:
            v_grid = self._reshape_to_grid(v)
            
            if self.multi_scale:
                result_grid = torch.zeros_like(v_grid)
                for i, kernel_fft in enumerate(self.kernel_ffts):
                    scale_result = self._apply_kernel_fft(v_grid, kernel_fft)
                    result_grid += self.weights[i] * scale_result
            else:
                result_grid = self._apply_kernel_fft(v_grid, self.kernel_fft)
            
            return self._reshape_from_grid(result_grid)
    
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
