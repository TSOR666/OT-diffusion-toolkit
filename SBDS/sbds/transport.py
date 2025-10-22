"""Transport solvers for SBDS."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import common
from .kernels import KernelDerivativeRFF

warnings = common.warnings
LazyTensor = common.LazyTensor
KEOPS_AVAILABLE = common.KEOPS_AVAILABLE
tqdm = common.tqdm

__all__ = ["FFTOptimalTransport", "HilbertSinkhornDivergence"]


class FFTOptimalTransport:
    """
    FFT-based Optimal Transport implementation.
    
    Based on "FFT-OT: A Fast Algorithm for Optimal Transportation" (Solomon et al.)
    This class implements:
    1. Fast convolution-based Sinkhorn iterations for grid-structured data
    2. Multi-scale FFT-OT for efficient computation
    3. GPU-accelerated implementation
    """
    def __init__(
        self,
        epsilon: float = 0.01,
        max_iter: int = 50,
        tol: float = 1e-5,
        kernel_type: str = 'gaussian',
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        multiscale: bool = True,
        scale_levels: int = 3,
        fallback_block_size: int = 2048
    ):
        """
        Initialize the FFT-OT solver.
        
        Args:
            epsilon: Entropic regularization parameter
            max_iter: Maximum number of Sinkhorn iterations
            tol: Convergence tolerance
            kernel_type: Type of kernel ('gaussian', 'laplacian')
            device: Device to use for computation
            multiscale: Whether to use multi-scale acceleration
            scale_levels: Number of scale levels for multi-scale approach
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_type = kernel_type
        self.device = device
        self.multiscale = multiscale
        self.scale_levels = scale_levels
        self.fallback_block_size = fallback_block_size
        
    def _is_grid_structured(self, x: torch.Tensor) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if data is structured on a regular grid.
        
        Args:
            x: Input tensor [batch_size, *dims] or coordinates [batch_size, ndim]
            
        Returns:
            Tuple of (is_grid, grid_shape)
        """
        # If x has more than 2 dimensions, it's likely already in grid format
        if x.dim() > 2:
            return True, list(x.shape[1:])
            
        # Try to infer grid structure from coordinates
        # This is a simple heuristic that could be enhanced
        if x.dim() == 2:
            ndim = x.size(1)
            if ndim <= 3:  # Only try for 1D, 2D, or 3D data
                # Check if coordinates form a regular grid
                for d in range(ndim):
                    unique_coords = torch.unique(x[:, d])
                    # If number of unique coordinates is too high, probably not a grid
                    if len(unique_coords) > 100:
                        return False, None
                        
                grid_shape = []
                for d in range(ndim):
                    unique_coords = torch.unique(x[:, d])
                    grid_shape.append(len(unique_coords))
                    
                # Check if product of grid shape matches batch size
                if np.prod(grid_shape) == x.size(0):
                    return True, grid_shape
                
        return False, None
    
    def _reshape_to_grid(self, x: torch.Tensor, grid_shape: List[int]) -> torch.Tensor:
        """
        Reshape flattened data to grid format.
        
        Args:
            x: Flattened data [batch_size, channels]
            grid_shape: Shape of the grid
            
        Returns:
            Grid-structured data [channels, *grid_shape]
        """
        batch_size = x.size(0)
        channels = x.size(1) if x.dim() > 1 else 1
        
        if x.dim() == 1:
            x = x.unsqueeze(1)
            
        if np.prod(grid_shape) != batch_size:
            raise ValueError(f"Product of grid shape {grid_shape} does not match batch size {batch_size}")
            
        return x.transpose(0, 1).reshape(channels, *grid_shape)
        
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
            kernel = torch.exp(-dist_sq / (epsilon + 1e-30))
        elif self.kernel_type == 'laplacian':
            kernel = torch.exp(-torch.sqrt(dist_sq + 1e-10) / epsilon)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
        # Compute FFT of kernel
        kernel = torch.fft.ifftshift(kernel)
        kernel_fft = torch.fft.rfftn(kernel)
        
        return kernel_fft
        
    def _sinkhorn_fft(
        self, 
        mu: torch.Tensor, 
        nu: torch.Tensor,
        epsilon: float, 
        kernel_fft: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Sinkhorn iterations using FFT convolutions.
        
        Args:
            mu: Source distribution (grid format)
            nu: Target distribution (grid format)
            epsilon: Regularization parameter
            kernel_fft: Pre-computed FFT of kernel (optional)
            
        Returns:
            Tuple of (transport_plan, dual_u, dual_v)
        """
        # Ensure mu and nu are properly normalized
        mu = mu / mu.sum()
        nu = nu / nu.sum()
        
        # Compute kernel FFT if not provided
        if kernel_fft is None:
            kernel_fft = self._compute_kernel_fft(mu.shape, epsilon)
            
        # Initialize dual potentials
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        
        # Sinkhorn iterations
        for i in range(self.max_iter):
            # Save previous iteration for convergence check
            u_prev = u.clone()
            
            # Update v: exp(v) = nu / (K^T exp(u))
            Ku = self._apply_kernel_fft(torch.exp(u), kernel_fft)
            v = torch.log(nu + 1e-15) - torch.log(Ku + 1e-15)
            
            # Update u: exp(u) = mu / (K exp(v))
            Kv = self._apply_kernel_fft(torch.exp(v), kernel_fft)
            u = torch.log(mu + 1e-15) - torch.log(Kv + 1e-15)
            
            # Check convergence
            err = torch.max(torch.abs(u - u_prev))
            if err < self.tol:
                break
        
        a = torch.exp(u)
        b = torch.exp(v)
        Kv = self._apply_kernel_fft(b, kernel_fft)
        objective = (u * mu).sum() + (v * nu).sum() - epsilon * (a * Kv).sum()
        return objective, u, v
        
    def _multiscale_sinkhorn_fft(
        self, 
        mu: torch.Tensor, 
        nu: torch.Tensor, 
        epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-scale implementation of FFT-OT.
        
        Args:
            mu: Source distribution (grid format)
            nu: Target distribution (grid format)
            epsilon: Regularization parameter
            
        Returns:
            Tuple of (transport_plan, dual_u, dual_v)
        """
        if not self.multiscale:
            return self._sinkhorn_fft(mu, nu, epsilon)
            
        # Get original shape
        shape = mu.shape
        
        # Start from coarsest scale
        current_shape = [max(4, s // (2**self.scale_levels)) for s in shape]
        
        # Downsample distributions to coarsest scale
        mu_coarse = torch.nn.functional.interpolate(
            mu.unsqueeze(0).unsqueeze(0), size=current_shape, 
            mode='bilinear' if len(shape) == 2 else 'trilinear'
        ).squeeze(0).squeeze(0)
        
        nu_coarse = torch.nn.functional.interpolate(
            nu.unsqueeze(0).unsqueeze(0), size=current_shape, 
            mode='bilinear' if len(shape) == 2 else 'trilinear'
        ).squeeze(0).squeeze(0)
        
        # Solve at coarsest scale
        _, u_coarse, v_coarse = self._sinkhorn_fft(mu_coarse, nu_coarse, epsilon)
        
        # Progressively refine solution
        for level in range(self.scale_levels, -1, -1):
            # Compute scale for current level
            current_shape = [max(4, s // (2**level)) for s in shape]
            
            # Skip if already at this scale
            if level == self.scale_levels:
                continue
                
            # Upsample potentials from previous scale
            u_upsampled = torch.nn.functional.interpolate(
                u_coarse.unsqueeze(0).unsqueeze(0), size=current_shape, 
                mode='bilinear' if len(shape) == 2 else 'trilinear'
            ).squeeze(0).squeeze(0)
            
            v_upsampled = torch.nn.functional.interpolate(
                v_coarse.unsqueeze(0).unsqueeze(0), size=current_shape, 
                mode='bilinear' if len(shape) == 2 else 'trilinear'
            ).squeeze(0).squeeze(0)
            
            # Downsample distributions to current scale
            mu_current = torch.nn.functional.interpolate(
                mu.unsqueeze(0).unsqueeze(0), size=current_shape, 
                mode='bilinear' if len(shape) == 2 else 'trilinear'
            ).squeeze(0).squeeze(0)
            
            nu_current = torch.nn.functional.interpolate(
                nu.unsqueeze(0).unsqueeze(0), size=current_shape, 
                mode='bilinear' if len(shape) == 2 else 'trilinear'
            ).squeeze(0).squeeze(0)
            
            # Compute kernel FFT at current scale
            kernel_fft = self._compute_kernel_fft(current_shape, epsilon)
            
            # Refine solution using upsampled potentials as initialization
            _, u_current, v_current = self._sinkhorn_fft(
                mu_current, nu_current, epsilon, kernel_fft
            )
            
            # Update for next iteration
            u_coarse, v_coarse = u_current, v_current
        
            # Final solve at full resolution (always)
        objective, u_final, v_final = self._sinkhorn_fft(mu, nu, epsilon)
        return objective, u_final, v_final

    def _compute_gradient_on_grid(self, u: torch.Tensor, grid_spacing: Optional[List[float]] = None) -> List[torch.Tensor]:
        """
        Compute gradient of potential on grid using finite differences with proper boundary handling.
        
        Args:
            u: Potential on grid [*grid_shape]
            grid_spacing: Spacing between grid points for each dimension, defaults to 1.0
            
        Returns:
            List of gradient components, one for each dimension
        """
        # Get grid shape
        grid_shape = u.shape
        ndim = len(grid_shape)
        
        # Default grid spacing to 1.0 if not provided
        if grid_spacing is None:
            grid_spacing = [1.0] * ndim
        
        # Initialize gradients
        gradients = []
        
        # Compute gradient along each dimension using central differences
        for d in range(ndim):
            # Initialize gradient tensor
            grad_d = torch.zeros_like(u)
            
            # Interior points: central difference
            interior_slices = [slice(1, -1) if i == d else slice(None) for i in range(ndim)]
            forward_slices = [slice(2, None) if i == d else slice(None) for i in range(ndim)]
            backward_slices = [slice(0, -2) if i == d else slice(None) for i in range(ndim)]
            
            # Apply central difference for interior points
            grad_d[tuple(interior_slices)] = (u[tuple(forward_slices)] - u[tuple(backward_slices)]) / (2 * grid_spacing[d])
            
            # Left boundary: forward difference
            left_slices = [slice(0, 1) if i == d else slice(None) for i in range(ndim)]
            left_forward_slices = [slice(1, 2) if i == d else slice(None) for i in range(ndim)]
            grad_d[tuple(left_slices)] = (u[tuple(left_forward_slices)] - u[tuple(left_slices)]) / grid_spacing[d]
            
            # Right boundary: backward difference
            right_slices = [slice(-1, None) if i == d else slice(None) for i in range(ndim)]
            right_backward_slices = [slice(-2, -1) if i == d else slice(None) for i in range(ndim)]
            grad_d[tuple(right_slices)] = (u[tuple(right_slices)] - u[tuple(right_backward_slices)]) / grid_spacing[d]
            
            gradients.append(grad_d)
        
        return gradients
        
    def optimal_transport(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute optimal transport between distributions.
        
        Args:
            x: Source distribution (can be grid or points)
            y: Target distribution (can be grid or points)
            weights_x: Weights for source distribution (optional)
            weights_y: Weights for target distribution (optional)
            
        Returns:
            Tuple of (transport_cost, dual_u, dual_v)
        """
        # Check if data is grid-structured
        is_grid_x, grid_shape_x = self._is_grid_structured(x)
        is_grid_y, grid_shape_y = self._is_grid_structured(y)
        
        # Both must be grid-structured with same shape for FFT-OT
        if is_grid_x and is_grid_y and grid_shape_x == grid_shape_y:
            grid_shape = grid_shape_x
            
            # Prepare distributions
            if x.dim() > 2:  # Already in grid format
                mu = x.sum(0) if x.dim() > len(grid_shape) else x
            else:
                # Reshape to grid
                values = weights_x if weights_x is not None else torch.ones(x.size(0), device=x.device)
                mu = self._reshape_to_grid(values.unsqueeze(1), grid_shape).squeeze(0)
                
            if y.dim() > 2:  # Already in grid format
                nu = y.sum(0) if y.dim() > len(grid_shape) else y
            else:
                # Reshape to grid
                values = weights_y if weights_y is not None else torch.ones(y.size(0), device=y.device)
                nu = self._reshape_to_grid(values.unsqueeze(1), grid_shape).squeeze(0)
                
                        # Run FFT-OT
            objective, u, v = self._multiscale_sinkhorn_fft(mu, nu, self.epsilon)
            # Transport cost (regularized OT dual objective)
            cost = objective
            return cost, u, v
        else:
            # Fall back to streaming Sinkhorn for non-grid data
            warnings.warn(
                "Data is not grid-structured; using chunked Sinkhorn fallback",
                RuntimeWarning,
            )
            if x.dim() > 2:
                x = x.reshape(x.size(0), -1)
            if y.dim() > 2:
                y = y.reshape(y.size(0), -1)

            weights_x = weights_x if weights_x is not None else torch.ones(x.size(0), device=self.device) / max(1, x.size(0))
            weights_y = weights_y if weights_y is not None else torch.ones(y.size(0), device=self.device) / max(1, y.size(0))
            weights_x = (weights_x / weights_x.sum()).to(self.device)
            weights_y = (weights_y / weights_y.sum()).to(self.device)

            cost, u, v = self._sinkhorn_blockwise_fallback(
                x.to(self.device),
                y.to(self.device),
                weights_x,
                weights_y,
                self.epsilon,
                self.max_iter,
                self.tol,
                self.fallback_block_size
            )
            return cost, u, v

    def _sinkhorn_blockwise_fallback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
        epsilon: float,
        max_iter: int,
        tol: float,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Numerically stable Sinkhorn iterations using blockwise cost computation."""
        device = x.device
        dtype = torch.float32

        N, M = x.size(0), y.size(0)
        log_u = torch.zeros(N, device=device, dtype=dtype)
        log_v = torch.zeros(M, device=device, dtype=dtype)

        log_a = torch.log(weights_x.to(device=device, dtype=dtype) + 1e-10)
        log_b = torch.log(weights_y.to(device=device, dtype=dtype) + 1e-10)

        row_block = max(1, block_size // max(1, int(math.sqrt(M) + 1)))
        col_block = block_size

        for _ in range(max_iter):
            log_u_prev = log_u.clone()

            log_u = self._sinkhorn_block_update(
                x, y, log_v, epsilon, log_a, row_block, col_block, update_rows=True
            )
            log_v = self._sinkhorn_block_update(
                y, x, log_u, epsilon, log_b, col_block, row_block, update_rows=False
            )

            err = torch.max(torch.abs(log_u - log_u_prev))
            if err < tol:
                break

        # Compute transport cost without forming dense matrix
        total_cost = 0.0
        for i_start in range(0, N, row_block):
            i_end = min(i_start + row_block, N)
            x_chunk = x[i_start:i_end]
            log_u_chunk = log_u[i_start:i_end]
            for j_start in range(0, M, col_block):
                j_end = min(j_start + col_block, M)
                y_chunk = y[j_start:j_end]
                log_v_chunk = log_v[j_start:j_end]

                C_chunk = torch.cdist(x_chunk, y_chunk, p=2).pow(2)
                log_P_chunk = (
                    log_u_chunk[:, None]
                    + log_v_chunk[None, :]
                    - C_chunk / epsilon
                )
                P_chunk = torch.exp(log_P_chunk)
                total_cost += torch.sum(P_chunk * C_chunk)

        return total_cost.to(x.dtype), log_u.to(x.dtype), log_v.to(x.dtype)

    def _sinkhorn_block_update(
        self,
        x_primary: torch.Tensor,
        x_secondary: torch.Tensor,
        log_dual_secondary: torch.Tensor,
        epsilon: float,
        log_marginal: torch.Tensor,
        primary_block: int,
        secondary_block: int,
        update_rows: bool,
    ) -> torch.Tensor:
        """Compute a single Sinkhorn update in blocks to limit memory pressure."""
        device = x_primary.device
        dtype = log_dual_secondary.dtype
        size_primary = x_primary.size(0)
        updated = torch.empty(size_primary, device=device, dtype=dtype)

        for i_start in range(0, size_primary, primary_block):
            i_end = min(i_start + primary_block, size_primary)
            primary_chunk = x_primary[i_start:i_end]
            log_marginal_chunk = log_marginal[i_start:i_end]

            accum = torch.full(
                (i_end - i_start,),
                -float('inf'),
                device=device,
                dtype=dtype
            )

            for j_start in range(0, x_secondary.size(0), secondary_block):
                j_end = min(j_start + secondary_block, x_secondary.size(0))
                secondary_chunk = x_secondary[j_start:j_end]
                log_dual_chunk = log_dual_secondary[j_start:j_end]

                # Compute local costs
                C_chunk = torch.cdist(primary_chunk, secondary_chunk, p=2).pow(2)
                log_kernel = -C_chunk / epsilon + log_dual_chunk[None, :]
                if update_rows:
                    accum = torch.logaddexp(accum, torch.logsumexp(log_kernel, dim=1))
                else:
                    accum = torch.logaddexp(accum, torch.logsumexp(log_kernel, dim=0))

            updated_chunk = log_marginal_chunk - accum
            updated[i_start:i_end] = updated_chunk

        return updated





class HilbertSinkhornDivergence:
    """
    Implements the Hilbert Sinkhorn Divergence for optimal transport.
    
    Based on:
    - "Hilbert Sinkhorn Divergence for Optimal Transport" (Genevay et al., 2021)
    - "Sample Complexity of Sinkhorn Divergences" (Genevay et al., 2019)
    
    This provides a theoretically sound extension of the Sinkhorn divergence in RKHS,
    with better sample complexity and numerical stability.
    """
    def __init__(
        self,
        epsilon: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-5,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        adaptive_epsilon: bool = True,
        kernel_type: str = 'gaussian',
        sigma: float = 1.0,
        debiased: bool = True,
        use_rff: bool = True,
        rff_features: int = 1024,
        accelerated: bool = True,  # Whether to use accelerated computational methods
    ):
        """
        Initialize the Hilbert Sinkhorn Divergence.
        
        Args:
            epsilon: Entropy regularization parameter
            max_iter: Maximum number of Sinkhorn iterations
            tol: Tolerance for Sinkhorn convergence
            device: Computation device
            adaptive_epsilon: Whether to adapt epsilon based on data scale
            kernel_type: Type of kernel ('gaussian', 'laplacian', etc.)
            sigma: Kernel bandwidth parameter
            debiased: Whether to use the debiased version
            use_rff: Whether to use Random Fourier Features for acceleration
            rff_features: Number of random features for RFF
            accelerated: Whether to use accelerated algorithms
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.adaptive_epsilon = adaptive_epsilon
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.debiased = debiased
        self.use_rff = use_rff
        self.rff_features = rff_features
        self.accelerated = accelerated
        
        # Initialize RFF if requested
        self.rff = None
        
        # Initialize FFT-OT for grid-structured data
        self.fft_ot = FFTOptimalTransport(
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
            kernel_type=kernel_type,
            device=device,
            multiscale=True
        )
        
    def _initialize_rff(self, dim: int):
        """
        Initialize RFF for the given dimension.
        
        Args:
            dim: Dimensionality of the data
        """
        if self.rff is None or self.rff.input_dim != dim:
            self.rff = KernelDerivativeRFF(
                input_dim=dim,
                feature_dim=self.rff_features,
                sigma=self.sigma,
                kernel_type=self.kernel_type,
                device=self.device,
                orthogonal=True
            )
            
    def _compute_cost_matrix(self, x: torch.Tensor, y: torch.Tensor, use_rff: bool = None) -> torch.Tensor:
        """
        Compute cost matrix for Sinkhorn algorithm.
        
        Args:
            x: First point cloud [batch_x, dim]
            y: Second point cloud [batch_y, dim]
            use_rff: Whether to use RFF for computation
            
        Returns:
            Cost matrix [batch_x, batch_y]
        """
        if use_rff is None:
            use_rff = self.use_rff
            
        if use_rff:
            # Ensure RFF is initialized
            dim = x.size(1) if x.dim() > 1 else 1
            self._initialize_rff(dim)
            
            # Use RFF for efficient cost computation
            # For Gaussian kernel, distance in feature space approximates MMD distance
            x_features = self.rff.compute_features(x)
            y_features = self.rff.compute_features(y)
            
            # Normalize features to ensure ||Φ(x)||² = k(x,x) = 1
            x_features = F.normalize(x_features, p=2, dim=1)
            y_features = F.normalize(y_features, p=2, dim=1)
            
            # Compute squared distances in feature space
            x_norm = (x_features**2).sum(1, keepdim=True)
            y_norm = (y_features**2).sum(1, keepdim=True)
            xy = x_features @ y_features.T
            C = x_norm + y_norm.T - 2 * xy
            
            # Scale to match original distance
            C = C * (x.size(1) / self.rff_features)
        else:
            # Direct computation
            C = torch.cdist(x, y, p=2).pow(2)
            
        return C
        
    def _sinkhorn_algorithm(
        self, 
        C: torch.Tensor, 
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
        eps: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sinkhorn algorithm for entropic optimal transport.
        
        Args:
            C: Cost matrix [batch_x, batch_y]
            weights_x: Weights for first distribution (optional)
            weights_y: Weights for second distribution (optional)
            eps: Regularization parameter (optional)
            
        Returns:
            Tuple of (optimal_cost, u, v)
        """
        eps = eps or self.epsilon
        nx, ny = C.shape
        
        # Initialize weights if not provided
        if weights_x is None:
            weights_x = torch.ones(nx, device=C.device) / nx
        else:
            weights_x = weights_x / weights_x.sum()
            
        if weights_y is None:
            weights_y = torch.ones(ny, device=C.device) / ny
        else:
            weights_y = weights_y / weights_y.sum()
            
        # Initialize dual potentials
        u = torch.zeros(nx, device=C.device)
        v = torch.zeros(ny, device=C.device)
        
        # Kernel matrix and its log (cached for efficiency)
        K = torch.exp(-C / eps)
        log_K = -C / eps  # Cache log(K) to avoid recomputation
        
        # Log weights (cached)
        log_weights_x = torch.log(weights_x)
        log_weights_y = torch.log(weights_y)
        
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u_prev = u.clone()
            
            # Update v using cached log values
            v = log_weights_y - torch.logsumexp(u[:, None] + log_K, dim=0)
            
            # Update u using cached log values
            u = log_weights_x - torch.logsumexp(v[None, :] + log_K, dim=1)
            
            # Check convergence
            err = torch.max(torch.abs(u - u_prev))
            if err < self.tol:
                break
                
        # Compute optimal transport plan and cost
        P = torch.exp(u[:, None] + v[None, :]) * K
        cost = torch.sum(P * C)
        
        # Add entropy term for the actual Sinkhorn divergence
        entropy = -torch.sum(P * torch.log(P + 1e-15))
        total_cost = cost - eps * entropy
        
        return total_cost, u, v
        
    def _accelerated_sinkhorn(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
        eps: Optional[float] = None
    ) -> torch.Tensor:
        """
        Accelerated Sinkhorn divergence computation.
        
        Args:
            x: First point cloud [batch_x, dim]
            y: Second point cloud [batch_y, dim]
            weights_x: Weights for first distribution (optional)
            weights_y: Weights for second distribution (optional)
            eps: Regularization parameter (optional)
            
        Returns:
            Sinkhorn divergence value
        """
        # Check if we can use FFT-OT
        is_grid_x, grid_shape_x = self.fft_ot._is_grid_structured(x)
        is_grid_y, grid_shape_y = self.fft_ot._is_grid_structured(y)
        
        if self.accelerated and is_grid_x and is_grid_y and grid_shape_x == grid_shape_y:
            # Use FFT-OT for grid data
            S_xy, _, _ = self.fft_ot.optimal_transport(x, y, weights_x, weights_y)
            
            if self.debiased:
                # Compute baseline terms
                S_xx, _, _ = self.fft_ot.optimal_transport(x, x, weights_x, weights_x)
                S_yy, _, _ = self.fft_ot.optimal_transport(y, y, weights_y, weights_y)
                
                # Compute debiased divergence
                return S_xy - 0.5 * (S_xx + S_yy)
            else:
                return S_xy
        else:
            # Use RFF-accelerated Sinkhorn for non-grid data
            eps = eps or self.epsilon
            
            # Flatten data if needed
            if x.dim() > 2:
                x = x.reshape(x.size(0), -1)
            if y.dim() > 2:
                y = y.reshape(y.size(0), -1)
                
            # Compute cost matrices
            C_xy = self._compute_cost_matrix(x, y)
            
            # Compute cross-term
            S_xy, _, _ = self._sinkhorn_algorithm(C_xy, weights_x, weights_y, eps)
            
            if self.debiased:
                # Compute baseline terms for debiasing
                C_xx = self._compute_cost_matrix(x, x)
                C_yy = self._compute_cost_matrix(y, y)
                
                # Compute Sinkhorn distances for baseline terms
                S_xx, _, _ = self._sinkhorn_algorithm(C_xx, weights_x, weights_x, eps)
                S_yy, _, _ = self._sinkhorn_algorithm(C_yy, weights_y, weights_y, eps)
                
                # Compute debiased divergence
                return S_xy - 0.5 * (S_xx + S_yy)
            else:
                return S_xy
                
    def compute_divergence(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
        eps: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute the Hilbert Sinkhorn Divergence between two point clouds.
        
        Args:
            x: First point cloud [batch_x, dim]
            y: Second point cloud [batch_y, dim]
            weights_x: Weights for first distribution (optional)
            weights_y: Weights for second distribution (optional)
            eps: Regularization parameter (optional)
            
        Returns:
            Hilbert Sinkhorn Divergence value
        """
        # Determine epsilon based on data scale if adaptive
        if self.adaptive_epsilon and eps is None:
            # Scale epsilon based on average distance
            if x.dim() > 2:
                x_flat = x.reshape(x.size(0), -1)
                y_flat = y.reshape(y.size(0), -1)
            else:
                x_flat = x
                y_flat = y
                
            # Compute average distance using RFF if possible
            if self.use_rff:
                dim = x_flat.size(1)
                self._initialize_rff(dim)
                x_features = self.rff.compute_features(x_flat)
                y_features = self.rff.compute_features(y_flat)
                
                # Quick estimate of average distance in feature space
                x_sample = x_features[:min(100, x_features.size(0))]
                y_sample = y_features[:min(100, y_features.size(0))]
                avg_dist = torch.cdist(x_sample, y_sample, p=2).mean()
                # Scale back to original space
                avg_dist = avg_dist * math.sqrt(dim / self.rff_features)
            else:
                # More expensive direct computation
                x_sample = x_flat[:min(100, x_flat.size(0))]
                y_sample = y_flat[:min(100, y_flat.size(0))]
                avg_dist = torch.cdist(x_sample, y_sample, p=2).mean()
                
            eps = self.epsilon * avg_dist.item()
        else:
            eps = eps or self.epsilon
            
        # Use accelerated computation when appropriate
        return self._accelerated_sinkhorn(x, y, weights_x, weights_y, eps)
        
    def estimate_sample_complexity(self, dim: int, n_samples: int) -> Dict[str, float]:
        """
        Estimate theoretical guarantees on sample complexity.
        
        Args:
            dim: Dimensionality of the data
            n_samples: Number of samples
            
        Returns:
            Dictionary with sample complexity estimates
        """
        # Sample complexity from "Sample Complexity of Sinkhorn Divergences"
        # Standard Sinkhorn: O(n^(-1/(d+4)))
        # Debiased Sinkhorn: O(n^(-1/2)) (dimension-independent for debiased)
        # RFF-accelerated: O(n^(-1/2) + sqrt(log(n)/D))
        
        std_complexity = n_samples**(-1/(dim+4))
        debiased_complexity = n_samples**(-1/2)
        rff_complexity = n_samples**(-1/2) + math.sqrt(math.log(n_samples) / self.rff_features)
        
        return {
            'standard_sinkhorn': std_complexity,
            'debiased_sinkhorn': debiased_complexity,
            'rff_accelerated': rff_complexity
        }


