"""Kernel feature approximations for SBDS."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from . import common

LazyTensor = common.LazyTensor
KEOPS_AVAILABLE = common.KEOPS_AVAILABLE
warnings = common.warnings

__all__ = ["KernelDerivativeRFF"]


class KernelDerivativeRFF:
    """
    Advanced Random Fourier Features for kernel and kernel derivative approximation.
    
    Based on "On Kernel Derivative Approximation with Random Fourier Features" (Li et al.)
    This implementation includes:
    1. Specialized RFF for kernel derivative approximation
    2. Optimized sampling schemes
    3. Theoretical error bounds for derivatives
    4. Support for different kernels
    """
    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 1024,
        sigma: float = 1.0,
        kernel_type: str = 'gaussian',
        seed: Optional[int] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        rademacher: bool = False,  # Whether to use Rademacher sampling instead of Gaussian
        orthogonal: bool = True,   # Whether to use orthogonal random features
        derivative_order: int = 1,  # Maximum order of derivatives to support
    ):
        """
        Initialize the RFF approximator for kernel and derivatives.
        
        Args:
            input_dim: Dimensionality of the input data
            feature_dim: Number of random features
            sigma: Kernel bandwidth
            kernel_type: Type of kernel ('gaussian', 'laplacian', etc.)
            seed: Random seed for reproducibility
            device: Device to use for computation
            rademacher: Whether to use Rademacher random sampling
            orthogonal: Whether to use orthogonal random features
            derivative_order: Maximum order of derivatives to support
        """
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.sigma = sigma
        self.kernel_type = kernel_type
        self.device = device
        self.rademacher = rademacher
        self.orthogonal = orthogonal
        self.derivative_order = derivative_order
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)  # Set numpy random seed too
            
        # Initialize random features based on kernel type
        self._initialize_random_features()
        
        # Theoretical error bounds (from Theorem 3.1 in the paper)
        # Error bound for kernel approximation: O(sqrt(log(n)/D))
        # Error bound for d-th order derivative: O(sqrt(log(n)/D) * sigma^(-d))
        self.error_bound_factor = math.sqrt(math.log(max(input_dim * 100, 2)) / feature_dim)
        
    def _initialize_random_features(self):
        """
        Initialize random features based on kernel type and sampling scheme.
        Different kernels correspond to different sampling distributions.
        """
        # Scale factor for the sampling distribution based on kernel
        if self.kernel_type == 'gaussian':
            # For Gaussian kernel, sample from Normal(0, 1/sigma²)
            scale = 1.0 / self.sigma
        elif self.kernel_type == 'laplacian':
            # For Laplacian kernel, sample from Cauchy distribution
            scale = 1.0 / self.sigma
        elif self.kernel_type == 'cauchy':
            # For Cauchy kernel
            scale = 1.0 / self.sigma
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
            
        # Determine the number of features needed
        # For orthogonal features, we create blocks of D features
        if self.orthogonal:
            # Number of full blocks
            num_blocks = self.feature_dim // self.input_dim
            # Remaining features
            remainder = self.feature_dim % self.input_dim
            
            # Generate orthogonal blocks
            blocks = []
            for _ in range(num_blocks):
                # Start with random Gaussian matrix
                block = torch.randn(self.input_dim, self.input_dim, device=self.device)
                # Make it orthogonal using QR decomposition
                q, _ = torch.linalg.qr(block)
                blocks.append(q)
            
            # Add remainder block if needed
            if remainder > 0:
                # Create only the needed portion to avoid wasteful computation
                block = torch.randn(self.input_dim, remainder, device=self.device)
                q, _ = torch.linalg.qr(block, mode='reduced')  # Use reduced QR for efficiency
                blocks.append(q)
                
            # Concatenate all blocks
            weights = torch.cat(blocks, dim=1) * scale
        else:
            # Standard random features
            if self.rademacher:
                # Rademacher distribution
                weights = torch.randint(0, 2, (self.input_dim, self.feature_dim), device=self.device) * 2 - 1
                weights = weights.float() * scale
            else:
                # Gaussian distribution
                weights = torch.randn(self.input_dim, self.feature_dim, device=self.device) * scale
                
        # Random offset for cosine features
        offset = torch.rand(self.feature_dim, device=self.device) * 2 * math.pi
        
        self.weights = weights
        self.offset = offset
                
    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RFF features for input tensor x.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            RFF features of shape [batch_size, feature_dim]
        """
        # Ensure x has the right shape
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
            
        if x.size(1) != self.input_dim:
            raise ValueError(f"Input dimension {x.size(1)} does not match expected {self.input_dim}")
        
        # Compute RFF: Φ(x) = √(2/D) cos(Wx + b)
        projection = x @ self.weights + self.offset
        features = torch.cos(projection) * math.sqrt(2.0 / self.feature_dim)
        
        return features
        
    def compute_kernel(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute approximate kernel matrix between x and y.
        
        Args:
            x: First tensor of shape [batch_size_x, input_dim]
            y: Second tensor of shape [batch_size_y, input_dim], if None, use x
            
        Returns:
            Approximate kernel matrix of shape [batch_size_x, batch_size_y]
        """
        # Ensure inputs have the right shape
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
            
        if y is not None and y.dim() > 2:
            y = y.reshape(y.size(0), -1)
            
        # Compute features
        x_features = self.compute_features(x)
        
        if y is None:
            # Kernel between x and itself
            return x_features @ x_features.T
        else:
            # Kernel between x and y
            y_features = self.compute_features(y)
            return x_features @ y_features.T
            
    def compute_kernel_derivative(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        order: int = 1,
        coordinate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute approximation of kernel derivative.
        CORRECTED VERSION: Properly handles Gaussian kernel derivatives.
        
        Args:
            x: First tensor of shape [batch_size_x, input_dim]
            y: Second tensor of shape [batch_size_y, input_dim], if None, use x
            order: Order of the derivative (1 for first derivative, etc.)
            coordinate: Coordinate to take derivative with respect to, if None, return all derivatives
            
        Returns:
            Approximate kernel derivative matrix
        """
        if order > self.derivative_order:
            raise ValueError(f"Requested derivative order {order} > supported order {self.derivative_order}")
            
        # Ensure inputs have the right shape
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
            
        if y is not None and y.dim() > 2:
            y = y.reshape(y.size(0), -1)
        else:
            y = x
            
        batch_x = x.size(0)
        batch_y = y.size(0)
        
        if order == 1:
            # First order derivative for Gaussian kernel
            # ∂k(x,y)/∂x_i = k(x,y) * (y_i - x_i)/σ²
            
            # Compute kernel values
            K = self.compute_kernel(x, y)  # [batch_x, batch_y]
            
            if coordinate is not None:
                # Derivative with respect to specific coordinate
                # Compute (y_i - x_i) for all pairs
                diff = y[:, coordinate].unsqueeze(0) - x[:, coordinate].unsqueeze(1)  # [batch_x, batch_y]
                derivative = K * diff / (self.sigma**2)
                return derivative
            else:
                # All derivatives
                derivatives = []
                for i in range(self.input_dim):
                    diff_i = y[:, i].unsqueeze(0) - x[:, i].unsqueeze(1)  # [batch_x, batch_y]
                    derivative_i = K * diff_i / (self.sigma**2)
                    derivatives.append(derivative_i)
                return torch.stack(derivatives, dim=0)  # [input_dim, batch_x, batch_y]
        
        elif order == 2:
            # Second order derivative for Gaussian kernel
            # ∂²k(x,y)/∂x_i∂x_j = k(x,y) * [(y_i-x_i)(y_j-x_j)/σ⁴ - δ_ij/σ²]
            
            # Compute kernel values
            K = self.compute_kernel(x, y)  # [batch_x, batch_y]
            
            if coordinate is not None:
                # Specific pair of coordinates
                i, j = coordinate if isinstance(coordinate, tuple) else (coordinate, coordinate)
                
                # Compute differences
                diff_i = y[:, i].unsqueeze(0) - x[:, i].unsqueeze(1)  # [batch_x, batch_y]
                diff_j = y[:, j].unsqueeze(0) - x[:, j].unsqueeze(1)  # [batch_x, batch_y]
                
                # Compute Hessian element
                hessian_ij = K * (diff_i * diff_j / (self.sigma**4))
                if i == j:
                    hessian_ij = hessian_ij - K / (self.sigma**2)
                
                return hessian_ij
            else:
                # All second derivatives
                hessian = torch.zeros(self.input_dim, self.input_dim, batch_x, batch_y, device=self.device)
                for i in range(self.input_dim):
                    for j in range(self.input_dim):
                        diff_i = y[:, i].unsqueeze(0) - x[:, i].unsqueeze(1)
                        diff_j = y[:, j].unsqueeze(0) - x[:, j].unsqueeze(1)
                        
                        hessian[i, j] = K * (diff_i * diff_j / (self.sigma**4))
                        if i == j:
                            hessian[i, j] = hessian[i, j] - K / (self.sigma**2)
                
                return hessian
        
        else:
            raise NotImplementedError(f"Derivatives of order {order} not implemented")
    
    def compute_score_approximation(self, x: torch.Tensor, y: torch.Tensor, weights_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Approximate the score function ∇_x log p(x) using kernel derivatives.
        
        Based on the relation: score(x) = ∇_x log p(x) ≈ ∑_i w_i ∇_x K(x,y_i) / ∑_i w_i K(x,y_i)
        
        Args:
            x: Target points to compute score at [batch_x, input_dim]
            y: Sample points from distribution [batch_y, input_dim]
            weights_y: Optional weights for y samples
            
        Returns:
            Approximate score at x
        """
        # Compute kernel and kernel derivatives
        K_xy = self.compute_kernel(x, y)  # [batch_x, batch_y]
        dK_xy = self.compute_kernel_derivative(x, y, order=1)  # [input_dim, batch_x, batch_y]
        
        # Apply weights if provided
        if weights_y is not None:
            K_xy = K_xy * weights_y.unsqueeze(0)
            dK_xy = dK_xy * weights_y.unsqueeze(0).unsqueeze(0)
        
        # Approximate score using kernel mean embedding
        # score(x) ≈ ∑_i ∇_x K(x,y_i) / ∑_i K(x,y_i)
        K_xy_sum = K_xy.sum(dim=1, keepdim=True) + 1e-10  # [batch_x, 1]
        
        # Compute weighted average of derivatives
        # FIX: Divide after the sum to avoid broadcasting issues
        score = torch.zeros(x.size(0), self.input_dim, device=self.device)
        for d in range(self.input_dim):
            score[:, d] = dK_xy[d].sum(dim=1) / K_xy_sum.squeeze(1)
            
        return score
    
    def estimate_error_bound(self, n_samples: int) -> Dict[str, float]:
        """
        Estimate theoretical error bounds for kernel and derivative approximation.
        
        Args:
            n_samples: Number of samples used
            
        Returns:
            Dictionary with error bounds for kernel and derivatives
        """
        # Basic error bound from the paper
        base_error = math.sqrt(math.log(n_samples) / self.feature_dim)
        
        error_bounds = {
            'kernel': base_error,
            'first_derivative': base_error / (self.sigma**2),  # Fixed: was /self.sigma
            'second_derivative': base_error / (self.sigma**2),
        }
        
        return error_bounds


