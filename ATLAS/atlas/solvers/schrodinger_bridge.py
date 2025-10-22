
import inspect
import logging
import math
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.kernel_config import KernelConfig
from ..config.sampler_config import SamplerConfig
from ..kernels import (
    KernelOperator,
    DirectKernelOperator,
    FFTKernelOperator,
    NystromKernelOperator,
    RFFKernelOperator,
)
from ..utils.random import set_seed


class SchrodingerBridgeSolver:
    """
    Unified Schrodinger Bridge solver based on operator theory.
    
    This implementation unifies different computational approaches under
    a common operator-theoretic framework, representing the SB problem
    as finding the fixed point of an RKHS operator.
    """
    def __init__(
        self,
        score_model: nn.Module, 
        noise_schedule: Callable,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        kernel_config: Optional[KernelConfig] = None,
        sampler_config: Optional[SamplerConfig] = None,
    ):
        """
        Initialize the unified SB solver.
        
        Args:
            score_model: Neural network predicting the score
            noise_schedule: Function returning noise level at time t
            device: Device to run computations on
            kernel_config: Kernel configuration
            sampler_config: Sampler configuration
        """
        self.score_model = score_model
        self.noise_schedule = noise_schedule
        self.device = device
        
        # Use provided configs or set defaults
        if kernel_config is None:
            kernel_config = KernelConfig()
        if sampler_config is None:
            sampler_config = SamplerConfig()
            
        # Extract configuration parameters
        self.epsilon = kernel_config.epsilon
        self.adaptive_epsilon = kernel_config.adaptive_epsilon
        self.solver_type = kernel_config.solver_type
        self.kernel_type = kernel_config.kernel_type
        self.rff_features = kernel_config.rff_features
        self.n_landmarks = kernel_config.n_landmarks
        
        self.sb_iterations = sampler_config.sb_iterations
        self.error_tolerance = sampler_config.error_tolerance
        self.use_linear_solver = sampler_config.use_linear_solver
        self.use_multiscale = kernel_config.multi_scale
        self.use_mixed_precision = sampler_config.use_mixed_precision
        self.scale_factors = kernel_config.scale_factors
        self.seed = sampler_config.seed
        
        # Set random seed if provided
        if self.seed is not None:
            set_seed(self.seed)
        
        # Initialize performance tracking
        self.perf_stats = {
            'total_time': 0.0,
            'kernel_time': 0.0,
            'sb_time': 0.0,
            'methods_used': {},
            'times_per_step': [],
            'memory_usage': [],
            'mean_weights': [],
            'hierarchy': [],
        }
        
        # Initialize logger
        self.logger = logging.getLogger("SchrodingerBridgeSolver")
        self.logger.setLevel(logging.INFO if sampler_config.verbose_logging else logging.WARNING)
        
        # Initialize kernel operator cache
        self.kernel_operators: OrderedDict[str, KernelOperator] = OrderedDict()
        self.max_kernel_cache_size = kernel_config.max_kernel_cache_size

    def _schedule_value_to_tensor(
        self, value: Union[float, torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        """Convert a schedule value to a tensor that matches a reference tensor."""
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device=reference.device, dtype=reference.dtype)
            if value.ndim == 0:
                return value
            return value.reshape(-1)[0]
        return torch.tensor(float(value), device=reference.device, dtype=reference.dtype)

    def _schedule_to_tensor(self, t: float, reference: torch.Tensor) -> torch.Tensor:
        """Evaluate the noise schedule at time ``t`` and return a tensor value."""
        return self._schedule_value_to_tensor(self.noise_schedule(t), reference)

    def _compute_sigma(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the noise scale sigma(t) from alpha(t)."""
        info = torch.finfo(alpha.dtype)
        alpha_clamped = torch.clamp(alpha, min=info.tiny, max=1.0 - info.tiny)
        one_minus_alpha = torch.clamp(alpha.new_tensor(1.0) - alpha_clamped, min=info.tiny)
        ratio = one_minus_alpha / alpha_clamped
        return torch.sqrt(ratio)

    def _compute_sde_coefficients(
        self, t: float, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift coefficients f(t) and g(t)^2 for the probability flow ODE."""
        alpha_t = self._schedule_to_tensor(t, reference)
        sigma_t = self._compute_sigma(alpha_t)

        delta = 1e-3
        t_upper = min(1.0, t + delta)
        t_lower = max(0.0, t - delta)

        if t_upper == t_lower:
            delta = 1e-4
            t_upper = min(1.0, t + delta)
            t_lower = max(0.0, t - delta)
            if t_upper == t_lower:
                zero = torch.zeros_like(sigma_t)
                return zero, zero

        alpha_upper = self._schedule_value_to_tensor(
            self.noise_schedule(t_upper), reference
        )
        alpha_lower = self._schedule_value_to_tensor(
            self.noise_schedule(t_lower), reference
        )

        sigma_upper = self._compute_sigma(alpha_upper)
        sigma_lower = self._compute_sigma(alpha_lower)

        denom = max(t_upper - t_lower, 1e-6)
        denom_tensor = sigma_t.new_tensor(denom)
        sigma_prime = (sigma_upper - sigma_lower) / denom_tensor

        sigma_safe = torch.clamp(sigma_t, min=torch.finfo(sigma_t.dtype).tiny)
        f_t = -sigma_prime / sigma_safe
        g_sq_t = -2.0 * sigma_t * sigma_prime
        g_sq_t = torch.clamp(g_sq_t, min=0.0)

        return f_t, g_sq_t

    def _compute_score(
        self,
        x: torch.Tensor,
        t: float,
        conditioning: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute score (gradient of log density) using the score model.
        
        Args:
            x: Input tensor
            t: Current time
        
        Returns:
            Score tensor
        """
        with torch.no_grad():
            forward_fn = self.score_model.forward if hasattr(self.score_model, "forward") else self.score_model
            t_tensor = torch.full((x.shape[0],), t, device=x.device)
            noise_pred = None
            
            if conditioning is not None:
                try:
                    signature = inspect.signature(forward_fn)
                except (TypeError, ValueError):
                    signature = None
                
                if signature is not None:
                    params = signature.parameters
                    if "condition" in params:
                        noise_pred = self.score_model(x, t_tensor, condition=conditioning)
                    elif "conditioning" in params:
                        noise_pred = self.score_model(x, t_tensor, conditioning=conditioning)
                    elif isinstance(conditioning, dict):
                        accepted_kwargs = {k: v for k, v in conditioning.items() if k in params}
                        if accepted_kwargs:
                            noise_pred = self.score_model(x, t_tensor, **accepted_kwargs)
                else:
                    # Fallback if signature inspection fails; prefer explicit kw name
                    try:
                        noise_pred = self.score_model(x, t_tensor, condition=conditioning)
                    except TypeError:
                        pass
            
            if noise_pred is None:
                noise_pred = self.score_model(x, t_tensor)
        
        # Convert noise prediction to score
        alpha_t = self._schedule_to_tensor(t, noise_pred)
        denom = torch.sqrt(torch.clamp(1 - alpha_t, min=torch.finfo(noise_pred.dtype).tiny))
        score = -noise_pred / denom

        return score

    def _compute_drift(
        self,
        x: torch.Tensor,
        t: float,
        dt: float,
        conditioning: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute the drift term for the transport map using score function.
        
        Args:
            x: Input tensor
            t: Current time
            dt: Time step
            conditioning: Optional conditioning payload
        
        Returns:
            Drift tensor
        """
        score = self._compute_score(x, t, conditioning=conditioning)
        f_t, g_sq_t = self._compute_sde_coefficients(t, score)

        # Probability flow ODE drift: f(t) * x + g(t)^2 * score / 2
        drift = f_t * x + 0.5 * g_sq_t * score

        return drift * dt
    
    def _is_grid_structured(self, x: torch.Tensor) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if data is structured on a regular grid.

        Args:
            x: Input tensor

        Returns:
            Tuple of (is_grid, grid_shape)
        """
        # If x has more than 2 dimensions, it's likely already in grid format
        if x.dim() > 2:
            return True, list(x.shape[1:])

        # Try to infer grid structure from coordinates
        if x.dim() == 2:
            ndim = x.size(1)
            if ndim <= 3:  # Only try for 1D, 2D, or 3D data
                # Compute unique coordinates once per dimension (performance optimization)
                unique_coords_list = [torch.unique(x[:, d]) for d in range(ndim)]

                # Check if number of unique coordinates is too high for any dimension
                for unique_coords in unique_coords_list:
                    if len(unique_coords) > 100:
                        return False, None

                # Build grid shape from cached unique coordinates
                grid_shape = [len(unique_coords) for unique_coords in unique_coords_list]

                # Check if product of grid shape matches batch size
                if np.prod(grid_shape) == x.size(0):
                    return True, grid_shape

        return False, None
    
    def _select_optimal_kernel_operator(
        self, 
        x: torch.Tensor, 
        epsilon: float
    ) -> KernelOperator:
        """
        Select the optimal kernel operator based on data characteristics.
        
        This method chooses the most appropriate kernel operator implementation
        based on data shape, size, dimensionality, and the problem constraints.
        
        Args:
            x: Input data
            epsilon: Regularization parameter
            
        Returns:
            Appropriate KernelOperator instance
        """
        batch_size, *data_dims = x.shape
        
        # Get data dimensionality
        if len(data_dims) > 0:
            data_dim = np.prod(data_dims)
        else:
            data_dim = x.size(1) if x.dim() > 1 else 1
            
        # Check if data is grid-structured
        is_grid, grid_shape = self._is_grid_structured(x)
        
        # Set method based on theoretical error bounds or specific request
        if self.solver_type != 'auto':
            method = self.solver_type
        elif is_grid and len(grid_shape) <= 3:
            # Grid data is best handled by FFT
            method = 'fft'
        elif batch_size <= 1000:
            # Small problems: use direct method
            method = 'direct'
        elif data_dim > 100:
            # High-dimensional data: use RFF
            method = 'rff'
        elif batch_size > 10000:
            # Very large batch: need efficient approximation
            if data_dim < 20:
                # Low dimensional data works well with Nystrom
                method = 'nystrom'
            else:
                # Higher dimensions: use RFF
                method = 'rff'
        else:
            # Default to RFF as a balanced choice
            method = 'rff'
        
        # Track method usage
        if method in self.perf_stats['methods_used']:
            self.perf_stats['methods_used'][method] += 1
        else:
            self.perf_stats['methods_used'][method] = 1
            
        # Create and return the appropriate kernel operator
        kernel_start_time = time.time()
        
        # Check for special case: FFT requested but data is not grid-structured
        if method == 'fft' and not is_grid:
            self.logger.warning("Data is not grid-structured, falling back to RFF")
            # Use RFF instead of recursively calling this function
            method = 'rff'
        
        # Cache key based on batch and data dimensions
        cache_key = f"{method}_{batch_size}_{data_dim}_{epsilon:.5f}"
        
        operator = self.kernel_operators.get(cache_key)
        if operator is not None:
            self.kernel_operators.move_to_end(cache_key)
            self.logger.debug(f"Using cached {method} operator")
        else:
            if method == 'direct':
                operator = DirectKernelOperator(
                    kernel_type=self.kernel_type,
                    epsilon=epsilon,
                    device=self.device
                )
            elif method == 'rff':
                operator = RFFKernelOperator(
                    input_dim=data_dim,
                    feature_dim=self.rff_features,
                    kernel_type=self.kernel_type,
                    epsilon=epsilon,
                    device=self.device,
                    orthogonal=True,
                    multi_scale=self.use_multiscale,
                    scale_factors=self.scale_factors,
                    seed=self.seed
                )
            elif method == 'nystrom':
                # Select landmarks (either random or based on score magnitudes)
                if batch_size <= self.n_landmarks:
                    landmarks = x
                else:
                    # Random selection for simplicity
                    indices = torch.randperm(batch_size, device=self.device)[:self.n_landmarks]
                    landmarks = x[indices]
                
                operator = NystromKernelOperator(
                    landmarks=landmarks,
                    kernel_type=self.kernel_type,
                    epsilon=epsilon,
                    device=self.device,
                    seed=self.seed
                )
            elif method == 'fft':
                operator = FFTKernelOperator(
                    grid_shape=grid_shape,
                    kernel_type=self.kernel_type,
                    epsilon=epsilon,
                    device=self.device,
                    multi_scale=self.use_multiscale,
                    scale_factors=self.scale_factors
                )
            else:
                raise ValueError(f"Unknown method: {method}")
                
            # Cache the operator with LRU eviction
            self.kernel_operators[cache_key] = operator
            self.kernel_operators.move_to_end(cache_key)
            
            while len(self.kernel_operators) > self.max_kernel_cache_size:
                evicted_key, evicted_operator = self.kernel_operators.popitem(last=False)
                if hasattr(evicted_operator, "clear_cache"):
                    try:
                        evicted_operator.clear_cache()
                    except Exception:
                        pass
                self.logger.debug(f"Evicted kernel operator cache entry {evicted_key}")
        
        self.perf_stats['kernel_time'] += time.time() - kernel_start_time
        
        return operator
    
    def _solve_Schrodinger_bridge(
        self, 
        kernel_op: KernelOperator,
        x: torch.Tensor,
        max_iter: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the Schrodinger Bridge problem using the provided kernel operator.
        
        This implements both traditional iterative approach and Light SB's linear system approach.
        
        Args:
            kernel_op: Kernel operator instance
            x: Input data
            max_iter: Maximum iterations
            
        Returns:
            Tuple of potential functions (f, g)
        """
        batch_size = x.size(0)
        
        # Choose between Light SB approach (linear system) or traditional iteration
        if self.use_linear_solver:
            # Light SB approach: solve (I - K)f = 1 directly
            # Initialize with ones
            f = torch.ones(batch_size, device=self.device)
            
            # Set up linear operator for conjugate gradient
            def linear_op(v):
                return v - kernel_op.apply(x, v)
            
            # Right-hand side is vector of ones
            b = torch.ones(batch_size, device=self.device)
            
            # Solve using conjugate gradient
            f, convergence = self._conjugate_gradient(
                linear_op, b, x0=f, max_iter=max_iter, tol=self.error_tolerance
            )
            
            # Compute g from f
            Kf = kernel_op.apply(x, f)
            g = torch.ones_like(f) / (Kf + 1e-10)
        else:
            # Traditional iterative approach
            # Initialize potentials
            f = torch.ones(batch_size, device=self.device)
            g = torch.ones_like(f)
            
            # Iterative updates
            for i in range(max_iter):
                f_prev = f.clone()
                
                # Update f: f = 1 / (K @ g)
                Kg = kernel_op.apply(x, g)
                f = 1.0 / (Kg + 1e-10)
                
                # Update g: g = 1 / (K^T @ f)
                KTf = kernel_op.apply_transpose(x, f)
                g = 1.0 / (KTf + 1e-10)
                
                # Check convergence
                if torch.max(torch.abs(f - f_prev)) < self.error_tolerance:
                    break
        
        return f, g
    
    def _conjugate_gradient(
        self, 
        A_func: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        max_iter: int = 50,
        tol: float = 1e-5
    ) -> Tuple[torch.Tensor, bool]:
        """
        Conjugate gradient method for solving Ax = b.
        
        Args:
            A_func: Function that computes A@x
            b: Right-hand side vector
            x0: Initial guess (optional)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple of (solution, converged)
        """
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()
            
        r = b - A_func(x)
        p = r.clone()
        rsold = torch.sum(r * r)
        
        # Early exit if already below tolerance
        if torch.sqrt(rsold) < tol:
            return x, True
        
        for i in range(max_iter):
            Ap = A_func(p)
            
            # Compute denominator safely
            denom = torch.sum(p * Ap)
            
            # Check for nearly zero or negative denominator
            if denom < 1e-12:
                denom_value = denom.detach().cpu().item()
                self.logger.warning(
                    "Conjugate gradient restart at iteration %d due to near-zero curvature (denominator %.3e).",
                    i,
                    denom_value,
                )
                # If residual is already small enough, return current solution
                if torch.sqrt(rsold) < tol * 10:  # Slightly relaxed tolerance
                    return x, True
                
                # Otherwise restart with steepest descent direction
                p = r.clone()
                Ap = A_func(p)
                denom = torch.sum(p * Ap)
                
                # If still problematic, return current best solution
                if denom < 1e-12:
                    self.logger.warning(
                        "Conjugate gradient failed to recover after restart at iteration %d; returning partial solution.",
                        i,
                    )
                    return x, False
            
            alpha = rsold / denom
            x = x + alpha * p
            
            # Periodically compute full residual to avoid drift
            if i % 10 == 0:
                r = b - A_func(x)
            else:
                r = r - alpha * Ap
            
            rsnew = torch.sum(r * r)
            rsnorm = torch.sqrt(rsnew)
            
            # Check convergence
            if rsnorm < tol:
                return x, True
            
            # If the residual is increasing or unstable, exit early
            if rsnew > rsold * 1.5:
                prev_residual = torch.sqrt(rsold).detach().cpu().item()
                new_residual = rsnorm.detach().cpu().item()
                self.logger.warning(
                    "Conjugate gradient residual increased at iteration %d (%.3e -> %.3e); returning partial solution.",
                    i,
                    prev_residual,
                    new_residual,
                )
                # Return current solution but indicate not fully converged
                return x, False
                
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
            
        final_residual = torch.sqrt(rsold).detach().cpu().item()
        self.logger.warning(
            "Conjugate gradient reached max iterations (%d) without convergence; residual norm %.3e.",
            max_iter,
            final_residual,
        )
        return x, False
    
    def _construct_transport_map(
        self,
        f: torch.Tensor,
        g: torch.Tensor,
        kernel_op: KernelOperator,
        x_curr: torch.Tensor,
        x_next_pred: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Construct the optimal transport map based on potentials and kernel.
        
        Args:
            f: First potential function
            g: Second potential function
            kernel_op: Kernel operator instance
            x_curr: Current points
            x_next_pred: Predicted next points
            
        Returns:
            Transport map function
        """
        batch_size = x_curr.size(0)
        
        # Pre-compute some values for efficiency while maintaining numerical stability
        # Use more conservative bounds to prevent numerical overflow in downstream operations
        fg_raw = torch.nan_to_num(f * g, nan=1e-10, posinf=1e6, neginf=1e-10)
        fg = torch.clamp(fg_raw, min=1e-10, max=1e6)  # [batch_size]
        
        def transport_map(z: torch.Tensor) -> torch.Tensor:
            """
            Apply the optimal transport map to new points z.
            
            Args:
                z: Input points [batch_z, ...]
                
            Returns:
                Transported points [batch_z, ...]
            """
            z_shape = z.shape
            z_flat = z.reshape(z.size(0), -1)
            x_curr_flat = x_curr.reshape(batch_size, -1)
            x_next_flat = x_next_pred.reshape(batch_size, -1)
            
            # Need a different approach for each kernel operator type
            if isinstance(kernel_op, DirectKernelOperator):
                # For direct kernel, implement efficient barycentric projection
                # FIXED: Use element-wise multiplication with broadcasting
                K_zx = kernel_op._compute_kernel_matrix(z_flat, x_curr_flat)
                P_zx = K_zx * fg.unsqueeze(0)  # [batch_z, batch_size]
                
                # Normalize weights
                row_sums = torch.clamp(P_zx.sum(dim=1, keepdim=True), min=1e-10)
                P_zx_norm = P_zx / row_sums
                
                # Apply transport
                z_next = P_zx_norm @ x_next_flat
                
            elif isinstance(kernel_op, RFFKernelOperator):
                # For RFF, use feature space operations
                z_features = kernel_op.compute_features(z_flat)
                x_features = kernel_op.compute_features(x_curr_flat)
                
                # FIXED: Efficient computation using element-wise multiplication
                weighted_features = x_features * fg.unsqueeze(1)  # [batch_size, feature_dim]
                
                # Apply to compute transport weights
                P_zx = z_features @ weighted_features.T  # [batch_z, batch_size]
                
                # Normalize weights
                row_sums = torch.clamp(P_zx.sum(dim=1, keepdim=True), min=1e-10)
                P_zx_norm = P_zx / row_sums
                
                # Apply transport
                z_next = P_zx_norm @ x_next_flat
                
            elif isinstance(kernel_op, NystromKernelOperator):
                # For Nystrom, use landmark-based transport
                K_zl = kernel_op._compute_kernel(z_flat, kernel_op.landmarks)
                
                # FIXED: Compute weights efficiently
                landmark_size = min(kernel_op.n_landmarks, batch_size)
                landmark_fg = fg[:landmark_size]
                
                # Approximate transport weights using landmarks
                P_zl = K_zl * landmark_fg.unsqueeze(0)  # [batch_z, n_landmarks]
                
                # Normalize weights
                row_sums = torch.clamp(P_zl.sum(dim=1, keepdim=True), min=1e-10)
                P_zl_norm = P_zl / row_sums
                
                # Apply transport
                z_next = P_zl_norm @ x_next_flat[:landmark_size]
                
            elif isinstance(kernel_op, FFTKernelOperator):
                # For FFT, use grid-based transport
                # This is a simplified version - more accurate would use gradient of potentials
                alpha = 0.5  # Interpolation parameter
                z_next = (1 - alpha) * z_flat + alpha * x_next_flat
            else:
                # Generic fallback
                # Create a simple interpolation between current and predicted
                z_next = 0.5 * z_flat + 0.5 * x_next_flat
            
            return z_next.reshape(z_shape)
        
        return transport_map
    
    def solve_once(
        self,
        x: torch.Tensor,
        t_curr: float,
        t_next: float,
        conditioning: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Perform a single SB step from t_curr to t_next.
        
        Args:
            x: Current state tensor
            t_curr: Current timestep
            t_next: Next timestep
            conditioning: Optional conditioning payload
            
        Returns:
            Next state tensor
        """
        dt = t_curr - t_next
        
        # Compute predicted next position using score-based drift
        y_pred = x + self._compute_drift(x, t_curr, dt, conditioning=conditioning)
        
        # Adjust regularization parameter if adaptive
        eps = self.epsilon
        if self.adaptive_epsilon:
            alpha_t = self.noise_schedule(t_curr)
            if isinstance(alpha_t, torch.Tensor):
                alpha_value = float(alpha_t.detach().cpu().item())
            else:
                alpha_value = float(alpha_t)
            alpha_value = max(0.0, min(1.0, alpha_value))
            eps = self.epsilon * math.sqrt(max(1e-12, 1.0 - alpha_value))
        
        # Select optimal kernel operator
        kernel_op = self._select_optimal_kernel_operator(x, eps)
        self._last_kernel_method = type(kernel_op).__name__.replace('KernelOperator', '')
        
        # Solve Schrodinger Bridge problem
        sb_start_time = time.time()
        f, g = self._solve_Schrodinger_bridge(
            kernel_op, x, max_iter=self.sb_iterations
        )
        self.perf_stats['sb_time'] += time.time() - sb_start_time
        
        # Construct optimal transport map
        transport_map = self._construct_transport_map(
            f, g, kernel_op, x, y_pred
        )
        
        # Apply transport map
        x_next = transport_map(x)
        
        return x_next
    
    def sample(
        self,
        shape: Tuple[int, ...],
        timesteps: List[float],
        verbose: bool = True,
        callback: Optional[Callable] = None,
        conditioning: Optional[Dict[str, Any]] = None,
        prompts: Optional[List[str]] = None,
        negative_prompts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Generate samples using the optimal transport map.

        NOTE: For advanced features like hierarchical sampling and CLIP conditioning,
        use AdvancedHierarchicalDiffusionSampler instead. This is a simplified sampling loop.

        Args:
            shape: Shape of samples to generate (batch_size, *data_dims)
            timesteps: List of timesteps for the reverse process (decreasing)
            verbose: Whether to show progress bar
            callback: Optional callback function after each step
            conditioning: Precomputed conditioning payload
            prompts: Ignored in this simplified sampler (use AdvancedHierarchicalDiffusionSampler)
            negative_prompts: Ignored in this simplified sampler (use AdvancedHierarchicalDiffusionSampler)

        Returns:
            Tensor of generated samples
        """
        if len(shape) < 2:
            raise ValueError("Shape must include batch and channel dimensions.")

        timesteps = sorted(timesteps, reverse=True)
        if len(timesteps) < 2:
            raise ValueError("Timesteps must contain at least two values in descending order.")

        # Reset performance accumulators
        self.perf_stats['times_per_step'].clear()
        self.perf_stats['memory_usage'].clear()
        self.perf_stats['mean_weights'].clear()
        self.perf_stats['hierarchy'] = []
        self.perf_stats['kernel_time'] = 0.0
        self.perf_stats['sb_time'] = 0.0

        total_start_time = time.time()

        # Initialize with random noise
        x_t = torch.randn(shape, device=self.device)

        # Simplified sampling loop without hierarchical features
        # For advanced features, use AdvancedHierarchicalDiffusionSampler
        from tqdm import tqdm
        iterator = range(len(timesteps) - 1)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling", leave=False)

        for idx in iterator:
            t_curr = float(timesteps[idx])
            t_next = float(timesteps[idx + 1])

            # Perform single SB step
            x_t = self.solve_once(
                x_t,
                t_curr,
                t_next,
                conditioning=conditioning,
            )

            if callback is not None:
                callback(x_t, t_curr, t_next)

        self.perf_stats['total_time'] = time.time() - total_start_time
        return x_t
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the solver.
        
        Returns:
            Dictionary of performance metrics
        """
        # Compute derived metrics
        if self.perf_stats['total_time'] > 0:
            self.perf_stats['kernel_pct'] = 100 * self.perf_stats['kernel_time'] / self.perf_stats['total_time']
            self.perf_stats['sb_pct'] = 100 * self.perf_stats['sb_time'] / self.perf_stats['total_time']
        
        return self.perf_stats.copy()

    def auto_tune_parameters(self, x: torch.Tensor, error_tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Auto-tune parameters based on theoretical error bounds and data characteristics.
        
        Args:
            x: Representative input data
            error_tolerance: Target error tolerance
            
        Returns:
            Dictionary of tuned parameters
        """
        n_samples = x.shape[0]
        results = {}
        
        # Tune RFF parameters
        if self.solver_type == 'rff' or self.solver_type == 'auto':
            self.rff_features = 1024  # Starting point
            
            # Create initial RFF operator
            input_dim = x.shape[1] if x.dim() == 2 else np.prod(x.shape[1:])
            rff_op = RFFKernelOperator(
                input_dim=input_dim,
                feature_dim=self.rff_features,
                kernel_type=self.kernel_type,
                epsilon=self.epsilon,
                device=self.device
            )
            
            # Increase features until error bound is satisfied
            while rff_op.get_error_bound(n_samples) > error_tolerance:
                self.rff_features *= 2
                rff_op = RFFKernelOperator(
                    input_dim=input_dim,
                    feature_dim=self.rff_features,
                    kernel_type=self.kernel_type,
                    epsilon=self.epsilon,
                    device=self.device
                )
                
                # Cap at reasonable maximum
                if self.rff_features > 32768:
                    break
                    
            results['rff_features'] = self.rff_features
            results['rff_error_bound'] = rff_op.get_error_bound(n_samples)
        
        # Tune Nystrom parameters
        if self.solver_type == 'Nystrom' or self.solver_type == 'auto':
            self.n_landmarks = min(100, n_samples // 10)  # Starting point
            
            while self.n_landmarks < n_samples:
                # Increase landmarks until error bound is satisfied
                error_bound = math.sqrt(n_samples / self.n_landmarks)
                
                if error_bound <= error_tolerance:
                    break
                
                self.n_landmarks = min(self.n_landmarks * 2, n_samples)
                
            results['n_landmarks'] = self.n_landmarks
            results['Nystrom_error_bound'] = math.sqrt(n_samples / self.n_landmarks)
            
        return results


#############################################
# Advanced Hierarchical Diffusion Sampler   #
