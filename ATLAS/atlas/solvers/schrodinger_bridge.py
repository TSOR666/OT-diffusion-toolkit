
import logging
import math
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.cuda.amp import autocast
except Exception:  # pragma: no cover - autocast optional
    autocast = None

from ..config.kernel_config import KernelConfig
from ..config.sampler_config import SamplerConfig
from ..kernels import (
    KernelOperator,
    DirectKernelOperator,
    FFTKernelOperator,
    NystromKernelOperator,
    RFFKernelOperator,
)
from ..utils.noise_prediction import NoisePredictionAdapter
from ..utils.random import set_seed


_VALID_KERNEL_TYPES = {"gaussian", "laplacian", "cauchy"}


class SchroedingerBridgeSolver:
    """
    Unified Schrodinger Bridge solver based on operator theory.
    
    This implementation unifies different computational approaches under
    a common operator-theoretic framework, representing the SB problem
    as finding the fixed point of an RKHS operator.

    The ``noise_schedule`` callable is expected to return alpha(t) values in (0, 1]
    that define the noise-to-signal ratio used to compute sigma(t).
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
            noise_schedule: Function returning alpha(t) in (0, 1] at time t
            device: Device to run computations on
            kernel_config: Kernel configuration
            sampler_config: Sampler configuration
        """
        self.noise_schedule = noise_schedule
        self.device = device
        self._score_model: nn.Module
        self.noise_predictor: NoisePredictionAdapter
        self.score_model = score_model
        
        # Use provided configs or set defaults
        if kernel_config is None:
            kernel_config = KernelConfig()
        if sampler_config is None:
            sampler_config = SamplerConfig()
            
        # Extract configuration parameters
        self.epsilon = kernel_config.epsilon
        self.adaptive_epsilon = kernel_config.adaptive_epsilon
        self.solver_type = kernel_config.solver_type

        kernel_type = kernel_config.kernel_type.lower()
        if kernel_type not in _VALID_KERNEL_TYPES:
            raise ValueError(
                f"Unsupported kernel type '{kernel_config.kernel_type}'. "
                f"Supported types are: {sorted(_VALID_KERNEL_TYPES)}"
            )
        self.kernel_type = kernel_type
        self.rff_features = kernel_config.rff_features
        self.n_landmarks = kernel_config.n_landmarks

        self.sb_iterations = sampler_config.sb_iterations
        self.error_tolerance = sampler_config.error_tolerance
        self.use_linear_solver = sampler_config.use_linear_solver
        self.use_multiscale = kernel_config.multi_scale
        self.use_mixed_precision = (
            sampler_config.use_mixed_precision
            and self.device.type == "cuda"
            and autocast is not None
        )
        self.scale_factors = kernel_config.scale_factors
        self.seed = sampler_config.seed
        self.cg_relative_tolerance = sampler_config.cg_relative_tolerance
        self.cg_absolute_tolerance = sampler_config.cg_absolute_tolerance
        if self.use_mixed_precision:
            self._amp_context = lambda: autocast(device_type="cuda", dtype=torch.float16)  # type: ignore[call-arg]
        else:
            self._amp_context = nullcontext

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
        self.logger = logging.getLogger("SchroedingerBridgeSolver")
        self.logger.setLevel(logging.INFO if sampler_config.verbose_logging else logging.WARNING)
        
        # Initialize kernel operator cache
        self.kernel_operators: "OrderedDict[Tuple[Any, ...], KernelOperator]" = OrderedDict()
        self.max_kernel_cache_size = int(kernel_config.max_kernel_cache_size)
        if self.max_kernel_cache_size <= 0:
            raise ValueError("max_kernel_cache_size must be positive")

        # Numerical guard for conjugate gradient denominators (dtype-aware, computed dynamically)
        # We use a multiplier of eps to ensure numerical stability across different dtypes
        self._min_curvature_multiplier = 1000.0

        # Instrumentation fields
        self.perf_stats.update(
            {
                "kernel_cache_hits": 0,
                "kernel_cache_misses": 0,
                "kernel_cache_evictions": 0,
                "cg_iterations_total": 0,
                "cg_solve_count": 0,
                "cg_restart_count": 0,
                "cg_failure_count": 0,
                "cg_last_residual": None,
            }
        )

    @property
    def score_model(self) -> nn.Module:
        return self._score_model

    @score_model.setter
    def score_model(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("score_model must be a torch.nn.Module")
        self._score_model = model
        self.noise_predictor = NoisePredictionAdapter(model)

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
        """
        Evaluate the noise schedule at time ``t`` and return alpha(t) as a tensor.

        The schedule must produce values in the open interval (0, 1], which are later
        converted to sigma(t) using the standard sigma^2 = (1 - alpha) / alpha relation.
        """
        return self._schedule_value_to_tensor(self.noise_schedule(t), reference)

    def _compute_sigma(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute the noise scale sigma(t) from alpha(t): sigma^2 = (1-alpha)/alpha."""
        info = torch.finfo(alpha.dtype)
        alpha_clamped = torch.clamp(alpha, min=info.tiny, max=1.0)
        one_minus_alpha = torch.clamp(alpha.new_tensor(1.0) - alpha_clamped, min=info.tiny)
        return torch.sqrt(one_minus_alpha / alpha_clamped)

    def _compute_sde_coefficients(
        self, t: float, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute drift coefficients f(t) and g(t)^2 for the probability flow ODE."""
        alpha_t = self._schedule_to_tensor(t, reference)
        delta = 1e-3
        t_upper = min(1.0, t + delta)
        t_lower = max(0.0, t - delta)

        if t_upper == t_lower:
            delta = 1e-4
            t_upper = min(1.0, t + delta)
            t_lower = max(0.0, t - delta)
            if t_upper == t_lower:
                zero = torch.zeros_like(alpha_t)
                return zero, zero

        alpha_upper = self._schedule_value_to_tensor(
            self.noise_schedule(t_upper), reference
        )
        alpha_lower = self._schedule_value_to_tensor(
            self.noise_schedule(t_lower), reference
        )

        denom = max(t_upper - t_lower, 1e-6)
        denom_tensor = alpha_t.new_tensor(denom)
        alpha_prime = (alpha_upper - alpha_lower) / denom_tensor

        info = torch.finfo(alpha_t.dtype)
        alpha_safe = torch.clamp(alpha_t, min=info.tiny)
        beta_t = torch.clamp(-alpha_prime / alpha_safe, min=0.0)
        f_t = -0.5 * beta_t
        g_sq_t = beta_t

        return f_t, torch.clamp(g_sq_t, min=0.0)

    def _compute_score(
        self,
        x: torch.Tensor,
        t: float,
        conditioning: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute the score (gradient of log density) using the noise predictor.
        """
        if x.ndim == 0:
            raise ValueError("Input tensor must include a batch dimension for scoring.")

        with torch.no_grad():
            with self._amp_context():
                noise_pred = self.noise_predictor.predict_noise(x, t, conditioning)
        noise_pred = noise_pred.to(x.dtype)

        alpha_t = self._schedule_to_tensor(t, noise_pred)
        sigma_t = self._compute_sigma(alpha_t)
        score = -noise_pred / sigma_t
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
        # Treat tensors with explicit spatial dimensions as grid data while keeping
        # channel dimensions separate. We assume layouts of the form (B, C, H, W[, D]).
        if x.dim() >= 4:
            grid_shape = list(x.shape[2:])
            if grid_shape:
                return True, grid_shape

        if x.dim() == 3:
            # Handle either channel-first 1D data (B, C, L) or channel-free (B, H, W).
            if x.shape[1] in {1, 2, 3, 4} and x.shape[2] > 1:
                return True, [x.shape[2]]
            grid_shape = list(x.shape[1:])
            if grid_shape:
                return True, grid_shape

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
        if method == 'fft' and (not is_grid or not grid_shape):
            self.logger.warning("Data is not grid-structured, falling back to RFF")
            # Use RFF instead of recursively calling this function
            method = 'rff'
        
        cacheable = method != 'nystrom'
        cache_key = None
        operator: Optional[KernelOperator] = None
        if cacheable:
            cache_key = self._make_kernel_cache_key(
                method,
                batch_size,
                data_dim,
                epsilon,
                grid_shape if method == 'fft' else None,
                original_shape=tuple(x.shape),
            )
            operator = self.kernel_operators.get(cache_key)
            if operator is not None:
                self.perf_stats["kernel_cache_hits"] += 1
                self.kernel_operators.move_to_end(cache_key)
                self.logger.debug(f"Using cached {method} operator")

        if operator is None:
            self.perf_stats["kernel_cache_misses"] += 1
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
                
            if cacheable and cache_key is not None:
                self.kernel_operators[cache_key] = operator
                self.kernel_operators.move_to_end(cache_key)

                while len(self.kernel_operators) > self.max_kernel_cache_size:
                    evicted_key, evicted_operator = self.kernel_operators.popitem(last=False)
                    self.perf_stats["kernel_cache_evictions"] += 1
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

            # Compute g from f with improved numerical stability
            Kf = kernel_op.apply(x, f)

            # Use clamping to prevent numerical instability
            if (Kf < 1e-10).any() or (Kf > 1e8).any():
                self.logger.warning(
                    "Light Schrodinger bridge: Kf out of range [%.2e, %.2e]",
                    float(Kf.min()),
                    float(Kf.max()),
                )
            g = torch.ones_like(f) / torch.clamp(Kf, min=1e-8, max=1e8)
        else:
            # Traditional iterative approach
            # Initialize potentials
            f = torch.ones(batch_size, device=self.device)
            g = torch.ones_like(f)

            # Iterative updates
            for i in range(max_iter):
                f_prev = f.clone()

                # Update f: f = 1 / (K @ g) with numerical stability
                Kg = kernel_op.apply(x, g)
                if (Kg < 1e-10).any() or (Kg > 1e8).any():
                    self.logger.warning(
                        "Schrodinger bridge iteration %d: Kg out of range [%.2e, %.2e]",
                        i,
                        float(Kg.min()),
                        float(Kg.max()),
                    )
                f = 1.0 / torch.clamp(Kg, min=1e-8, max=1e8)

                # Update g: g = 1 / (K^T @ f) with numerical stability
                KTf = kernel_op.apply_transpose(x, f)
                if (KTf < 1e-10).any() or (KTf > 1e8).any():
                    self.logger.warning(
                        "Schrodinger bridge iteration %d: KTf out of range [%.2e, %.2e]",
                        i,
                        float(KTf.min()),
                        float(KTf.max()),
                    )
                g = 1.0 / torch.clamp(KTf, min=1e-8, max=1e8)
                
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
        tol: Optional[float] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Conjugate gradient method for solving Ax = b.

        Args:
            A_func: Function that computes A@x
            b: Right-hand side vector
            x0: Initial guess (optional)
            max_iter: Maximum iterations
            tol: Optional override for relative tolerance

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

        iterations = 0
        self.perf_stats["cg_solve_count"] += 1

        rel_tol = tol if tol is not None else self.cg_relative_tolerance
        abs_tol = max(self.cg_absolute_tolerance, 0.0)
        dtype_info = torch.finfo(r.dtype)

        def _finalize(converged: bool, residual: torch.Tensor) -> Tuple[torch.Tensor, bool]:
            residual_value = float(residual.detach().cpu().item())
            self.perf_stats["cg_iterations_total"] += iterations
            self.perf_stats["cg_last_residual"] = residual_value
            if not converged:
                self.perf_stats["cg_failure_count"] += 1
            return x, converged

        initial_residual = torch.sqrt(rsold)
        baseline = max(float(initial_residual.detach().cpu().item()), dtype_info.tiny)
        threshold = max(abs_tol, rel_tol * baseline)

        # Early exit if already below tolerance
        if float(initial_residual.detach().cpu().item()) <= threshold:
            return _finalize(True, initial_residual)

        for i in range(max_iter):
            iterations += 1
            Ap = A_func(p)

            # Compute denominator safely with dtype-aware epsilon
            denom = torch.sum(p * Ap)
            denom_is_finite = torch.isfinite(denom)
            denom_abs = torch.abs(denom)
            dtype_info = torch.finfo(denom.dtype)
            min_denom = torch.sqrt(torch.tensor(dtype_info.eps, dtype=denom.dtype))

            if (not denom_is_finite) or denom_abs < min_denom:
                denom_value = float(denom.detach().cpu().item()) if denom_is_finite else float("nan")
                self.logger.warning(
                    "Conjugate gradient restart at iteration %d due to unstable curvature (denominator %.3e).",
                    i,
                    denom_value,
                )
                self.perf_stats["cg_restart_count"] += 1
                current_residual = torch.sqrt(rsold)
                if float(current_residual.detach().cpu().item()) <= max(threshold, 10 * dtype_info.tiny):
                    return _finalize(True, current_residual)

                # Otherwise restart with steepest descent direction
                p = r.clone()
                Ap = A_func(p)
                denom = torch.sum(p * Ap)
                denom_is_finite = torch.isfinite(denom)
                denom_abs = torch.abs(denom)

                # If still problematic, return current best solution
                if (not denom_is_finite) or denom_abs <= min_denom:
                    self.logger.warning(
                        "Conjugate gradient failed to recover after restart at iteration %d; returning partial solution.",
                        i,
                    )
                    return _finalize(False, current_residual)

            alpha = rsold / denom
            if not torch.isfinite(alpha):
                self.logger.warning(
                    "Conjugate gradient produced non-finite step at iteration %d; returning partial solution.",
                    i,
                )
                return _finalize(False, torch.sqrt(rsold))
            x = x + alpha * p

            # Periodically compute full residual to avoid drift
            if i % 10 == 0:
                r = b - A_func(x)
            else:
                r = r - alpha * Ap

            rsnew = torch.sum(r * r)
            rsnorm = torch.sqrt(rsnew)
            rsnorm_value = float(rsnorm.detach().cpu().item())

            # Check convergence
            if rsnorm_value <= threshold:
                return _finalize(True, rsnorm)

            # If the residual is increasing or unstable, exit early
            if rsnew > rsold * 1.02:
                prev_residual = torch.sqrt(rsold).detach().cpu().item()
                self.logger.warning(
                    "Conjugate gradient residual increased at iteration %d (%.3e -> %.3e); returning partial solution.",
                    i,
                    prev_residual,
                    rsnorm_value,
                )
                # Return current solution but indicate not fully converged
                return _finalize(False, rsnorm)

            # Compute beta with numerical stability check
            rsold_safe = torch.clamp(rsold, min=torch.finfo(rsold.dtype).tiny)
            beta = rsnew / rsold_safe
            p = r + beta * p
            rsold = rsnew

        final_residual = torch.sqrt(rsold).detach().cpu().item()
        self.logger.warning(
            "Conjugate gradient reached max iterations (%d) without convergence; residual norm %.3e.",
            max_iter,
            final_residual,
        )
        return _finalize(False, torch.sqrt(rsold))

    def _make_kernel_cache_key(
        self,
        method: str,
        batch_size: int,
        data_dim: int,
        epsilon: float,
        grid_shape: Optional[Iterable[int]] = None,
        original_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[Any, ...]:
        """Construct a robust cache key for kernel operators."""

        device_key = (self.device.type, getattr(self.device, "index", None))
        normalized_epsilon = round(float(epsilon), 12)

        key_components: List[Any] = [
            method,
            self.kernel_type,
            normalized_epsilon,
            batch_size,
            data_dim,
            device_key,
            original_shape,  # Include original shape to prevent collisions
        ]

        if grid_shape is not None:
            key_components.append(tuple(int(g) for g in grid_shape))

        if method == 'rff':
            key_components.extend(
                [
                    int(self.rff_features),
                    bool(self.use_multiscale),
                    tuple(float(s) for s in self.scale_factors) if self.use_multiscale else None,
                    self.seed,
                ]
            )
        elif method == 'nystrom':
            key_components.extend([int(self.n_landmarks), self.seed])
        elif method == 'fft':
            key_components.append(bool(self.use_multiscale))

        return tuple(key_components)

    def clear_kernel_cache(self) -> None:
        """Clear all cached kernel operators and their internal state."""

        for operator in self.kernel_operators.values():
            if hasattr(operator, "clear_cache"):
                try:
                    operator.clear_cache()
                except Exception:
                    continue
        self.kernel_operators.clear()
    
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
            
            pairwise_kernel = getattr(kernel_op, "pairwise", None)
            pairwise_weights: Optional[torch.Tensor] = None
            if callable(pairwise_kernel):
                try:
                    pairwise_weights = pairwise_kernel(z_flat, x_curr_flat)
                except NotImplementedError:
                    pairwise_weights = None
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.debug("Pairwise kernel evaluation failed: %s", exc)
                    pairwise_weights = None

            if pairwise_weights is not None:
                P_zx = pairwise_weights.to(fg.dtype) * fg.unsqueeze(0)
                row_sums = torch.clamp(P_zx.sum(dim=1, keepdim=True), min=1e-10)
                P_zx_norm = P_zx / row_sums
                z_next = P_zx_norm @ x_next_flat
            elif isinstance(kernel_op, FFTKernelOperator):
                alpha = 0.5  # Interpolation parameter
                z_next = (1 - alpha) * z_flat + alpha * x_next_flat
            else:
                raise RuntimeError(
                    "Kernel operator must implement pairwise evaluation or support FFT transport "
                    "to construct a transport map."
                )
            
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
            scale_factor = math.sqrt(max(1e-3, 1.0 - alpha_value))
            eps = max(self.epsilon / 100.0, self.epsilon * scale_factor)
        
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

        timesteps = self.validate_timesteps(timesteps)

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

    def validate_timesteps(self, timesteps: Sequence[float]) -> List[float]:
        """Validate and normalise a sequence of timesteps for sampling."""

        if len(timesteps) < 2:
            raise ValueError("Timesteps must contain at least two values.")

        validated: List[float] = []
        for idx, raw in enumerate(timesteps):
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Timestep at index {idx} is not finite: {raw}")
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Timesteps must lie in the range [0, 1]. Invalid value: {value}"
                )
            validated.append(value)

        schedule = sorted(validated, reverse=True)
        spacing_tol = 1e-6

        # Ensure strictly decreasing order with minimum spacing
        for prev, curr in zip(schedule, schedule[1:]):
            min_delta = spacing_tol * max(1.0, abs(prev))
            if prev - curr <= min_delta:
                raise ValueError(
                    "Timesteps must be strictly decreasing with spacing larger than 1e-6."
                )

        return schedule
    
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
        if self.solver_type == 'nystrom' or self.solver_type == 'auto':
            self.n_landmarks = min(100, n_samples // 10)  # Starting point
            
            while self.n_landmarks < n_samples:
                error_bound = 1.0 / math.sqrt(max(self.n_landmarks, 1))
                
                if error_bound <= error_tolerance:
                    break
                
                self.n_landmarks = min(self.n_landmarks * 2, n_samples)
                
            results['n_landmarks'] = self.n_landmarks
            results['nystrom_error_bound'] = 1.0 / math.sqrt(max(self.n_landmarks, 1))
            
        return results


#############################################
# Advanced Hierarchical Diffusion Sampler   #


# Backwards compatibility alias with previous misspelling
SchrodingerBridgeSolver = SchroedingerBridgeSolver
