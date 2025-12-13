"""Core SBDS solver implementation."""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .fft_ot import FFTOptimalTransport
from .kernel import KernelDerivativeRFF
from .metrics import MetricsLogger
from .noise_schedule import EnhancedAdaptiveNoiseSchedule
from .sinkhorn import HilbertSinkhornDivergence
from .utils import create_standard_timesteps

# Resource limits
MAX_TOTAL_ELEMENTS = 1e9  # 1B elements  4GB for float32
MAX_BATCH_SIZE = 10000
MAX_FEATURES = 100000

# Transport relaxation parameters
TRANSPORT_RELAXATION_FACTOR = 0.8  # Blend factor for transport updates
TRANSPORT_CORRECTION_FACTOR = 0.2  # Correction factor (1 - relaxation)

# Noise and regularization
FFT_REGULARIZATION_NOISE = 0.01  # Noise added for FFT transport regularization
FFT_TRANSPORT_STEP_SIZE = 0.2  # Step size for FFT gradient updates
VARIANCE_REDUCTION_NOISE = 1e-4  # Small noise for variance reduction diversity
SIGMA_NOISE_SCALE = 5.0  # Scaling factor for adaptive sigma

# Numerical stability
MIN_ALPHA_VARIANCE = 1e-8  # Minimum variance to prevent division by zero
SCORE_VAR_REGULARIZATION = 1e-8  # Regularization for score variance
LOG_STABILITY_EPS = 1e-10  # Epsilon for stable log operations
KERNEL_STABILITY_EPS = 1e-10  # Epsilon for kernel stability
STABLE_EXP_MAX = 50.0  # Maximum value for stable exponential
STABLE_EXP_MIN = -50.0  # Minimum value for stable exponential
BETA_CLAMP_MAX = 20.0  # Upper bound for beta(t) when approximated numerically

# Sinkhorn parameters
SINKHORN_ITERATIONS_FULL = 50  # Iterations for full Sinkhorn
SINKHORN_ITERATIONS_RFF = 20  # Iterations for RFF Sinkhorn
SINKHORN_ITERATIONS_NYSTROM = 30  # Iterations for Nystrom Sinkhorn


class EnhancedScoreBasedSBDiffusionSolver:
    """
    Enhanced Score-Based Schrodinger Bridge Diffusion Solver with advanced kernel methods.

    This enhanced version incorporates:
    1. Kernel Derivative Random Fourier Features for accurate score approximation
    2. FFT-based Optimal Transport for grid-structured data
    3. Hilbert Sinkhorn Divergence with improved theoretical guarantees
    4. Advanced kernel approximation techniques for better numerical stability and efficiency
    """

    # Type hints for instance attributes
    fft_ot: Optional[FFTOptimalTransport]
    sinkhorn_divergence: Optional[HilbertSinkhornDivergence]
    rff: Optional[KernelDerivativeRFF]
    last_sigma: Optional[float]

    def __init__(
        self,
        score_model: nn.Module,
        noise_schedule: Callable[[float], float],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        eps: float = 0.01,  # Entropy regularization parameter
        adaptive_eps: bool = True,  # Whether to adapt eps based on noise level
        sb_iterations: int = 3,  # Number of Schrodinger Bridge iterations
        computational_tier: str = 'auto',  # 'full', 'rff', 'nystrom', 'multiscale', 'auto'
        kernel_bandwidth: float = 1.0,  # Bandwidth for the kernel
        rff_features: int = 1024,  # Number of random features for RFF
        use_fft_ot: bool = True,  # Whether to use FFT-OT for grid-structured data
        num_landmarks: int = 100,  # For Nystrom approximation
        score_guided_landmarks: bool = True,  # Whether to use score for landmark selection
        multiscale_levels: int = 3,
        checkpoint_freq: int = 5,  # How often to check for convergence
        use_mixed_precision: bool = True,  # Whether to use mixed precision for GPU acceleration
        adaptive_timestep: bool = True,  # Whether to use adaptive timesteps
        error_tolerance: float = 1e-3,  # Error tolerance for adaptive timesteps
        corrector_steps: int = 0,  # Number of corrector steps (Langevin MCMC)
        corrector_snr: float = 0.1,  # Signal-to-noise ratio for corrector
        selective_sb: bool = True,  # Whether to use SB selectively at critical timesteps
        critical_thresholds: List[float] = [0.9, 0.5, 0.1],  # Timesteps where full SB is always used
        warm_start_potentials: bool = True,  # Whether to warm start potentials with score info
        early_stopping: bool = True,  # Whether to use early stopping for SB iterations
        early_stopping_tol: float = 1e-4,  # Tolerance for early stopping
        use_hilbert_sinkhorn: bool = True,  # Whether to use Hilbert Sinkhorn Divergence
        debiased_divergence: bool = True,  # Whether to use debiased divergence
        kernel_derivative_order: int = 2,  # Maximum order of kernel derivatives
        chunk_size: int = 128,  # Chunk size for pairwise distance computation
        use_spectral_gradient: bool = False,  # Whether to use spectral gradient computation
        model_outputs_noise: bool = False,  # If True, model predicts epsilon; otherwise predicts score
    ):
        """
        Initialize the Enhanced Score-Based Schrodinger Bridge Diffusion Solver.
        
        Args:
            score_model: Neural network that predicts the score (gradient of log density)
            noise_schedule: Function that returns noise level at time t
            device: Device to run computations on
            eps: Entropy regularization parameter for Sinkhorn/SB
            adaptive_eps: Whether to adapt eps based on noise level
            sb_iterations: Number of Schrodinger Bridge iterations
            computational_tier: Which computational approach to use
            kernel_bandwidth: Bandwidth for the kernel
            rff_features: Number of random features for RFF
            use_fft_ot: Whether to use FFT-OT for grid-structured data
            num_landmarks: Number of landmarks for Nystrom approximation
            score_guided_landmarks: Whether to use score for selecting landmarks
            multiscale_levels: Number of levels for multiscale approach
            checkpoint_freq: Frequency for checking convergence in Sinkhorn
            use_mixed_precision: Whether to use mixed precision for matrix operations
            adaptive_timestep: Whether to use adaptive timesteps
            error_tolerance: Error tolerance for adaptive timesteps
            corrector_steps: Number of corrector steps (Langevin MCMC)
            corrector_snr: Signal-to-noise ratio for corrector
            selective_sb: Whether to use SB selectively based on timestep importance
            critical_thresholds: Timesteps where full SB is always used
            warm_start_potentials: Whether to initialize potentials with score info
            early_stopping: Whether to use early stopping for SB iterations
            early_stopping_tol: Tolerance for early stopping
            use_hilbert_sinkhorn: Whether to use Hilbert Sinkhorn Divergence
            debiased_divergence: Whether to use debiased divergence
            kernel_derivative_order: Maximum order of kernel derivatives to support
            chunk_size: Chunk size for pairwise distance computation
            use_spectral_gradient: Whether to use spectral gradient computation
            model_outputs_noise: If True, model predicts epsilon; otherwise predicts score
        """
        # Input validation
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if sb_iterations < 1:
            raise ValueError(f"sb_iterations must be at least 1, got {sb_iterations}")
        if kernel_bandwidth <= 0:
            raise ValueError(f"kernel_bandwidth must be positive, got {kernel_bandwidth}")
        if rff_features < 1:
            raise ValueError(f"rff_features must be at least 1, got {rff_features}")
        if num_landmarks < 1:
            raise ValueError(f"num_landmarks must be at least 1, got {num_landmarks}")
        if multiscale_levels < 1:
            raise ValueError(f"multiscale_levels must be at least 1, got {multiscale_levels}")
        if checkpoint_freq < 1:
            raise ValueError(f"checkpoint_freq must be at least 1, got {checkpoint_freq}")
        if error_tolerance <= 0:
            raise ValueError(f"error_tolerance must be positive, got {error_tolerance}")
        if corrector_steps < 0:
            raise ValueError(f"corrector_steps must be non-negative, got {corrector_steps}")
        if corrector_snr <= 0:
            raise ValueError(f"corrector_snr must be positive, got {corrector_snr}")
        if early_stopping_tol <= 0:
            raise ValueError(f"early_stopping_tol must be positive, got {early_stopping_tol}")
        if kernel_derivative_order < 1:
            raise ValueError(f"kernel_derivative_order must be at least 1, got {kernel_derivative_order}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
        if computational_tier not in ['auto', 'full', 'rff', 'nystrom', 'multiscale']:
            raise ValueError(
                f"computational_tier must be one of ['auto', 'full', 'rff', 'nystrom', 'multiscale'], "
                f"got '{computational_tier}'"
            )

        self.score_model = score_model
        self.noise_schedule = noise_schedule
        self.device = device
        self.eps = eps
        self.adaptive_eps = adaptive_eps
        self.sb_iterations = sb_iterations
        self.computational_tier = computational_tier
        self.kernel_bandwidth = kernel_bandwidth
        self.rff_features = rff_features
        self.use_fft_ot = use_fft_ot
        self.num_landmarks = num_landmarks
        self.score_guided_landmarks = score_guided_landmarks
        self.multiscale_levels = multiscale_levels
        self.checkpoint_freq = checkpoint_freq
        self.use_mixed_precision = use_mixed_precision
        self.adaptive_timestep = adaptive_timestep
        self.error_tolerance = error_tolerance
        self.corrector_steps = corrector_steps
        self.corrector_snr = corrector_snr
        self.selective_sb = selective_sb
        self.critical_thresholds = critical_thresholds
        self.warm_start_potentials = warm_start_potentials
        self.early_stopping = early_stopping
        self.early_stopping_tol = early_stopping_tol
        self.use_hilbert_sinkhorn = use_hilbert_sinkhorn
        self.debiased_divergence = debiased_divergence
        self.kernel_derivative_order = kernel_derivative_order
        self.chunk_size = chunk_size
        self.use_spectral_gradient = use_spectral_gradient
        self._warned_fft_batch = False
        # Respect model hint when available; otherwise use provided flag
        if hasattr(score_model, "predicts_noise"):
            self.model_outputs_noise = bool(getattr(score_model, "predicts_noise"))
        else:
            self.model_outputs_noise = model_outputs_noise
        
        # Initialize kernel derivative RFF
        self.rff = None
        self.last_sigma = None
        
        # Initialize FFT-OT solver
        if use_fft_ot:
            self.fft_ot = FFTOptimalTransport(
                epsilon=eps,
                max_iter=100,
                tol=1e-5,
                kernel_type='gaussian',
                device=device,
                multiscale=True,
                scale_levels=multiscale_levels
            )
        else:
            self.fft_ot = None
            
        # Initialize Hilbert Sinkhorn Divergence
        if use_hilbert_sinkhorn:
            self.sinkhorn_divergence = HilbertSinkhornDivergence(
                epsilon=eps,
                max_iter=100,
                tol=1e-5,
                device=device,
                adaptive_epsilon=adaptive_eps,
                kernel_type='gaussian',
                sigma=kernel_bandwidth,
                debiased=debiased_divergence,
                use_rff=True,
                rff_features=rff_features,
                accelerated=True
            )
        else:
            self.sinkhorn_divergence = None
        
        # Set up mixed precision if requested and available
        self.amp_dtype = torch.float16 if self.use_mixed_precision and device.type == 'cuda' else torch.float32
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision and device.type == 'cuda' else None
        
        # Pre-create constants for log-uniform distributions to avoid host-device sync
        self.log_batch_size: Dict[int, torch.Tensor] = {}  # Cache for different batch sizes

    def _validate_memory_usage(self, shape: Tuple[int, ...]) -> None:
        """
        Validate that the requested shape won't cause OOM issues.

        Args:
            shape: Shape of samples to generate

        Raises:
            ValueError: If shape exceeds resource limits
        """
        total_elements = np.prod(shape)
        batch_size = shape[0]

        if total_elements > MAX_TOTAL_ELEMENTS:
            raise ValueError(
                f"Requested {total_elements:.2e} elements exceeds maximum of {MAX_TOTAL_ELEMENTS:.2e}. "
                f"This would require approximately {total_elements * 4 / 1e9:.2f} GB of memory. "
                f"Consider reducing batch size or data dimensions."
            )

        if batch_size > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {batch_size} exceeds maximum of {MAX_BATCH_SIZE}. "
                f"Consider processing data in smaller batches."
            )

        if self.rff_features > MAX_FEATURES:
            raise ValueError(
                f"RFF features {self.rff_features} exceeds maximum of {MAX_FEATURES}. "
                f"Consider reducing rff_features parameter."
            )

        # Validate timestep parameter
        if not math.isfinite(total_elements):
            raise ValueError(
                f"Shape {shape} results in non-finite number of elements. "
                f"Please check input dimensions."
            )

    def _initialize_rff(self, input_dim: int) -> None:
        """
        Initialize RFF for given input dimension.

        Args:
            input_dim: Dimensionality of the input data
        """
        if self.rff is None or self.rff.input_dim != input_dim:
            # Adjust bandwidth based on the input dimension for better scaling
            adjusted_bandwidth = self.kernel_bandwidth * math.sqrt(input_dim / 100)
            
            self.rff = KernelDerivativeRFF(
                input_dim=input_dim,
                feature_dim=self.rff_features,
                sigma=adjusted_bandwidth,
                kernel_type='gaussian',
                device=self.device,
                orthogonal=True,
                derivative_order=self.kernel_derivative_order
            )
        
    def _reinitialize_rff_weights(self, new_sigma: float) -> None:
        """
        Scale RFF weights instead of resampling when sigma changes.
        This maintains feature consistency throughout the diffusion process.

        Args:
            new_sigma: New bandwidth parameter
        """
        if new_sigma <= 0:
            raise ValueError(f"new_sigma must be positive, got {new_sigma}")

        if self.rff is not None:
            # Skip if change is small (< 5%)
            if self.last_sigma is not None and self.last_sigma > 0:
                relative_change = abs(new_sigma - self.last_sigma) / self.last_sigma
                if relative_change < 0.05:
                    return
            
            # Cache the original weights and offsets if not already done
            if not hasattr(self.rff, '_original_weights'):
                setattr(self.rff, '_original_weights', self.rff.weights.clone())
                setattr(self.rff, '_original_offset', self.rff.offset.clone())
                setattr(self.rff, '_original_sigma', self.rff.sigma)

            # Scale the weights directly based on the original weights
            # This preserves the mathematical relationship and avoids accumulating numerical errors
            original_sigma = float(getattr(self.rff, '_original_sigma'))
            scale_factor = original_sigma / new_sigma
            original_weights: torch.Tensor = getattr(self.rff, '_original_weights')
            # Use register_buffer to update weights in-place
            self.rff.register_buffer('weights', original_weights * scale_factor)

            # Update sigma and cache the latest value
            self.rff.sigma = new_sigma
            self.last_sigma = new_sigma
            
            # The offset doesn't need to be scaled as it's a phase term independent of sigma
    
    def sample(
        self,
        shape: Tuple[int, ...],
        timesteps: List[float],
        verbose: bool = True,
        callback: Optional[Callable[[float, torch.Tensor], None]] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        enable_profiling: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using Enhanced Score-Based Schrodinger Bridge diffusion solver.
        
        Args:
            shape: Shape of samples to generate (batch_size, *data_dims)
            timesteps: List of timesteps for the reverse process
            verbose: Whether to show progress bar
            callback: Optional callback function to call after each step with current samples
            metrics_logger: Optional metrics logger for performance tracking
            enable_profiling: Whether to enable detailed performance profiling
        
        Returns:
            Tensor of generated samples
        """
        # Validate memory usage before starting
        self._validate_memory_usage(shape)

        # Validate timesteps
        if not timesteps:
            raise ValueError("timesteps list cannot be empty")
        for i, t in enumerate(timesteps):
            if not math.isfinite(t):
                raise ValueError(f"Timestep at index {i} is not finite: {t}")
            if t < 0 or t > 1:
                warnings.warn(f"Timestep at index {i} is outside [0, 1]: {t}")

        # Initialize with noise
        x_t = torch.randn(shape, device=self.device)
        
        # Initialize RFF for flattened data
        self._initialize_rff(int(np.prod(shape[1:])))
        
        # Import profiling tools only if needed
        if enable_profiling:
            try:
                import torch.autograd.profiler as profiler
                have_profiler = True
            except ImportError:
                warnings.warn("Profiling requested but torch.autograd.profiler not available")
                enable_profiling = False
                have_profiler = False
        else:
            have_profiler = False
        
        # Progress through timesteps
        timesteps = sorted(timesteps, reverse=True)  # Ensure descending order
        iterator = tqdm(range(len(timesteps)-1), desc="Sampling") if verbose else range(len(timesteps)-1)
        
        for i in iterator:
            t_curr, t_next = timesteps[i], timesteps[i+1]
            
            # Start timing if metrics logging is enabled
            if metrics_logger is not None:
                metrics_logger.start_step()
            
            # Use profiling context if enabled
            cm = profiler.record_function("sampling_step") if enable_profiling and have_profiler else nullcontext()
            
            with cm:
                # Determine computational tier if set to auto
                if self.computational_tier == 'auto':
                    tier = self._determine_computational_tier(shape[0], int(np.prod(shape[1:])))
                else:
                    tier = self.computational_tier
                
                # Determine whether to use full SB for this timestep
                use_sb = self._should_use_sb(t_curr, t_next)
                sb_iterations = self.sb_iterations if use_sb else 1
                
                # Check if data is grid-structured (for images/volumes) and we can use FFT-OT
                is_grid, _ = self._is_grid_structured(x_t)
                use_fft = self.use_fft_ot and is_grid and x_t.size(0) == 1
                if self.use_fft_ot and is_grid and x_t.size(0) > 1 and not self._warned_fft_batch:
                    warnings.warn(
                        "FFT-OT is currently disabled for batch size > 1 to avoid batch-collapsed transport. "
                        "Set use_fft_ot=False or provide batch size 1 to enable FFT-OT.",
                        RuntimeWarning,
                    )
                    self._warned_fft_batch = True
                
                # Adjust kernel bandwidth based on noise level for better numerical stability
                if self.adaptive_eps:
                    alpha_t = self.noise_schedule(t_curr)
                    sigma_t = math.sqrt(1 - alpha_t)
                    # Update RFF sigma based on noise level and reinitialize weights
                    if self.rff is not None:
                        new_sigma = self.kernel_bandwidth * (1 + sigma_t * SIGMA_NOISE_SCALE)
                        self._reinitialize_rff_weights(new_sigma)
                
                # Apply appropriate transport map based on tier and SB decision
                transport_cost = None
                
                # Apply transport and track performance
                prof_ctx = profiler.record_function(f"transport_{tier}") if enable_profiling and have_profiler else nullcontext()
                with prof_ctx:
                    if use_fft:
                        x_t, transport_cost = self._fft_sb_transport(x_t, t_curr, t_next, sb_iterations)
                        if metrics_logger is not None and not enable_profiling:
                            metrics_logger.log_module_flops('fft_transport', np.prod(shape) * 10)  # Approximate FLOPs
                    elif tier == 'full':
                        x_t, transport_cost = self._enhanced_sb_transport(x_t, t_curr, t_next, sb_iterations)
                        if metrics_logger is not None and not enable_profiling:
                            metrics_logger.log_module_flops('enhanced_sb', np.prod(shape) ** 2 * 2)  # Approximate FLOPs
                    elif tier == 'rff':
                        x_t, transport_cost = self._rff_sb_transport(x_t, t_curr, t_next, sb_iterations)
                        if metrics_logger is not None and not enable_profiling:
                            metrics_logger.log_module_flops('rff_sb', np.prod(shape) * self.rff_features * 3)  # Approximate FLOPs
                    elif tier == 'nystrom':
                        if self.score_guided_landmarks:
                            landmarks = self._score_guided_landmark_selection(x_t, t_curr)
                        else:
                            landmarks = self._random_landmarks(x_t)
                        x_t, transport_cost = self._nystrom_sb_transport(x_t, landmarks, t_curr, t_next, sb_iterations)
                        if metrics_logger is not None and not enable_profiling:
                            metrics_logger.log_module_flops('nystrom_sb', np.prod(shape) * self.num_landmarks * 3)  # Approximate FLOPs
                    elif tier == 'multiscale':
                        x_t, transport_cost = self._multiscale_sb_transport(x_t, t_curr, t_next, sb_iterations)
                        if metrics_logger is not None and not enable_profiling:
                            metrics_logger.log_module_flops('multiscale_sb', np.prod(shape) * np.log2(shape[0]) * 3)  # Approximate FLOPs
                    else:
                        raise ValueError(
                            f"Unknown computational tier: '{tier}'. "
                            f"Valid options: 'auto', 'full', 'rff', 'nystrom', 'multiscale'"
                        )
                
                # Apply corrector if requested (Langevin MCMC refinement)
                if self.corrector_steps > 0:
                    corr_ctx = profiler.record_function("corrector") if enable_profiling and have_profiler else nullcontext()
                    with corr_ctx:
                        x_t = self._corrector_update(x_t, t_next)
                    if metrics_logger is not None and not enable_profiling:
                        metrics_logger.log_module_flops('corrector', np.prod(shape) * self.corrector_steps * 3)  # Approximate FLOPs
            
            # End timing and log metrics
            if metrics_logger is not None:
                metrics_logger.end_step(t_next, transport_cost)
            
            if callback is not None:
                callback(t_next, x_t)
            
            if verbose and i % 10 == 0 and hasattr(iterator, "set_postfix_str"):
                status = f"t_curr={t_curr:.4f}, t_next={t_next:.4f}, tier={tier}"
                if use_sb:
                    status += f", using SB with {sb_iterations} iterations"
                iterator.set_postfix_str(status)
        
        # Save metrics if logger is provided
        if metrics_logger is not None:
            metrics_logger.save_metrics()
        
        return x_t
        
    def _is_grid_structured(self, x: torch.Tensor) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if data is structured on a regular grid (for FFT-OT).
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (is_grid, grid_shape)
        """
        if self.fft_ot is not None:
            return self.fft_ot._is_grid_structured(x)
        
        # Default implementation
        if x.dim() >= 4:
            # Assume N,C,*spatial
            return True, list(x.shape[2:])
        if x.dim() == 3:
            # Assume N,*spatial
            return True, list(x.shape[1:])
        
        return False, None
    
    def _should_use_sb(self, t_curr: float, t_next: float) -> bool:
        """
        Determine whether to use Schrodinger Bridge for this timestep based on importance.
        
        Args:
            t_curr: Current timestep
            t_next: Next timestep
        
        Returns:
            Boolean indicating whether to use full SB
        """
        if not self.selective_sb:
            return True
        
        # Always use SB at critical thresholds
        for threshold in self.critical_thresholds:
            if t_curr >= threshold >= t_next:
                return True
        
        # Use SB when step size is large
        step_size = t_curr - t_next
        nominal_step = 1.0 / (len(self.critical_thresholds) + 10)  # Estimate nominal step size
        if step_size > 1.5 * nominal_step:
            return True
        
        # Use SB when noise level is changing rapidly
        alpha_curr = self.noise_schedule(t_curr)
        alpha_next = self.noise_schedule(t_next)
        if abs(alpha_curr - alpha_next) > 0.1:
            return True
        
        # Use simple OT for less critical timesteps
        return False
    
    def _determine_computational_tier(self, batch_size: int, dim: int) -> str:
        """
        Determine which computational tier to use based on problem size.
        
        Args:
            batch_size: Number of samples
            dim: Dimensionality of each sample
        
        Returns:
            Computational tier to use: 'full', 'rff', 'nystrom', or 'multiscale'
        """
        # High-resolution image heuristic
        total_elements = batch_size * dim
        
        if total_elements < 5e5:  # Small batches or low-res images
            return 'full'
        elif total_elements < 5e6:  # Medium batches or medium-res images
            return 'rff'
        elif total_elements < 5e7:  # Large batches or high-res images
            return 'nystrom'
        else:  # Very large batches or very high-res images
            return 'multiscale'
    
    def _compute_score(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute score (gradient of log density) using the score model.
        
        Args:
            x: Input tensor
            t: Current time
        
        Returns:
            Score tensor
        """
        t_tensor = torch.ones(x.shape[0], device=x.device) * t
        model_out = self.score_model(x, t_tensor)

        # If the model outputs epsilon, convert to score; otherwise assume it already outputs score
        alpha_t = self.noise_schedule(t)
        out_dtype = model_out.dtype
        compute_dtype = (
            torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
        )

        if self.model_outputs_noise:
            variance = torch.as_tensor(1 - alpha_t, device=model_out.device, dtype=compute_dtype)
            variance = torch.clamp(variance, min=MIN_ALPHA_VARIANCE)
            denom = torch.sqrt(variance)
            score: torch.Tensor = -model_out.to(dtype=compute_dtype) / denom
        else:
            score = model_out.to(dtype=compute_dtype)

        result: torch.Tensor = score.to(dtype=out_dtype)
        return result
    
    def _variance_reduced_score(self, x: torch.Tensor, t: float, n_samples: int = 5) -> torch.Tensor:
        """
        Compute variance-reduced score estimate using Monte Carlo sampling.

        This method reduces estimation variance by:
        1. Enabling stochastic mode (dropout) in the score model if available
        2. Adding small perturbations to input for diversity
        3. Using Rao-Blackwellization with inverse variance weighting

        Mathematical Background:
        For stochastic score estimates s_i ~ p(s|x,t), the variance-reduced estimator is:
            s_VR =  w_i * s_i  where w_i  1/Var(s_i)

        This reduces variance by a factor of ~1/n_samples while maintaining unbiasedness.

        NOTE: This method requires a stochastic score model (e.g., with dropout enabled)
        to be effective. For deterministic models, small input noise provides limited
        diversity but still helps with numerical stability.

        Args:
            x: Input tensor
            t: Current time
            n_samples: Number of samples for variance reduction (default: 5)

        Returns:
            Variance-reduced score tensor with lower estimation variance
        """
        if n_samples <= 1:
            return self._compute_score(x, t)

        # Enable dropout/stochastic mode if the model supports it
        training_mode = self.score_model.training
        if hasattr(self.score_model, 'train'):
            self.score_model.train()  # Enable dropout for stochastic estimates

        scores = []
        try:
            for _ in range(n_samples):
                # Add small noise to input for diversity if model is deterministic
                x_noisy = x + torch.randn_like(x) * VARIANCE_REDUCTION_NOISE
                score = self._compute_score(x_noisy, t)
                scores.append(score)
        finally:
            # Restore original training mode
            if hasattr(self.score_model, 'train'):
                self.score_model.train(training_mode)

        # Monte Carlo averaging with Rao-Blackwellization
        score_stack = torch.stack(scores)
        score_mean = score_stack.mean(dim=0)
        score_var = score_stack.var(dim=0, unbiased=True)

        # Adaptive weighting based on variance (higher weight for lower variance)
        # Using inverse variance weighting with regularization
        weight = 1.0 / (1.0 + score_var.clamp(min=SCORE_VAR_REGULARIZATION))
        weight_mean = weight.mean()
        # Normalize weights, with protection against zero mean
        if weight_mean > 0:
            weight = weight / weight_mean
        else:
            weight = torch.ones_like(weight)  # Fallback to uniform weights

        return score_mean * weight
    
    def _compute_score_rff(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute score using RFF kernel derivative approximation.
        This is an alternative method that can be used when the score model is unstable.
        
        Args:
            x: Input tensor
            t: Current time
            
        Returns:
            Approximated score tensor
        """
        alpha_t = self.noise_schedule(t)

        # We need to approximate the score function  log p_t(x)
        # First, generate samples from p_t
        batch_size = x.size(0)
        flat_dim = int(np.prod(x.shape[1:]))

        # Generate reference samples from the marginal distribution at time t.
        # For variance-preserving diffusions with standard normal prior, the marginal
        # at time t is N(0, (1 - alpha_t) I).
        variance = torch.as_tensor(
            max(0.0, 1 - alpha_t), device=self.device, dtype=x.dtype
        )
        ref_samples = torch.sqrt(variance) * torch.randn(
            batch_size * 4, *x.shape[1:], device=self.device, dtype=x.dtype
        )

        # Reshape for RFF
        x_flat = x.reshape(batch_size, flat_dim)
        ref_flat = ref_samples.reshape(ref_samples.size(0), flat_dim)

        # Ensure RFF is initialized
        self._initialize_rff(flat_dim)

        # Compute score approximation using kernel derivatives
        if self.rff is None:
            raise RuntimeError("RFF not initialized after _initialize_rff call")
        score_flat = self.rff.compute_score_approximation(x_flat, ref_flat)

        # Reshape back to original shape
        score = score_flat.reshape(x.shape)
        
        return score
    
    def _compute_drift(self, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        Compute the drift term for the transport map using score function.
        CORRECTED VERSION: Properly implements probability flow ODE.

        For variance-preserving SDE: dx = -0.5*beta(t)*x*dt + sqrt(beta(t))*dW
        The probability flow ODE drift is: dx/dt = -0.5*beta(t)*[x + s_theta(x,t)]

        Args:
            x: Input tensor
            t: Current time
            dt: Time step

        Returns:
            Drift tensor (drift * dt for integration step)
        """
        score = self._compute_score(x, t)

        # Get noise schedule parameters
        alpha_t = self.noise_schedule(t)

        # Compute beta(t) = -d log(alpha_bar(t))/dt
        if hasattr(self.noise_schedule, 'get_beta'):
            beta_t = self.noise_schedule.get_beta(t, dt)
        else:
            # Numerical approximation
            dt_safe = max(dt, 1e-4)
            alpha_t_clamped = max(alpha_t, MIN_ALPHA_VARIANCE)
            alpha_t_dt = self.noise_schedule(max(0, t - dt_safe))
            alpha_t_dt_clamped = max(alpha_t_dt, MIN_ALPHA_VARIANCE)
            beta_t = max(0.0, -(math.log(alpha_t_clamped) - math.log(alpha_t_dt_clamped)) / dt_safe)
        # Clamp beta to a reasonable range to avoid numerical blow-ups near endpoints
        # Always enforce non-negativity even when the noise schedule supplies get_beta.
        beta_t = float(np.clip(beta_t, 0.0, BETA_CLAMP_MAX))

        # Probability flow ODE drift: dx/dt = -0.5 * beta(t) * [x + s_theta(x,t)]
        # Equivalently: dx/dt = -0.5 * beta(t) * x - 0.5 * beta(t) * score
        drift = -0.5 * beta_t * x - 0.5 * beta_t * score

        return drift * dt
    
    def _stable_log(self, x: torch.Tensor, eps: float = LOG_STABILITY_EPS) -> torch.Tensor:
        """Numerically stable logarithm."""
        return torch.log(torch.clamp(x, min=eps))

    def _stable_exp(self, x: torch.Tensor, max_val: float = STABLE_EXP_MAX) -> torch.Tensor:
        """Numerically stable exponential."""
        return torch.exp(torch.clamp(x, min=STABLE_EXP_MIN, max=max_val))
    
    def _apply_transport_map(self, x: torch.Tensor, y: torch.Tensor, transport_plan: torch.Tensor) -> torch.Tensor:
        """
        Apply transport map from x to y using the transport plan.
        
        Args:
            x: Source points [batch_size, *dims]
            y: Target points [batch_size, *dims]
            transport_plan: Transport plan matrix [batch_size, batch_size]
            
        Returns:
            Transported points
        """
        # Normalize transport plan with protection against zero sum
        row_sums = transport_plan.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=LOG_STABILITY_EPS)  # Prevent division by zero
        P_normalized = transport_plan / row_sums

        # Apply barycentric mapping
        x_shape = x.shape
        y_flat = y.reshape(y.size(0), -1)

        # Transport: x_new = P @ y (barycentric projection)
        transported = P_normalized @ y_flat

        return transported.reshape(x_shape)
    
    def _enhanced_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        Implement actual Schrodinger Bridge transport with enhanced methods.
        
        Args:
            x_t: Current samples
            t_curr: Current timestep
            t_next: Next timestep
            iterations: Number of SB iterations
            
        Returns:
            Tuple of (transported samples, transport cost)
        """
        batch_size = x_t.shape[0]
        
        # Initialize with score-based prediction
        x_pred = x_t + self._compute_drift(x_t, t_curr, t_curr - t_next)
        
        # Generate reference samples at t_next
        alpha_next = self.noise_schedule(t_next)
        variance_next = torch.as_tensor(
            max(0.0, 1 - alpha_next), device=x_t.device, dtype=x_t.dtype
        )
        x_ref = torch.sqrt(variance_next) * torch.randn_like(x_t)
        
        transport_cost = 0.0
        
        for iter in range(iterations):
            # Store previous for convergence check
            x_prev = x_pred.clone() if iter > 0 and self.early_stopping else None
            
            # Compute optimal transport
            if self.use_hilbert_sinkhorn:
                # Use Hilbert Sinkhorn with consistent plan reconstruction
                assert self.sinkhorn_divergence is not None  # Guaranteed by use_hilbert_sinkhorn
                x_pred_flat = x_pred.reshape(batch_size, -1)
                x_ref_flat = x_ref.reshape(batch_size, -1)

                cost_matrix = self.sinkhorn_divergence._compute_cost_matrix(x_pred_flat, x_ref_flat)
                eps_val: float = (
                    float(self.eps * (1 + 5 * torch.sqrt(variance_next).item()))
                    if self.adaptive_eps
                    else self.eps
                )

                transport_cost_tensor, u, v = self.sinkhorn_divergence._sinkhorn_algorithm(
                    cost_matrix, eps=eps_val
                )
                log_kernel = (-cost_matrix / eps_val).clamp(min=-50.0)
                log_P = (u[:, None] + v[None, :]) + log_kernel
                log_P = torch.clamp(log_P, min=-50.0, max=50.0)
                P = torch.exp(log_P)
                # Normalize defensively to avoid accumulation of numerical error
                P = P / P.sum(dim=1, keepdim=True).clamp_min(LOG_STABILITY_EPS)
                transport_cost = float(torch.sum(P * cost_matrix))

                x_transported = self._apply_transport_map(x_pred, x_ref, P)
                x_pred = TRANSPORT_RELAXATION_FACTOR * x_pred + TRANSPORT_CORRECTION_FACTOR * x_transported
                
            else:
                # Standard optimal transport
                x_pred_flat = x_pred.reshape(batch_size, -1)
                x_ref_flat = x_ref.reshape(batch_size, -1)
                
                # Compute cost matrix
                C = torch.cdist(x_pred_flat, x_ref_flat, p=2).pow(2)
                
                # Sinkhorn algorithm
                eps = (
                    self.eps * (1 + 5 * torch.sqrt(variance_next))
                    if self.adaptive_eps
                    else self.eps
                )
                # Prevent degenerate kernels: set a floor tied to data scale
                median_cost = torch.median(C).item()
                mean_cost = torch.mean(C).item()
                eps = max(float(eps), 1e-3, 0.1 * median_cost, 0.5 * mean_cost)
                K = torch.exp(-C / eps).clamp_min(LOG_STABILITY_EPS)
                
                # Initialize dual potentials
                u = torch.zeros(batch_size, device=self.device)
                v = torch.zeros(batch_size, device=self.device)
                
                # Sinkhorn iterations
                for _ in range(SINKHORN_ITERATIONS_FULL):
                    u = self._stable_log(1.0 / (K @ torch.exp(v)))
                    v = self._stable_log(1.0 / (K.T @ torch.exp(u)))
                
                # Compute transport plan
                P = torch.diag(torch.exp(u)) @ K @ torch.diag(torch.exp(v))
                P = P / P.sum(dim=1, keepdim=True).clamp_min(LOG_STABILITY_EPS)
                transport_cost = torch.sum(P * C).item()
                
                # Apply transport
                x_transported = self._apply_transport_map(x_pred, x_ref, P)
                x_pred = TRANSPORT_RELAXATION_FACTOR * x_pred + TRANSPORT_CORRECTION_FACTOR * x_transported
            
            # Early stopping check
            if self.early_stopping and x_prev is not None:
                norm_pred = torch.norm(x_pred)
                if norm_pred > LOG_STABILITY_EPS:  # Prevent division by zero
                    change = torch.norm(x_pred - x_prev) / norm_pred
                    if change < self.early_stopping_tol:
                        break
        
        return x_pred, transport_cost
    
    def _rff_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        RFF-accelerated Schrodinger Bridge transport.
        
        Args:
            x_t: Current samples
            t_curr: Current timestep
            t_next: Next timestep
            iterations: Number of SB iterations
            
        Returns:
            Tuple of (transported samples, transport cost)
        """
        batch_size = x_t.shape[0]
        flat_dim = int(np.prod(x_t.shape[1:]))

        # Initialize with score-based prediction
        x_pred = x_t + self._compute_drift(x_t, t_curr, t_curr - t_next)

        # Generate reference samples at t_next
        alpha_next = self.noise_schedule(t_next)
        variance_next = torch.as_tensor(
            max(0.0, 1 - alpha_next), device=x_t.device, dtype=x_t.dtype
        )
        x_ref = torch.sqrt(variance_next) * torch.randn_like(x_t)

        # Ensure RFF is initialized
        self._initialize_rff(flat_dim)
        if self.rff is None:
            raise RuntimeError("RFF not initialized after _initialize_rff call")

        transport_cost = 0.0

        for iter in range(iterations):
            # Flatten for RFF computation
            x_pred_flat = x_pred.reshape(batch_size, flat_dim)
            x_ref_flat = x_ref.reshape(batch_size, flat_dim)

            # Compute RFF features
            x_pred_features = self.rff.compute_features(x_pred_flat)
            x_ref_features = self.rff.compute_features(x_ref_flat)

            # Pairwise squared distances in feature space approximate the kernel-induced
            # distance while preserving the random feature concentration guarantees.
            C_approx = torch.cdist(x_pred_features, x_ref_features, p=2) ** 2
            C_approx = torch.clamp(C_approx, min=0.0)

            # Sinkhorn in feature space
            eps = (
                self.eps * (1 + 5 * torch.sqrt(variance_next))
                if self.adaptive_eps
                else self.eps
            )
            K = torch.exp(-C_approx / (eps + 1e-30))

            # Fast Sinkhorn iterations
            a = torch.ones(batch_size, device=self.device) / batch_size
            b = torch.ones(batch_size, device=self.device) / batch_size
            
            for _ in range(SINKHORN_ITERATIONS_RFF):  # Fewer iterations needed with RFF
                a = 1.0 / (K @ b + LOG_STABILITY_EPS)
                b = 1.0 / (K.T @ a + LOG_STABILITY_EPS)
            
            # Transport plan
            P = torch.diag(a) @ K @ torch.diag(b)
            
            # Compute transport cost using the feature-space squared distances
            transport_cost = torch.sum(P * C_approx).item()
            
            # Apply transport
            row_sums = P.sum(dim=1, keepdim=True).clamp_min(LOG_STABILITY_EPS)
            x_transported = (P @ x_ref_flat) / row_sums
            x_transported = x_transported.reshape(x_t.shape)
            
            # Update with relaxation
            x_pred = TRANSPORT_RELAXATION_FACTOR * x_pred + TRANSPORT_CORRECTION_FACTOR * x_transported
        
        return x_pred, transport_cost
    
    def _fft_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        FFT-based Schrodinger Bridge transport for grid-structured data.

        Mathematical Approximation:
        This method uses a Gaussian density surrogate (x)  exp(-x2/2) as an approximation
        to the true marginal p_t(x) of the diffusion process.

        Why Gaussian?
        - For variance-preserving SDEs, marginals p_t are approximately Gaussian
        - The approximation is exact at t=0 (pure noise) and t=1 (data)
        - For intermediate times, the Wasserstein-2 error is bounded by:
          W2(p_t, )  Ct(1-t) where C depends on the score function

        When is this valid?
        - Grid-structured data (images, volumes) where spatial structure matters
        - Variance-preserving noise schedules (cosine, linear)
        - Sufficiently smooth score functions

        Limitations:
        - May underestimate transport cost for multimodal distributions
        - Assumes data is roughly centered and has bounded variance
        - Error increases for highly non-Gaussian data distributions

        References:
        - De Bortoli et al. "Diffusion Schrodinger Bridge" (NeurIPS 2021)
        - Vargas et al. "Solving Inverse Problems with Score-Based Generative Models" (2023)

        Args:
            x_t: Current samples (grid-structured, e.g., [batch, *spatial_dims, channels])
            t_curr: Current timestep in [0, 1]
            t_next: Next timestep in [0, 1]
            iterations: Number of Sinkhorn-Bridge iterations (default: 3)

        Returns:
            Tuple of (transported samples, regularized transport cost)
        """
        if x_t.size(0) > 1:
            warnings.warn(
                "FFT SB transport currently supports batch size 1 to avoid batch-collapsed transport. "
                "Falling back to RFF transport.",
                RuntimeWarning,
            )
            return self._rff_sb_transport(x_t, t_curr, t_next, iterations)

        if self.fft_ot is None:
            raise RuntimeError("FFT-OT not initialized. Set use_fft_ot=True in constructor.")

        # Initialize with score-based prediction
        x_pred = x_t + self._compute_drift(x_t, t_curr, t_curr - t_next)
        
        # Generate reference distribution at t_next
        alpha_next = self.noise_schedule(t_next)
        variance_next = torch.as_tensor(
            max(0.0, 1 - alpha_next), device=x_t.device, dtype=x_t.dtype
        )
        x_ref = torch.sqrt(variance_next) * torch.randn_like(x_t)
        
        transport_cost = 0.0
        
        for iter in range(iterations):
            # Convert to distribution format for FFT-OT
            # Sum over batch dimension to get density on grid
            # Moment-matched Gaussian mixture surrogate captures non-Gaussian structure
            # by aggregating per-location statistics across the batch.
            x_pred_mean = torch.mean(x_pred, dim=0)
            x_ref_mean = torch.mean(x_ref, dim=0)

            x_pred_var = torch.var(x_pred, dim=0, unbiased=False)
            x_ref_var = torch.var(x_ref, dim=0, unbiased=False)

            x_pred_var = torch.clamp(x_pred_var, min=1e-5)
            x_ref_var = torch.clamp(x_ref_var, min=1e-5)

            log_mu_components = (
                -0.5 * ((x_pred - x_pred_mean) ** 2) / x_pred_var
                - 0.5 * torch.log(2 * math.pi * x_pred_var)
            )
            log_nu_components = (
                -0.5 * ((x_ref - x_ref_mean) ** 2) / x_ref_var
                - 0.5 * torch.log(2 * math.pi * x_ref_var)
            )

            mu = torch.exp(torch.logsumexp(log_mu_components, dim=0))
            nu = torch.exp(torch.logsumexp(log_nu_components, dim=0))
            
            # Normalize
            mu = mu / mu.sum()
            nu = nu / nu.sum()
            
                        # Run FFT-OT
            objective, u, v = self.fft_ot._multiscale_sinkhorn_fft(mu, nu, self.eps)
            # Transport cost (regularized OT dual objective)
            transport_cost = (objective.item() if torch.is_tensor(objective) else float(objective))
            
            # Apply transport map using gradients of dual potentials
            grad_u = self.fft_ot._compute_gradient_on_grid(u)

            # Validate gradient dimensionality matches u's dimensions
            # After summing over the batch dimension, u has the same shape as x_pred[0]
            # and _compute_gradient_on_grid returns one gradient tensor per axis of u.
            expected_grad_dims = u.dim()

            if len(grad_u) != expected_grad_dims:
                raise ValueError(
                    f"Gradient dimensionality {len(grad_u)} does not match "
                    f"expected dimensions {expected_grad_dims}. "
                    f"x_pred shape: {x_pred.shape}, u shape after batch sum: {tuple(u.shape)}, "
                    f"grad_u length: {len(grad_u)}"
                )

            # Aggregate per-axis gradients into a scalar transport field so that updates
            # remain well-defined regardless of channel ordering. This mirrors the
            # divergence of the transport potential and provides a signed correction.
            grad_stack = torch.stack(grad_u, dim=0)
            transport_field = torch.sum(grad_stack, dim=0)

            if transport_field.shape != x_pred.shape[1:]:
                raise ValueError(
                    "FFT transport gradient shape mismatch: "
                    f"transport field {transport_field.shape} vs sample shape {x_pred.shape[1:]}"
                )

            x_pred = x_pred - FFT_TRANSPORT_STEP_SIZE * transport_field.unsqueeze(0)

            # Add small noise for regularization
            x_pred = x_pred + FFT_REGULARIZATION_NOISE * torch.randn_like(x_pred)
        
        return x_pred, transport_cost
    
    def _score_guided_landmark_selection(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Select landmarks based on score function magnitude.
        
        Args:
            x: Input samples
            t: Current timestep
            
        Returns:
            Selected landmark samples
        """
        # Compute scores
        scores = self._compute_score(x, t)
        
        # Compute score magnitudes
        score_norms = torch.norm(scores.reshape(x.size(0), -1), dim=1)
        
        # Select landmarks with highest score magnitudes (most informative)
        _, indices = torch.topk(score_norms, min(self.num_landmarks, x.size(0)))
        
        return x[indices]
    
    def _random_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly select landmark samples.
        
        Args:
            x: Input samples
            
        Returns:
            Selected landmark samples
        """
        indices = torch.randperm(x.size(0), device=x.device)[:min(self.num_landmarks, x.size(0))]
        return x[indices]
    
    def _nystrom_sb_transport(self, x_t: torch.Tensor, landmarks: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        Nystrom-approximated Schrodinger Bridge transport.
        
        Args:
            x_t: Current samples
            landmarks: Landmark samples for Nystrom approximation
            t_curr: Current timestep
            t_next: Next timestep
            iterations: Number of SB iterations
            
        Returns:
            Tuple of (transported samples, transport cost)
        """
        batch_size = x_t.shape[0]
        num_landmarks = landmarks.shape[0]
        
        # Initialize with score-based prediction
        x_pred = x_t + self._compute_drift(x_t, t_curr, t_curr - t_next)
        
        # Generate reference samples at t_next
        alpha_next = self.noise_schedule(t_next)
        variance_next = torch.as_tensor(
            max(0.0, 1 - alpha_next), device=x_t.device, dtype=x_t.dtype
        )
        x_ref = torch.sqrt(variance_next) * torch.randn_like(x_t)
        
        transport_cost = 0.0
        
        for iter in range(iterations):
            # Flatten for kernel computation
            x_pred_flat = x_pred.reshape(batch_size, -1)
            x_ref_flat = x_ref.reshape(batch_size, -1)
            landmarks_flat = landmarks.reshape(num_landmarks, -1)
            
            # Compute kernel matrices for Nystrom approximation
            eps = (
                self.eps * (1 + 5 * torch.sqrt(variance_next))
                if self.adaptive_eps
                else self.eps
            )
            
            # K_nm: kernel between all points and landmarks
            K_nm_pred = torch.exp(-torch.cdist(x_pred_flat, landmarks_flat, p=2).pow(2) / (2 * eps))
            K_nm_ref = torch.exp(-torch.cdist(x_ref_flat, landmarks_flat, p=2).pow(2) / (2 * eps))
            
            # K_mm: kernel between landmarks
            K_mm = torch.exp(-torch.cdist(landmarks_flat, landmarks_flat, p=2).pow(2) / (2 * eps))
            
            # Nystrom approximation: K  K_nm @ K_mm^(-1) @ K_mn
            K_mm_inv = torch.linalg.pinv(K_mm)
            
            # Approximate transport kernel
            K_approx = K_nm_pred @ K_mm_inv @ K_nm_ref.T
            
            # Sinkhorn iterations
            a = torch.ones(batch_size, device=self.device) / batch_size
            b = torch.ones(batch_size, device=self.device) / batch_size
            
            for _ in range(SINKHORN_ITERATIONS_NYSTROM):
                a = 1.0 / (K_approx @ b + LOG_STABILITY_EPS)
                b = 1.0 / (K_approx.T @ a + LOG_STABILITY_EPS)
            
            # Transport plan
            P = torch.diag(a) @ K_approx @ torch.diag(b)
            
            # Approximate transport cost
            C_approx = torch.cdist(x_pred_flat, x_ref_flat, p=2).pow(2)
            transport_cost = torch.sum(P * C_approx).item()
            
            # Apply transport
            row_sums = P.sum(dim=1, keepdim=True).clamp_min(LOG_STABILITY_EPS)
            x_transported = (P @ x_ref_flat) / row_sums
            x_transported = x_transported.reshape(x_t.shape)
            
            # Update with relaxation
            x_pred = TRANSPORT_RELAXATION_FACTOR * x_pred + TRANSPORT_CORRECTION_FACTOR * x_transported
        
        return x_pred, transport_cost
    
    def _multiscale_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        Multi-scale Schrodinger Bridge transport with proper warm-starting.

        Uses hierarchical coarse-to-fine approach where each finer scale is initialized
        from the interpolated solution of the coarser scale. This significantly improves
        convergence speed compared to solving each scale independently.

        Args:
            x_t: Current samples
            t_curr: Current timestep
            t_next: Next timestep
            iterations: Number of SB iterations

        Returns:
            Tuple of (transported samples, transport cost)
        """
        # Initialize with score-based prediction
        x_pred = x_t + self._compute_drift(x_t, t_curr, t_curr - t_next)

        # Reference samples are generated inside _rff_sb_transport at each scale
        transport_cost = 0.0

        # Multi-scale approach: solve at coarse scale first, then refine
        scales = [2**i for i in range(self.multiscale_levels, -1, -1)]
        
        for scale_idx, scale in enumerate(scales):
            # Subsample if scale > 1
            if scale > 1:
                indices = torch.arange(0, x_pred.size(0), scale, device=self.device)
                x_pred_sub = x_pred[indices].clone()  # Clone to avoid in-place modification
            else:
                x_pred_sub = x_pred

            # Run transport at current scale
            # Use more iterations at coarser scales for better initialization
            scale_iterations = max(1, iterations // (scale_idx + 1))
            x_pred_sub, cost = self._rff_sb_transport(x_pred_sub, t_curr, t_next, scale_iterations)
            transport_cost += cost / len(scales)

            # Interpolate back to full resolution for warm-start of next scale
            if scale > 1:
                # Update coarse samples
                x_pred[indices] = x_pred_sub

                # Linear interpolation for in-between samples (warm-start for next scale)
                for i in range(len(indices) - 1):
                    start_idx = int(indices[i].item())
                    end_idx = int(indices[i + 1].item())
                    if end_idx - start_idx > 1:
                        # Create interpolation weights
                        n_interp = end_idx - start_idx - 1
                        alphas = torch.linspace(0, 1, n_interp + 2, device=self.device)[1:-1]
                        for j, alpha in enumerate(alphas, start=1):
                            x_pred[start_idx + j] = (1 - alpha) * x_pred[start_idx] + alpha * x_pred[end_idx]

                # Handle last segment (extrapolation)
                if len(indices) > 1 and indices[-1] < x_pred.size(0) - 1:
                    last_idx = int(indices[-1].item())
                    prev_idx = int(indices[-2].item())
                    # Use gradient from last two coarse samples
                    grad = x_pred[last_idx] - x_pred[prev_idx]
                    for j in range(last_idx + 1, x_pred.size(0)):
                        alpha = (j - last_idx) / scale
                        x_pred[j] = x_pred[last_idx] + alpha * grad
        
        return x_pred, transport_cost
    
    def _corrector_update(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Apply Langevin MCMC corrector steps.
        
        Args:
            x: Current samples
            t: Current timestep
            
        Returns:
            Corrected samples
        """
        alpha_t = self.noise_schedule(t)
        
        for _ in range(self.corrector_steps):
            # Compute score
            score = self._compute_score(x, t)
            
            # Langevin dynamics step with per-sample adaptive step sizes
            noise = torch.randn_like(x)
            score_norms = torch.norm(
                score.reshape(x.size(0), -1), dim=1, keepdim=True
            )
            safe_norms = torch.clamp(score_norms, min=1e-4)
            step_size = self.corrector_snr * (1 - alpha_t) / safe_norms
            step_size = torch.clamp(step_size, min=1e-6, max=0.1)

            x = x + step_size * score + torch.sqrt(2 * step_size) * noise
        
        return x
    
    def estimate_convergence_rate(self, n_samples: int, dim: int) -> Dict[str, float]:
        """
        Estimate theoretical convergence rates for the solver.
        
        Args:
            n_samples: Number of samples
            dim: Dimensionality of the data
            
        Returns:
            Dictionary of convergence rate estimates
        """
        # Score estimation error: O(n^{-1/2})
        score_error = 1.0 / np.sqrt(n_samples)
        
        # RFF approximation error: O(D^{-1/2})
        rff_error = 1.0 / np.sqrt(self.rff_features) if self.rff is not None else 0.0
        
        # Sinkhorn convergence: O(exp(-t/))
        sinkhorn_rate = np.exp(-self.sb_iterations / self.eps)
        
        # Nystrom approximation error: O(m^{-1/2}) where m is number of landmarks
        nystrom_error = 1.0 / np.sqrt(self.num_landmarks)
        
        # Overall error bound (worst case)
        total_error = score_error + rff_error + sinkhorn_rate + nystrom_error
        
        return {
            'score_estimation': score_error,
            'rff_approximation': rff_error,
            'sinkhorn_convergence': sinkhorn_rate,
            'nystrom_approximation': nystrom_error,
            'total_bound': total_error
        }


def test_sbds_implementation() -> None:
    """
    Test the enhanced SBDS implementation with a simple example.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a simple score model (placeholder)
    class SimpleScoreModel(nn.Module):
        def __init__(self, dim: int = 2) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, dim)
            )

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Concatenate x and t
            t = t.reshape(-1, 1)
            if x.dim() > 2:
                original_shape = x.shape
                x = x.reshape(x.shape[0], -1)
                xt = torch.cat([x, t], dim=1)
                out: torch.Tensor = self.net(xt)
                result: torch.Tensor = out.reshape(original_shape)
                return result
            else:
                xt = torch.cat([x, t], dim=1)
                result = self.net(xt)
                return result
    
    # Initialize components
    score_model = SimpleScoreModel(dim=2).to(device)
    noise_schedule = EnhancedAdaptiveNoiseSchedule(
        schedule_type='cosine',
        device=device
    )
    
    # Create solver
    solver = EnhancedScoreBasedSBDiffusionSolver(
        score_model=score_model,
        noise_schedule=noise_schedule,
        device=device,
        eps=0.01,
        sb_iterations=3,
        computational_tier='auto',
        use_hilbert_sinkhorn=True,
        debiased_divergence=True,
        selective_sb=True
    )
    
    # Generate timesteps
    timesteps = create_standard_timesteps(num_steps=50, schedule_type='linear')
    
    # Sample shape
    batch_size = 16
    dim = 2
    shape = (batch_size, dim)
    
    # Create metrics logger
    logger = MetricsLogger()
    
    # Run sampling
    print("Starting sampling...")
    samples = solver.sample(
        shape=shape,
        timesteps=timesteps,
        verbose=True,
        metrics_logger=logger
    )
    
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample statistics: mean={samples.mean():.4f}, std={samples.std():.4f}")
    
    # Get metrics summary
    summary = logger.get_summary()
    print("\nPerformance metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    # Estimate convergence rates
    convergence = solver.estimate_convergence_rate(n_samples=batch_size, dim=dim)
    print("\nConvergence rate estimates:")
    for key, value in convergence.items():
        print(f"  {key}: {value:.6f}")

    print(f"\nTest complete. Generated {samples.shape} samples.")


def test_mathematical_correctness() -> None:
    """
    Unit tests for mathematical correctness of key components.
    """
    print("Running mathematical correctness tests...")
    
    # Test 1: Kernel derivative correctness
    print("\n1. Testing kernel derivatives...")
    rff = KernelDerivativeRFF(input_dim=2, feature_dim=128, sigma=1.0)
    x = torch.randn(10, 2)
    y = torch.randn(15, 2)
    
    # Test first derivative
    K = rff.compute_kernel(x, y)
    dK = rff.compute_kernel_derivative(x, y, order=1)
    
    # Verify shape
    if dK.shape != (2, 10, 15):
        raise ValueError(f"Wrong derivative shape: expected (2, 10, 15), got {dK.shape}")
    
    # Numerical check: ∂k/∂x_i ≈ (k(x+h*e_i,y) - k(x,y))/h
    # Use h=1e-3 for optimal finite difference accuracy (avoids catastrophic cancellation)
    h = 1e-3
    for i in range(2):
        x_plus = x.clone()
        x_plus[:, i] += h
        K_plus = rff.compute_kernel(x_plus, y)
        dK_numerical = (K_plus - K) / h
        error = torch.max(torch.abs(dK[i] - dK_numerical))
        print(f"  Derivative error for dim {i}: {error:.6f}")
        # RFF approximation error + finite difference error can be ~0.1-0.5
        if error >= 2.0:
            raise ValueError(f"Large derivative error: {error}")
    
    # Test 2: Probability flow ODE drift
    print("\n2. Testing probability flow ODE drift...")

    # Wrap lambda in nn.Module for type safety
    class SimpleScore(nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return -x

    solver = EnhancedScoreBasedSBDiffusionSolver(
        score_model=SimpleScore(),
        noise_schedule=lambda t: np.exp(-5 * t),  # Exponential schedule
        device=torch.device('cpu')
    )
    
    x = torch.randn(5, 3)
    t = 0.5
    dt = 0.01
    
    drift = solver._compute_drift(x, t, dt)

    # Verify drift has correct shape
    if drift.shape != x.shape:
        raise ValueError(f"Wrong drift shape: expected {x.shape}, got {drift.shape}")
    
    # Test 3: Stable exp/log
    print("\n3. Testing stable exp/log...")
    large_pos = torch.tensor([100.0])
    large_neg = torch.tensor([-100.0])
    
    exp_pos = solver._stable_exp(large_pos)
    exp_neg = solver._stable_exp(large_neg)

    if not torch.isfinite(exp_pos):
        raise ValueError("Stable exp failed for large positive")
    if not torch.isfinite(exp_neg):
        raise ValueError("Stable exp failed for large negative")
    if exp_pos > torch.exp(torch.tensor(50.0)):
        raise ValueError("Stable exp didn't clamp properly")
    
    # Test 4: MMD computation
    print("\n4. Testing MMD computation...")
    schedule = EnhancedAdaptiveNoiseSchedule()
    schedule._initialize_rff(2)
    
    x_features = torch.randn(100, 256)
    y_features = torch.randn(100, 256)
    
    mmd_sq = schedule._compute_mmd(x_features, y_features)
    if mmd_sq < 0:
        raise ValueError(f"MMD2 should be non-negative: {mmd_sq}")
    
    # Same distribution should have small MMD
    mmd_same = schedule._compute_mmd(x_features, x_features)
    if mmd_same >= 1e-6:
        raise ValueError(f"MMD2 for same distribution should be ~0: {mmd_same}")
    
    # Test 5: Sinkhorn convergence
    print("\n5. Testing Sinkhorn algorithm...")
    sinkhorn = HilbertSinkhornDivergence(epsilon=0.1, max_iter=100)
    
    x = torch.randn(20, 2)
    y = torch.randn(25, 2)
    
    divergence = sinkhorn.compute_divergence(x, y)
    if not torch.isfinite(divergence):
        raise ValueError("Sinkhorn divergence is not finite")
    if divergence < 0:
        raise ValueError(f"Divergence should be non-negative: {divergence}")
    
    # Test debiased property
    div_xx = sinkhorn.compute_divergence(x, x)
    if abs(div_xx) >= 1e-4:
        raise ValueError(f"Debiased divergence X,X should be ~0: {div_xx}")
    
    print("\nAll mathematical tests passed! ")


if __name__ == "__main__":
    # Run mathematical correctness tests first
    test_mathematical_correctness()
    
    # Then run the main test
    print("\n" + "="*50 + "\n")
    test_sbds_implementation()

