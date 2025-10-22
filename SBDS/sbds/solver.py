"""Main SBDS solver implementation."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common
from .kernels import KernelDerivativeRFF
from .metrics import MetricsLogger
from .schedule import EnhancedAdaptiveNoiseSchedule
from .transport import FFTOptimalTransport, HilbertSinkhornDivergence

warnings = common.warnings
tqdm = common.tqdm
nullcontext = common.nullcontext

__all__ = ["EnhancedScoreBasedSBDiffusionSolver"]


class EnhancedScoreBasedSBDiffusionSolver:
    """
    Enhanced Score-Based Schrödinger Bridge Diffusion Solver with advanced kernel methods.
    
    This enhanced version incorporates:
    1. Kernel Derivative Random Fourier Features for accurate score approximation
    2. FFT-based Optimal Transport for grid-structured data
    3. Hilbert Sinkhorn Divergence with improved theoretical guarantees
    4. Advanced kernel approximation techniques for better numerical stability and efficiency
    """
    def __init__(
        self,
        score_model: nn.Module, 
        noise_schedule: Callable,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        eps: float = 0.01,  # Entropy regularization parameter
        adaptive_eps: bool = True,  # Whether to adapt eps based on noise level
        sb_iterations: int = 3,  # Number of Schrödinger Bridge iterations
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
        spectral_gradient: bool = False,  # Whether to use spectral gradient computation
    ):
        """
        Initialize the Enhanced Score-Based Schrödinger Bridge Diffusion Solver.
        
        Args:
            score_model: Neural network that predicts the score (gradient of log density)
            noise_schedule: Function that returns noise level at time t
            device: Device to run computations on
            eps: Entropy regularization parameter for Sinkhorn/SB
            adaptive_eps: Whether to adapt eps based on noise level
            sb_iterations: Number of Schrödinger Bridge iterations
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
            spectral_gradient: Whether to use spectral gradient computation
        """
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
        self.spectral_gradient = spectral_gradient
        
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
        self.log_batch_size = {}  # Cache for different batch sizes
    
    def _initialize_rff(self, input_dim: int):
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
        if self.rff is not None:
            # Skip if change is small (< 5%)
            if self.last_sigma is not None:
                relative_change = abs(new_sigma - self.last_sigma) / self.last_sigma
                if relative_change < 0.05:
                    return
            
            # Cache the original weights and offsets if not already done
            if not hasattr(self.rff, 'original_weights'):
                self.rff.original_weights = self.rff.weights.clone()
                self.rff.original_offset = self.rff.offset.clone()
                self.rff.original_sigma = self.rff.sigma
            
            # Scale the weights directly based on the original weights
            # This preserves the mathematical relationship and avoids accumulating numerical errors
            scale_factor = self.rff.original_sigma / new_sigma
            self.rff.weights = self.rff.original_weights * scale_factor
            
            # Update sigma and cache the latest value
            self.rff.sigma = new_sigma
            self.last_sigma = new_sigma
            
            # The offset doesn't need to be scaled as it's a phase term independent of sigma
    
    def sample(
        self, 
        shape: Tuple[int, ...], 
        timesteps: List[float],
        verbose: bool = True,
        callback: Optional[Callable] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        enable_profiling: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using Enhanced Score-Based Schrödinger Bridge diffusion solver.
        
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
        # Initialize with noise
        x_t = torch.randn(shape, device=self.device)
        
        # Initialize RFF for flattened data
        self._initialize_rff(np.prod(shape[1:]))
        
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
                    tier = self._determine_computational_tier(shape[0], np.prod(shape[1:]))
                else:
                    tier = self.computational_tier
                
                # Determine whether to use full SB for this timestep
                use_sb = self._should_use_sb(t_curr, t_next)
                sb_iterations = self.sb_iterations if use_sb else 1
                
                # Check if data is grid-structured (for images/volumes) and we can use FFT-OT
                is_grid, _ = self._is_grid_structured(x_t)
                use_fft = self.use_fft_ot and is_grid
                
                # Adjust kernel bandwidth based on noise level for better numerical stability
                if self.adaptive_eps:
                    alpha_t = self.noise_schedule(t_curr)
                    sigma_t = math.sqrt(1 - alpha_t)
                    # Update RFF sigma based on noise level and reinitialize weights
                    if self.rff is not None:
                        new_sigma = self.kernel_bandwidth * (1 + sigma_t * 5)
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
                        raise ValueError(f"Unknown computational tier: {tier}")
                
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
            
            if verbose and i % 10 == 0:
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
        if x.dim() > 2:
            # Multi-dimensional data (like images) is grid-structured
            return True, list(x.shape[1:])
        
        return False, None
    
    def _should_use_sb(self, t_curr: float, t_next: float) -> bool:
        """
        Determine whether to use Schrödinger Bridge for this timestep based on importance.
        
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
        with torch.no_grad():
            t_tensor = torch.ones(x.shape[0], device=x.device) * t
            noise_pred = self.score_model(x, t_tensor)
        
        # Convert noise prediction to score
        alpha_t = self.noise_schedule(t)
        score = -noise_pred / torch.sqrt(1 - alpha_t)
        
        return score
    
    def _variance_reduced_score(self, x: torch.Tensor, t: float, n_samples: int = 5) -> torch.Tensor:
        """
        Compute variance-reduced score estimate.
        
        Args:
            x: Input tensor
            t: Current time
            n_samples: Number of samples for variance reduction
            
        Returns:
            Variance-reduced score tensor
        """
        if n_samples <= 1:
            return self._compute_score(x, t)
            
        scores = []
        for _ in range(n_samples):
            score = self._compute_score(x, t)
            scores.append(score)
        
        # Use control variates or other variance reduction
        score_mean = torch.stack(scores).mean(dim=0)
        score_var = torch.stack(scores).var(dim=0)
        
        # Adaptive weighting based on variance
        weight = 1.0 / (1.0 + score_var)
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
        
        # We need to approximate the score function ∇ log p_t(x)
        # First, generate samples from p_t
        batch_size = x.size(0)
        flat_dim = np.prod(x.shape[1:])
        
        # Generate reference samples from current noise level
        # We use more samples for better approximation
        ref_samples = torch.randn(batch_size * 4, *x.shape[1:], device=self.device)
        ref_samples = torch.sqrt(alpha_t) * ref_samples + torch.sqrt(1 - alpha_t) * torch.randn_like(ref_samples)
        
        # Reshape for RFF
        x_flat = x.reshape(batch_size, flat_dim)
        ref_flat = ref_samples.reshape(ref_samples.size(0), flat_dim)
        
        # Ensure RFF is initialized
        self._initialize_rff(flat_dim)
        
        # Compute score approximation using kernel derivatives
        score_flat = self.rff.compute_score_approximation(x_flat, ref_flat)
        
        # Reshape back to original shape
        score = score_flat.reshape(x.shape)
        
        return score
    
    def _compute_drift(self, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        Compute the drift term for the transport map using score function.
        CORRECTED VERSION: Properly implements probability flow ODE.
        
        For variance-preserving SDE: dx = -0.5*β(t)*x*dt + sqrt(β(t))*dW
        The probability flow ODE is: dx/dt = -β(t)/2 * x - β(t)*σ(t)^2 * s_θ(x,t)
        where σ(t)^2 = 1 - α_bar(t)
        
        Args:
            x: Input tensor
            t: Current time
            dt: Time step
        
        Returns:
            Drift tensor
        """
        score = self._compute_score(x, t)
        
        # Get noise schedule parameters
        alpha_t = self.noise_schedule(t)
        
        # Compute beta(t) = -d log(alpha_bar(t))/dt
        if hasattr(self.noise_schedule, 'get_beta'):
            beta_t = self.noise_schedule.get_beta(t, dt)
        else:
            # Numerical approximation
            alpha_t_dt = self.noise_schedule(max(0, t - dt))
            if alpha_t > 0 and alpha_t_dt > 0:
                beta_t = -(np.log(alpha_t) - np.log(alpha_t_dt)) / dt
            else:
                beta_t = 0.02  # Default value
        
        # Compute σ(t)^2 = 1 - α_bar(t)
        sigma2 = 1.0 - alpha_t
        
        # Probability flow ODE drift: dx/dt = -β(t)/2 * x - β(t)*σ(t)^2 * s_θ(x,t)
        drift = -0.5 * beta_t * x - beta_t * sigma2 * score
        
        return drift * dt
    
    def _stable_log(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Numerically stable logarithm."""
        return torch.log(torch.clamp(x, min=eps))
    
    def _stable_exp(self, x: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
        """Numerically stable exponential."""
        return torch.exp(torch.clamp(x, min=-max_val, max=max_val))
    
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
        # Normalize transport plan
        P_normalized = transport_plan / transport_plan.sum(dim=1, keepdim=True)
        
        # Apply barycentric mapping
        x_shape = x.shape
        x_flat = x.reshape(x.size(0), -1)
        y_flat = y.reshape(y.size(0), -1)
        
        # Transport: x_new = P @ y
        transported = P_normalized @ y_flat
        
        return transported.reshape(x_shape)
    
    def _enhanced_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        Implement actual Schrödinger Bridge transport with enhanced methods.
        
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
        x_ref = torch.randn_like(x_t)
        x_ref = torch.sqrt(alpha_next) * x_ref + torch.sqrt(1 - alpha_next) * torch.randn_like(x_ref)
        
        transport_cost = 0.0
        
        for iter in range(iterations):
            # Store previous for convergence check
            x_prev = x_pred.clone() if iter > 0 and self.early_stopping else None
            
            # Compute optimal transport
            if self.use_hilbert_sinkhorn:
                # Use Hilbert Sinkhorn Divergence
                divergence = self.sinkhorn_divergence.compute_divergence(x_pred, x_ref)
                transport_cost = divergence.item()
                
                # Extract transport plan (approximate using kernel)
                x_pred_flat = x_pred.reshape(batch_size, -1)
                x_ref_flat = x_ref.reshape(batch_size, -1)
                
                # Compute soft assignment using RFF kernel
                if self.sinkhorn_divergence.rff is not None:
                    K = self.sinkhorn_divergence.rff.compute_kernel(x_pred_flat, x_ref_flat)
                else:
                    K = torch.exp(-torch.cdist(x_pred_flat, x_ref_flat, p=2).pow(2) / (2 * self.kernel_bandwidth**2))
                
                # Apply transport with relaxation
                P = K / K.sum(dim=1, keepdim=True)
                x_transported = self._apply_transport_map(x_pred, x_ref, P)
                x_pred = 0.8 * x_pred + 0.2 * x_transported
                
            else:
                # Standard optimal transport
                x_pred_flat = x_pred.reshape(batch_size, -1)
                x_ref_flat = x_ref.reshape(batch_size, -1)
                
                # Compute cost matrix
                C = torch.cdist(x_pred_flat, x_ref_flat, p=2).pow(2)
                
                # Sinkhorn algorithm
                eps = self.eps * (1 + 5 * torch.sqrt(1 - alpha_next)) if self.adaptive_eps else self.eps
                K = torch.exp(-C / eps)
                
                # Initialize dual potentials
                u = torch.zeros(batch_size, device=self.device)
                v = torch.zeros(batch_size, device=self.device)
                
                # Sinkhorn iterations
                for _ in range(50):
                    u = self._stable_log(1.0 / (K @ torch.exp(v)))
                    v = self._stable_log(1.0 / (K.T @ torch.exp(u)))
                
                # Compute transport plan
                P = torch.diag(torch.exp(u)) @ K @ torch.diag(torch.exp(v))
                transport_cost = torch.sum(P * C).item()
                
                # Apply transport
                x_transported = self._apply_transport_map(x_pred, x_ref, P)
                x_pred = 0.8 * x_pred + 0.2 * x_transported
            
            # Early stopping check
            if self.early_stopping and x_prev is not None:
                change = torch.norm(x_pred - x_prev) / torch.norm(x_pred)
                if change < self.early_stopping_tol:
                    break
        
        return x_pred, transport_cost
    
    def _rff_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        RFF-accelerated Schrödinger Bridge transport.
        
        Args:
            x_t: Current samples
            t_curr: Current timestep
            t_next: Next timestep
            iterations: Number of SB iterations
            
        Returns:
            Tuple of (transported samples, transport cost)
        """
        batch_size = x_t.shape[0]
        flat_dim = np.prod(x_t.shape[1:])
        
        # Initialize with score-based prediction
        x_pred = x_t + self._compute_drift(x_t, t_curr, t_curr - t_next)
        
        # Generate reference samples at t_next
        alpha_next = self.noise_schedule(t_next)
        x_ref = torch.randn_like(x_t)
        x_ref = torch.sqrt(alpha_next) * x_ref + torch.sqrt(1 - alpha_next) * torch.randn_like(x_ref)
        
        # Ensure RFF is initialized
        self._initialize_rff(flat_dim)
        
        transport_cost = 0.0
        
        for iter in range(iterations):
            # Flatten for RFF computation
            x_pred_flat = x_pred.reshape(batch_size, flat_dim)
            x_ref_flat = x_ref.reshape(batch_size, flat_dim)
            
            # Compute RFF features
            x_pred_features = self.rff.compute_features(x_pred_flat)
            x_ref_features = self.rff.compute_features(x_ref_flat)
            
            # Normalize features to ensure ||Φ(x)||² = k(x,x) = 1
            x_pred_features = F.normalize(x_pred_features, p=2, dim=1)
            x_ref_features = F.normalize(x_ref_features, p=2, dim=1)
            
            # Approximate kernel matrix using RFF
            K_approx = x_pred_features @ x_ref_features.T
            
            # Sinkhorn in feature space
            eps = self.eps * (1 + 5 * torch.sqrt(1 - alpha_next)) if self.adaptive_eps else self.eps
            C_approx = 2 - 2 * K_approx
            K = torch.exp(-C_approx / (eps + 1e-30))
            
            # Fast Sinkhorn iterations
            a = torch.ones(batch_size, device=self.device) / batch_size
            b = torch.ones(batch_size, device=self.device) / batch_size
            
            for _ in range(20):  # Fewer iterations needed with RFF
                a = 1.0 / (K @ b + 1e-10)
                b = 1.0 / (K.T @ a + 1e-10)
            
            # Transport plan
            P = torch.diag(a) @ K @ torch.diag(b)
            
            # Compute transport cost
            # Approximate cost using normalized RFF distance
            C_approx = 2 - 2 * K_approx  # Since features are normalized
            transport_cost = torch.sum(P * C_approx).item()
            
            # Apply transport
            x_transported = P @ x_ref_flat
            x_transported = x_transported.reshape(x_t.shape)
            
            # Update with relaxation
            x_pred = 0.8 * x_pred + 0.2 * x_transported
        
        return x_pred, transport_cost
    
    def _fft_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        FFT-based Schrödinger Bridge transport for grid-structured data.
        
        NOTE: This uses a Gaussian density surrogate ∝ exp(-x²/2) as an approximation
        to the true marginal p_t. This is a heuristic that works well for image data.
        
        Args:
            x_t: Current samples (grid-structured)
            t_curr: Current timestep
            t_next: Next timestep
            iterations: Number of SB iterations
            
        Returns:
            Tuple of (transported samples, transport cost)
        """
        # Initialize with score-based prediction
        x_pred = x_t + self._compute_drift(x_t, t_curr, t_curr - t_next)
        
        # Generate reference distribution at t_next
        alpha_next = self.noise_schedule(t_next)
        x_ref = torch.randn_like(x_t)
        x_ref = torch.sqrt(alpha_next) * x_ref + torch.sqrt(1 - alpha_next) * torch.randn_like(x_ref)
        
        transport_cost = 0.0
        
        for iter in range(iterations):
            # Convert to distribution format for FFT-OT
            # Sum over batch dimension to get density on grid
            # NOTE: Using Gaussian density as surrogate
            mu = torch.sum(torch.exp(-0.5 * x_pred**2), dim=0)
            nu = torch.sum(torch.exp(-0.5 * x_ref**2), dim=0)
            
            # Normalize
            mu = mu / mu.sum()
            nu = nu / nu.sum()
            
                        # Run FFT-OT
            objective, u, v = self.fft_ot._multiscale_sinkhorn_fft(mu, nu, self.eps)
            # Transport cost (regularized OT dual objective)
            transport_cost = (objective.item() if torch.is_tensor(objective) else float(objective))
            
            # Apply transport map using gradients of dual potentials
            grad_u = self.fft_ot._compute_gradient_on_grid(u)
            
            # Update each sample using the gradient
            for i in range(x_pred.size(0)):
                for d, grad in enumerate(grad_u):
                    # Move samples according to gradient
                    x_pred[i, ..., d] = x_pred[i, ..., d] - 0.2 * grad
            
            # Add small noise for regularization
            x_pred = x_pred + 0.01 * torch.randn_like(x_pred)
        
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
        Nyström-approximated Schrödinger Bridge transport.
        
        Args:
            x_t: Current samples
            landmarks: Landmark samples for Nyström approximation
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
        x_ref = torch.randn_like(x_t)
        x_ref = torch.sqrt(alpha_next) * x_ref + torch.sqrt(1 - alpha_next) * torch.randn_like(x_ref)
        
        transport_cost = 0.0
        
        for iter in range(iterations):
            # Flatten for kernel computation
            x_pred_flat = x_pred.reshape(batch_size, -1)
            x_ref_flat = x_ref.reshape(batch_size, -1)
            landmarks_flat = landmarks.reshape(num_landmarks, -1)
            
            # Compute kernel matrices for Nyström approximation
            eps = self.eps * (1 + 5 * torch.sqrt(1 - alpha_next)) if self.adaptive_eps else self.eps
            
            # K_nm: kernel between all points and landmarks
            K_nm_pred = torch.exp(-torch.cdist(x_pred_flat, landmarks_flat, p=2).pow(2) / (2 * eps))
            K_nm_ref = torch.exp(-torch.cdist(x_ref_flat, landmarks_flat, p=2).pow(2) / (2 * eps))
            
            # K_mm: kernel between landmarks
            K_mm = torch.exp(-torch.cdist(landmarks_flat, landmarks_flat, p=2).pow(2) / (2 * eps))
            
            # Nyström approximation: K ≈ K_nm @ K_mm^(-1) @ K_mn
            K_mm_inv = torch.linalg.pinv(K_mm)
            
            # Approximate transport kernel
            K_approx = K_nm_pred @ K_mm_inv @ K_nm_ref.T
            
            # Sinkhorn iterations
            a = torch.ones(batch_size, device=self.device) / batch_size
            b = torch.ones(batch_size, device=self.device) / batch_size
            
            for _ in range(30):
                a = 1.0 / (K_approx @ b + 1e-10)
                b = 1.0 / (K_approx.T @ a + 1e-10)
            
            # Transport plan
            P = torch.diag(a) @ K_approx @ torch.diag(b)
            
            # Approximate transport cost
            C_approx = torch.cdist(x_pred_flat, x_ref_flat, p=2).pow(2)
            transport_cost = torch.sum(P * C_approx).item()
            
            # Apply transport
            x_transported = P @ x_ref_flat
            x_transported = x_transported.reshape(x_t.shape)
            
            # Update with relaxation
            x_pred = 0.8 * x_pred + 0.2 * x_transported
        
        return x_pred, transport_cost
    
    def _multiscale_sb_transport(self, x_t: torch.Tensor, t_curr: float, t_next: float, iterations: int = 3) -> Tuple[torch.Tensor, float]:
        """
        Multi-scale Schrödinger Bridge transport.
        
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
        
        # Generate reference samples at t_next
        alpha_next = self.noise_schedule(t_next)
        x_ref = torch.randn_like(x_t)
        x_ref = torch.sqrt(alpha_next) * x_ref + torch.sqrt(1 - alpha_next) * torch.randn_like(x_ref)
        
        transport_cost = 0.0
        
        # Multi-scale approach: solve at coarse scale first, then refine
        scales = [2**i for i in range(self.multiscale_levels, -1, -1)]
        
        for scale in scales:
            # Subsample if scale > 1
            if scale > 1:
                indices = torch.arange(0, x_pred.size(0), scale, device=self.device)
                x_pred_sub = x_pred[indices]
                x_ref_sub = x_ref[indices]
            else:
                x_pred_sub = x_pred
                x_ref_sub = x_ref
            
            # Run transport at current scale
            for iter in range(iterations // len(scales)):
                # Use RFF for efficiency at each scale
                x_pred_sub, cost = self._rff_sb_transport(x_pred_sub, t_curr, t_next, 1)
                transport_cost += cost / len(scales)
            
            # Interpolate back to full resolution if needed
            if scale > 1:
                # Simple nearest neighbor interpolation
                x_pred[indices] = x_pred_sub
                
                # Smooth interpolation for in-between samples
                for i in range(len(indices) - 1):
                    start_idx = indices[i]
                    end_idx = indices[i + 1]
                    for j in range(start_idx + 1, end_idx):
                        alpha = (j - start_idx) / (end_idx - start_idx)
                        x_pred[j] = (1 - alpha) * x_pred[start_idx] + alpha * x_pred[end_idx]
        
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
            
            # Langevin dynamics step
            noise = torch.randn_like(x)
            step_size = self.corrector_snr * (1 - alpha_t) / torch.norm(score.reshape(x.size(0), -1), dim=1, keepdim=True).mean()
            
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
        
        # Sinkhorn convergence: O(exp(-t/ε))
        sinkhorn_rate = np.exp(-self.sb_iterations / self.eps)
        
        # Nyström approximation error: O(m^{-1/2}) where m is number of landmarks
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


