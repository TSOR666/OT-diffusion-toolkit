
"""Scheduling utilities for the SBDS solver."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from .kernels import KernelDerivativeRFF

__all__ = ["create_standard_timesteps", "spectral_gradient", "EnhancedAdaptiveNoiseSchedule"]


def create_standard_timesteps(
    num_steps: int = 50,
    schedule_type: str = "linear",
) -> List[float]:
    """Create standard timesteps for the reverse process.

    Args:
        num_steps: Number of timesteps
        schedule_type: Type of schedule ('linear', 'quadratic', 'log')

    Returns:
        Timesteps in descending order (T to 0)
    """
    if schedule_type == "linear":
        return torch.linspace(1.0, 0.0, num_steps + 1).tolist()
    if schedule_type == "quadratic":
        return (torch.linspace(1.0, 0.0, num_steps + 1) ** 2).tolist()
    if schedule_type == "log":
        return torch.exp(torch.linspace(0.0, -10.0, num_steps + 1)).tolist()
    raise ValueError(f"Unknown schedule type: {schedule_type}")


def spectral_gradient(
    u: torch.Tensor,
    grid_spacing: Optional[List[float]] = None,
    apply_filter: bool = True,
) -> List[torch.Tensor]:
    """Compute gradient using FFT for spectral accuracy.

    Args:
        u: Input tensor on grid [*grid_shape]
        grid_spacing: Spacing between grid points for each dimension (default: 1.0)
        apply_filter: Whether to apply anti-aliasing filter

    Returns:
        List of gradient components, one for each dimension
    """
    grid_shape = u.shape
    ndim = len(grid_shape)

    if grid_spacing is None:
        grid_spacing = [1.0] * ndim

    gradients: List[torch.Tensor] = []

    for d in range(ndim):
        N = grid_shape[d]
        if N % 2 == 0:
            k = torch.cat(
                [
                    torch.arange(0, N // 2, device=u.device),
                    torch.tensor([0], device=u.device),
                    torch.arange(-N // 2 + 1, 0, device=u.device),
                ]
            )
        else:
            k = torch.cat(
                [
                    torch.arange(0, (N - 1) // 2 + 1, device=u.device),
                    torch.arange(-(N - 1) // 2, 0, device=u.device),
                ]
            )

        k = k * (2 * np.pi / (N * grid_spacing[d]))

        k_shape = [1] * ndim
        k_shape[d] = N
        k = k.reshape(k_shape)

        u_fft = torch.fft.fftn(u)

        if apply_filter:
            filter_shape = [grid_shape[i] for i in range(ndim)]
            center = [s // 2 for s in filter_shape]
            indices = torch.meshgrid(
                [torch.arange(s, device=u.device) for s in filter_shape], indexing="ij"
            )
            dist_sq = sum(((idx - c) / (s * 0.5)) ** 2 for idx, c, s in zip(indices, center, filter_shape))
            smooth_filter = torch.exp(-dist_sq)
            u_fft = u_fft * smooth_filter

        du_fft = (1j * k) * u_fft
        du = torch.fft.ifftn(du_fft)
        gradients.append(du.real)

    return gradients


class EnhancedAdaptiveNoiseSchedule:
    """
    Enhanced noise schedule with RFF-based adaptive importance sampling.
    Incorporates techniques from "Sample Complexity of Sinkhorn Divergences"
    to identify critical timesteps based on distribution changes.
    """
    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'cosine',
        num_timesteps: int = 1000,
        rff_features: int = 256,  # RFF features for importance sampling
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_mmd: bool = True,  # Whether to use MMD distance for importance sampling
        seed: Optional[int] = None,  # Random seed for reproducibility
    ):
        """
        Initialize enhanced adaptive noise schedule.
        
        Args:
            beta_start: Starting value for noise variance
            beta_end: Ending value for noise variance
            schedule_type: Type of schedule ('linear', 'cosine', 'quadratic')
            num_timesteps: Number of timesteps for precomputing schedule
            rff_features: Number of random features for RFF
            device: Device to use for computation
            use_mmd: Whether to use MMD distance for importance sampling
            seed: Random seed for reproducibility
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.rff_features = rff_features
        self.device = device
        self.use_mmd = use_mmd
        self.seed = seed
        
        # Precompute alphas for efficiency
        self.timesteps = torch.linspace(0, 1, num_timesteps)
        self.alpha_bars = self._compute_alpha_bars(self.timesteps)
        
        # Initialize kernel approximator for importance sampling
        self.rff = None
    
    def _compute_alpha_bars(self, t_values: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha_bar values for given timesteps.
        
        Args:
            t_values: Timestep values in [0, 1]
        
        Returns:
            Alpha_bar values
        """
        if self.schedule_type == 'linear':
            betas = self.beta_start + t_values * (self.beta_end - self.beta_start)
            alphas = 1.0 - betas
            return torch.cumprod(alphas, dim=0)
        elif self.schedule_type == 'cosine':
            s = 0.008
            return torch.cos((t_values + s) / (1 + s) * np.pi / 2).pow(2)
        elif self.schedule_type == 'quadratic':
            betas = self.beta_start + t_values**2 * (self.beta_end - self.beta_start)
            alphas = 1.0 - betas
            return torch.cumprod(alphas, dim=0)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def __call__(self, t: float) -> float:
        """
        Get alpha_bar (cumulative product of alphas) at time t.
        
        Args:
            t: Time point in [0, 1]
        
        Returns:
            alpha_bar at time t
        """
        # Find closest precomputed value for efficiency
        idx = torch.argmin(torch.abs(self.timesteps - t))
        return self.alpha_bars[idx].item()
    
    def get_beta(self, t: float, dt: float = 1e-3) -> float:
        """
        Get beta(t) = -d log(alpha_bar(t))/dt
        
        Args:
            t: Time point
            dt: Small time increment for numerical derivative
            
        Returns:
            beta at time t
        """
        # Use analytical derivative for cosine schedule
        if self.schedule_type == 'cosine':
            s = 0.008
            # α_bar(t) = cos²((t+s)/(1+s) * π/2)
            # log(α_bar(t)) = 2*log(cos((t+s)/(1+s) * π/2))
            # d/dt log(α_bar(t)) = -2*tan((t+s)/(1+s) * π/2) * π/(2(1+s))
            arg = (t + s) / (1 + s) * np.pi / 2
            beta = 2 * np.tan(arg) * np.pi / (2 * (1 + s))
            return beta
        
        # Numerical derivative for other schedules
        alpha_t = self(t)
        alpha_t_dt = self(max(0, t - dt))
        
        # Avoid log(0)
        if alpha_t <= 0 or alpha_t_dt <= 0:
            return self.beta_end
            
        beta = -(np.log(alpha_t) - np.log(alpha_t_dt)) / dt
        return beta
    
    def _initialize_rff(self, dim: int):
        """
        Initialize the RFF approximator for efficient MMD computation.
        
        Args:
            dim: Dimensionality of the data
        """
        if self.rff is None:
            # Set seed if provided, for reproducibility
            if self.seed is not None:
                torch.manual_seed(self.seed)
                np.random.seed(self.seed)
            
            self.rff = KernelDerivativeRFF(
                input_dim=dim,
                feature_dim=self.rff_features,
                sigma=1.0,  # Will be adjusted based on data
                device=self.device,
                orthogonal=True,
                derivative_order=1,
                seed=self.seed
            )
    
    def _compute_mmd(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD) squared using RFF features.
        
        Args:
            x_features: RFF features of first distribution [batch_x, feature_dim]
            y_features: RFF features of second distribution [batch_y, feature_dim]
            
        Returns:
            MMD squared (maintains additive property)
        """
        # MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        # With RFF: MMD^2(P, Q) ≈ ||mean(Φ(x)) - mean(Φ(y))||^2
        
        x_mean = x_features.mean(dim=0)
        y_mean = y_features.mean(dim=0)
        
        mmd_squared = torch.sum((x_mean - y_mean) ** 2)
        return mmd_squared  # Return MMD^2 for additive property
    
    def get_adaptive_timesteps(
        self, 
        n_steps: int, 
        score_model: nn.Module, 
        device: torch.device,
        shape: Tuple[int, ...],
        snr_weighting: bool = True,
        use_mmd: Optional[bool] = None,
        rng_seed: Optional[int] = None
    ) -> List[float]:
        """
        Generate adaptive timesteps based on score function behavior.
        Places more steps where the score function changes rapidly.
        
        Uses an MMD-based approach from "Sample Complexity of Sinkhorn Divergences"
        to identify timesteps where distributions change most rapidly.
        
        Args:
            n_steps: Number of timesteps to generate
            score_model: Score model for estimating score norms
            device: Device to use for computation
            shape: Shape of test samples to use
            snr_weighting: Whether to use SNR-based weighting
            use_mmd: Whether to use MMD distance (overrides self.use_mmd)
            rng_seed: Random seed for reproducibility
            
        Returns:
            List of timesteps
        """
        # Use instance setting if not explicitly provided
        use_mmd = self.use_mmd if use_mmd is None else use_mmd
        
        # Set random seed for reproducibility
        if rng_seed is not None:
            np.random.seed(rng_seed)
        elif self.seed is not None:
            np.random.seed(self.seed)
        
        # Generate a small batch of test samples
        batch_size = min(8, shape[0])
        test_shape = (batch_size,) + shape[1:]
        
        # Start from noise
        x = torch.randn(test_shape, device=device)
        
        # Sample a grid of timesteps
        grid_size = 20
        t_grid = torch.linspace(0.05, 0.95, grid_size)
        
        # Initialize RFF if using MMD
        if use_mmd:
            self._initialize_rff(np.prod(shape[1:]))
        
        # Store data at each timestep
        data_at_timesteps = []
        score_norms = []
        rff_features = []
        
        # Generate noisy samples at each grid point
        with torch.no_grad():
            for t in t_grid:
                alpha_t = self(t)
                x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * torch.randn_like(x)
                t_tensor = torch.ones(batch_size, device=device) * t
                score = score_model(x_t, t_tensor)
                
                # Store data
                data_at_timesteps.append(x_t)
                
                # Compute score norm
                norm = torch.norm(score.reshape(batch_size, -1), dim=1).mean().item()
                score_norms.append(norm)
                
                # Compute RFF features if using MMD
                if use_mmd:
                    x_t_flat = x_t.reshape(batch_size, -1)
                    rff_feat = self.rff.compute_features(x_t_flat)
                    rff_features.append(rff_feat)
        
        # Compute importance of each region based on distribution changes
        if use_mmd and len(rff_features) > 1:
            # Using MMD² distance between consecutive timesteps
            importance = []
            for i in range(1, len(rff_features)):
                # MMD² distance approximates squared Wasserstein distance between distributions
                mmd_squared = self._compute_mmd(rff_features[i], rff_features[i-1])
                # Apply SNR weighting if requested
                if snr_weighting:
                    t = t_grid[i].item()
                    alpha_t = self(t)
                    snr = alpha_t / (1 - alpha_t)
                    weight = snr / (1 + snr)
                    mmd_squared = mmd_squared * weight
                
                importance.append(mmd_squared.item())
        else:
            # Using simple score norm changes
            importance = []
            for i in range(1, len(score_norms)):
                importance.append(abs(score_norms[i] - score_norms[i-1]))
        
        # Add small constant to ensure all regions get some timesteps
        importance = np.array(importance) + 1e-5
        importance = importance / importance.sum()
        
        # Efficiently distribute timesteps according to importance using a single call
        # Rather than multinomial sampling in a loop, which creates bias
        segment_indices = np.random.choice(
            len(importance), 
            size=n_steps-1, 
            p=importance,
            replace=True
        )
        
        # Convert to counts using bincount
        counts = np.bincount(segment_indices, minlength=len(importance))
        
        # Generate fine timesteps within each selected grid segment
        timesteps = [1.0]  # Start with t=1 (noise)
        for i, count in enumerate(counts):
            if count > 0:
                # Add timesteps within this segment
                if i < len(t_grid) - 1:  # Ensure we don't go out of bounds
                    left, right = t_grid[i].item(), t_grid[i+1].item()
                    segment_steps = np.linspace(left, right, count + 1)[:-1]
                    timesteps.extend(segment_steps.tolist())
        
        # Add t=0 (clean data)
        timesteps.append(0.0)
        
        # Sort in descending order (from noise to clean)
        return sorted(timesteps, reverse=True)


