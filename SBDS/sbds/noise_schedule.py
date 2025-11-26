"""Noise scheduling utilities for SBDS."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import KernelDerivativeRFF


class EnhancedAdaptiveNoiseSchedule:
    """Enhanced noise schedule with adaptive timestep selection."""

    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
        num_timesteps: int = 1000,
        rff_features: int = 256,
        device: torch.device | None = None,
        use_mmd: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        # Input validation
        if beta_start <= 0:
            raise ValueError(f"beta_start must be positive, got {beta_start}")
        if beta_end <= 0:
            raise ValueError(f"beta_end must be positive, got {beta_end}")
        if beta_start >= beta_end:
            raise ValueError(f"beta_start must be less than beta_end, got {beta_start} >= {beta_end}")
        if num_timesteps < 1:
            raise ValueError(f"num_timesteps must be at least 1, got {num_timesteps}")
        if rff_features < 1:
            raise ValueError(f"rff_features must be at least 1, got {rff_features}")
        if schedule_type not in ["linear", "cosine", "quadratic"]:
            raise ValueError(
                f"schedule_type must be one of ['linear', 'cosine', 'quadratic'], "
                f"got '{schedule_type}'"
            )

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.rff_features = rff_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mmd = use_mmd
        self.seed = seed

        self.timesteps = torch.linspace(0, 1, num_timesteps)
        self.alpha_bars = self._compute_alpha_bars(self.timesteps)
        self.rff: KernelDerivativeRFF | None = None

    def _compute_alpha_bars(self, t_values: torch.Tensor) -> torch.Tensor:
        if self.schedule_type == "linear":
            betas = self.beta_start + t_values * (self.beta_end - self.beta_start)
            alphas = 1.0 - betas
            return torch.cumprod(alphas, dim=0)
        if self.schedule_type == "cosine":
            s_val = 0.008
            return torch.cos((t_values + s_val) / (1 + s_val) * np.pi / 2).pow(2)
        if self.schedule_type == "quadratic":
            betas = self.beta_start + t_values**2 * (self.beta_end - self.beta_start)
            alphas = 1.0 - betas
            return torch.cumprod(alphas, dim=0)
        raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def __call__(self, t: float) -> float:
        idx = torch.argmin(torch.abs(self.timesteps - t))
        return float(self.alpha_bars[idx])

    def get_beta(self, t: float, dt: float = 1e-3) -> float:
        # Clamp t to avoid tan(pi/2) and log(0) blow-ups near the endpoints
        t_clamped = min(max(t, 0.0), 1.0 - 1e-4)
        if self.schedule_type == "cosine":
            s_val = 0.008
            arg = (t_clamped + s_val) / (1 + s_val) * np.pi / 2
            # tan tends to infinity as arg -> pi/2; clamp to a safe maximum
            tan_val = np.tan(arg)
            tan_val = np.clip(tan_val, -1e6, 1e6)
            return float(2 * tan_val * np.pi / (2 * (1 + s_val)))

        alpha_t = max(self(t_clamped), 1e-6)
        alpha_t_dt = max(self(max(0.0, t_clamped - dt)), 1e-6)
        return float(-(np.log(alpha_t) - np.log(alpha_t_dt)) / dt)

    def _initialize_rff(self, dim: int) -> None:
        if self.rff is not None:
            return

        self.rff = KernelDerivativeRFF(
            input_dim=dim,
            feature_dim=self.rff_features,
            sigma=1.0,
            device=self.device,
            orthogonal=True,
            derivative_order=1,
            seed=self.seed,
        )

    def _compute_mmd(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        x_mean = x_features.mean(dim=0)
        y_mean = y_features.mean(dim=0)
        return torch.sum((x_mean - y_mean) ** 2)

    def _downsample_flatten(self, tensor: torch.Tensor, max_spatial: int = 16) -> torch.Tensor:
        """
        Downsample spatial dimensions before flattening for RFF stability and efficiency.
        Supports 4D (N, C, H, W) and 5D (N, C, D, H, W) tensors; falls back to
        adaptive 1D pooling for generic higher-rank inputs.
        """
        if tensor.dim() == 4:
            _, _, h, w = tensor.shape
            target = (min(max_spatial, h), min(max_spatial, w))
            pooled = F.adaptive_avg_pool2d(tensor, target)
            return pooled.flatten(1)
        if tensor.dim() == 5:
            _, _, d, h, w = tensor.shape
            target = (min(max_spatial, d), min(max_spatial, h), min(max_spatial, w))
            pooled = F.adaptive_avg_pool3d(tensor, target)
            return pooled.flatten(1)
        if tensor.dim() > 2:
            flat = tensor.flatten(1)
            target = min(max_spatial**2, flat.size(1))
            pooled = F.adaptive_avg_pool1d(flat.unsqueeze(1), target).squeeze(1)
            return pooled
        return tensor.reshape(tensor.size(0), -1)

    def get_adaptive_timesteps(
        self,
        n_steps: int,
        score_model: nn.Module,
        device: torch.device,
        shape: Tuple[int, ...],
        snr_weighting: bool = True,
        use_mmd: Optional[bool] = None,
        rng_seed: Optional[int] = None,
    ) -> List[float]:
        use_mmd = self.use_mmd if use_mmd is None else use_mmd

        rng = np.random.default_rng(self.seed if rng_seed is None else rng_seed)

        batch_size = min(8, shape[0])
        test_shape = (batch_size,) + shape[1:]

        try:
            model_dtype = next(score_model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32

        grid_size = 20
        t_grid = torch.linspace(0.05, 0.95, grid_size)

        if use_mmd:
            dummy = torch.zeros(test_shape, device=device, dtype=model_dtype)
            flat_dim = self._downsample_flatten(dummy).shape[1]
            self._initialize_rff(flat_dim)

        data_features = []
        score_norms: List[float] = []

        with torch.no_grad():
            for timestep in t_grid:
                alpha_t = self(timestep.item())
                variance = max(0.0, 1 - alpha_t)
                variance_tensor = torch.as_tensor(
                    variance, device=device, dtype=model_dtype
                )
                noisy = torch.sqrt(variance_tensor) * torch.randn(
                    test_shape, device=device, dtype=model_dtype
                )
                t_tensor = torch.ones(batch_size, device=device, dtype=model_dtype) * timestep
                score = score_model(noisy, t_tensor)
                score_norms.append(float(torch.norm(score.reshape(batch_size, -1), dim=1).mean()))

                if use_mmd and self.rff is not None:
                    noisy_flat = self._downsample_flatten(noisy)
                    data_features.append(self.rff.compute_features(noisy_flat))

        if use_mmd and len(data_features) > 1 and self.rff is not None:
            importance = []
            for idx in range(1, len(data_features)):
                mmd_sq = self._compute_mmd(data_features[idx], data_features[idx - 1])
                if snr_weighting:
                    t_val = t_grid[idx].item()
                    alpha_val = self(t_val)
                    weight = alpha_val * (1 - alpha_val)  # emphasize mid SNR region
                    mmd_sq = mmd_sq * weight
                importance.append(float(mmd_sq))
        else:
            importance = [abs(score_norms[i] - score_norms[i - 1]) for i in range(1, len(score_norms))]

        importance_arr = np.array(importance) + 1e-5
        importance_arr = importance_arr / importance_arr.sum()

        segment_indices = rng.choice(
            len(importance_arr), size=n_steps - 1, p=importance_arr, replace=True
        )
        counts = np.bincount(segment_indices, minlength=len(importance_arr))

        timesteps = [1.0]
        for idx, count in enumerate(counts):
            if count == 0 or idx >= len(t_grid) - 1:
                continue
            left = t_grid[idx].item()
            right = t_grid[idx + 1].item()
            segment_steps = np.linspace(left, right, count + 1)[:-1]
            timesteps.extend(segment_steps.tolist())

        timesteps.append(0.0)
        return sorted(timesteps, reverse=True)
