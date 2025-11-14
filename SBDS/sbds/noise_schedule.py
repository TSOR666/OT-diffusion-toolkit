"""Noise scheduling utilities for SBDS."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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
        if self.schedule_type == "cosine":
            s_val = 0.008
            arg = (t + s_val) / (1 + s_val) * np.pi / 2
            return float(2 * np.tan(arg) * np.pi / (2 * (1 + s_val)))

        alpha_t = self(t)
        alpha_t_dt = self(max(0.0, t - dt))
        if alpha_t <= 0 or alpha_t_dt <= 0:
            return self.beta_end
        return float(-(np.log(alpha_t) - np.log(alpha_t_dt)) / dt)

    def _initialize_rff(self, dim: int) -> None:
        if self.rff is not None:
            return

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

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

        if rng_seed is not None:
            np.random.seed(rng_seed)
        elif self.seed is not None:
            np.random.seed(self.seed)

        batch_size = min(8, shape[0])
        test_shape = (batch_size,) + shape[1:]

        try:
            model_dtype = next(score_model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32

        grid_size = 20
        t_grid = torch.linspace(0.05, 0.95, grid_size)

        if use_mmd:
            self._initialize_rff(int(np.prod(shape[1:])))

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
                    noisy_flat = noisy.reshape(batch_size, -1)
                    data_features.append(self.rff.compute_features(noisy_flat))

        if use_mmd and len(data_features) > 1 and self.rff is not None:
            importance = []
            for idx in range(1, len(data_features)):
                mmd_sq = self._compute_mmd(data_features[idx], data_features[idx - 1])
                if snr_weighting:
                    t_val = t_grid[idx].item()
                    alpha_val = self(t_val)
                    snr = alpha_val / (1 - alpha_val)
                    weight = snr / (1 + snr)
                    mmd_sq = mmd_sq * weight
                importance.append(float(mmd_sq))
        else:
            importance = [abs(score_norms[i] - score_norms[i - 1]) for i in range(1, len(score_norms))]

        importance_arr = np.array(importance) + 1e-5
        importance_arr = importance_arr / importance_arr.sum()

        segment_indices = np.random.choice(
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
