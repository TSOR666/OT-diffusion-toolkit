"""Transport modules for FastSB-OT."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common
from .config import FastSBOTConfig

logger = common.logger
compile_function_fixed = common.compile_function_fixed
log_sum_exp_stabilized = common.log_sum_exp_stabilized

__all__ = [
    "SlicedOptimalTransport",
    "MomentumTransport",
    "HierarchicalBridge",
    "TransportModule",
]


class SlicedOptimalTransport:
    """Memory-efficient sliced OT with proper N-point handling"""

    def __init__(self, memory_limit_mb: int = 100, sinkhorn_iters: int = 50,
                 sinkhorn_tol: float = 1e-5, projection_fn: Optional[Callable[[int, int], int]] = None,
                 generator: Optional[torch.Generator] = None):
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tol = sinkhorn_tol
        self.projection_fn = projection_fn
        self.generator = generator

    def should_use_sliced(self, x_batch: torch.Tensor) -> bool:
        """Determine if we should use sliced OT based on memory"""
        B = x_batch.shape[0]
        N = x_batch.shape[1]
        element_size = x_batch.element_size()
        cost_matrix_memory = B * N * N * element_size

        return cost_matrix_memory > self.memory_limit_bytes

    def transport(self, x_batch: torch.Tensor, y_batch: torch.Tensor,
                  eps: Union[float, torch.Tensor], n_projections: int = 100) -> torch.Tensor:
        """Choose between full and sliced OT based on memory"""
        # Handle flat (B, D) case by treating as (B, D, 1)
        if x_batch.dim() == 2:
            x_batch = x_batch.view(x_batch.shape[0], -1, 1)
            y_batch = y_batch.view(y_batch.shape[0], -1, 1)
            was_flat = True
        else:
            was_flat = False

        # Convert eps to float if tensor
        if torch.is_tensor(eps):
            eps_val = eps.item()
        else:
            eps_val = eps

        if self.should_use_sliced(x_batch):
            N = x_batch.shape[1]
            if self.projection_fn:
                n_proj = self.projection_fn(N, n_projections)
            elif N > 256 * 256:
                n_proj = min(n_projections, max(16, int(256**2 / N)))
            else:
                n_proj = n_projections
            result = self._sliced_ot_fixed(x_batch, y_batch, n_proj)
        else:
            result = self._full_ot(x_batch, y_batch, eps_val)

        if was_flat:
            result = result.squeeze(-1)

        return result

    def _sliced_ot_fixed(self, x: torch.Tensor, y: torch.Tensor,
                        n_projections: int) -> torch.Tensor:
        """Sliced OT with deterministic projections"""
        B, N, d = x.shape
        device = x.device
        dtype = x.dtype

        if os.environ.get('FASTSBOT_ASSERTS', '0') == '1':
            if x.shape != y.shape:
                raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor (B,N,d), got {x.dim()}D")

        transported = torch.zeros_like(x)
        transported_proj = x.new_zeros(B, N)  # Zero-init for safety
        x_proj = x.new_empty(B, N)
        y_proj = y.new_empty(B, N)

        for i in range(n_projections):
            # Generate and normalize theta in FP32 for stability
            theta = torch.randn(d, device=device, dtype=torch.float32, generator=self.generator)
            theta = F.normalize(theta, dim=0)
            theta = theta.to(dtype)  # Cast back to data dtype

            torch.matmul(x.view(B, N, d), theta, out=x_proj.view(B, N))
            torch.matmul(y.view(B, N, d), theta, out=y_proj.view(B, N))

            _, x_indices = torch.sort(x_proj, dim=1)
            y_sorted, _ = torch.sort(y_proj, dim=1)

            transported_proj.scatter_(1, x_indices, y_sorted)

            diff_proj = (transported_proj - x_proj).unsqueeze(-1)
            transported += diff_proj * theta.unsqueeze(0).unsqueeze(0)

        return x + transported / n_projections

    def _full_ot(self, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        """Full OT with FP32 Sinkhorn and matmul for numeric stability"""
        B, N, d = x.shape

        if os.environ.get('FASTSBOT_ASSERTS', '0') == '1':
            if x.shape != y.shape:
                raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {x.dim()}D")

        x_expanded = x.unsqueeze(2)
        y_expanded = y.unsqueeze(1)

        diff = x_expanded - y_expanded
        C = torch.sum(diff ** 2, dim=-1)

        P_fp32 = self._sinkhorn_batch_fixed(C.float(), eps)
        y_fp32 = y.float() if y.dtype != torch.float32 else y
        out_fp32 = torch.bmm(P_fp32, y_fp32)
        return out_fp32.to(y.dtype)

    def _sinkhorn_batch_fixed(
        self,
        C_batch: torch.Tensor,
        eps: float,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
        row_marginals: Optional[torch.Tensor] = None,
        col_marginals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Support for custom marginals, defaults to uniform"""
        B, n, m = C_batch.shape
        device = C_batch.device
        dtype = C_batch.dtype

        if max_iter is None:
            max_iter = self.sinkhorn_iters
        if tol is None:
            tol = self.sinkhorn_tol

        C_min = C_batch.reshape(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        C_normalized = C_batch - C_min

        if row_marginals is None:
            log_a = C_batch.new_full((B, n), -math.log(n))
        else:
            log_a = torch.log(row_marginals + 1e-10)

        if col_marginals is None:
            log_b = C_batch.new_full((B, m), -math.log(m))
        else:
            log_b = torch.log(col_marginals + 1e-10)

        K_log = -C_normalized / eps

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        for iteration in range(max_iter):
            log_u_prev = log_u.clone()
            log_v_prev = log_v.clone()

            log_v = log_b - log_sum_exp_stabilized(K_log + log_u.unsqueeze(-1), dim=1)
            log_u = log_a - log_sum_exp_stabilized(K_log + log_v.unsqueeze(1), dim=2)

            if (iteration % 5 == 0) and (not C_batch.is_cuda):
                err_u = torch.abs(log_u - log_u_prev).max()
                err_v = torch.abs(log_v - log_v_prev).max()
                if max(err_u.item(), err_v.item()) < tol:
                    break

        log_v = log_b - log_sum_exp_stabilized(K_log + log_u.unsqueeze(-1), dim=1)

        log_P = log_u.unsqueeze(-1) + K_log + log_v.unsqueeze(1)
        P = torch.exp(log_P)

        return P





class MomentumTransport(nn.Module):
    """Transport with momentum for accelerated convergence"""

    def __init__(self, beta: float = 0.9, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.beta = beta
        self.device = device
        self.register_buffer('velocity', None, persistent=False)
        self.register_buffer('velocity_shape', None, persistent=False)

    def reset_velocity(self):
        """Reset momentum velocity"""
        self.velocity = None
        self.velocity_shape = None

    def apply_transport(self, x: torch.Tensor, drift: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Apply transport with momentum (handles shape changes)"""
        if self.velocity is not None and x.shape != self.velocity_shape:
            self.reset_velocity()

        if self.velocity is None:
            self.velocity = torch.zeros_like(x)
            self.velocity_shape = x.shape

        self.velocity = (self.beta * self.velocity.detach() + (1 - self.beta) * drift).detach()

        lookahead = x + self.beta * self.velocity

        transport_weight = self._compute_adaptive_weight(lookahead, alpha_bar_t)

        return x + transport_weight * self.velocity

    def _compute_adaptive_weight(self, x: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weight for transport"""
        if x.dim() == 4 and x.shape[-1] > 1 and x.shape[-2] > 1:
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            dx_center = dx[:, :, :-1, :]
            dy_center = dy[:, :, :, :-1]
            grad_mag = torch.sqrt(dx_center**2 + dy_center**2 + 1e-8)
            smoothness = 1.0 / (1.0 + grad_mag.mean(dim=(1, 2, 3), keepdim=True))
        else:
            smoothness = x.new_ones(x.shape[0], 1)

        alpha_clamped = torch.clamp(alpha_bar_t, 0.0, 1.0)
        time_weight = 1.0 - alpha_clamped
        weight = smoothness * (0.5 + 0.5 * time_weight)

        for _ in range(x.dim() - len(weight.shape)):
            weight = weight.unsqueeze(-1)

        return weight





class HierarchicalBridge(nn.Module):
    """Multi-scale Schrodinger bridge decomposition"""

    def __init__(self, scales: List[float] = [1.0, 0.5, 0.25], device: torch.device = torch.device('cpu')):
        super().__init__()
        self.scales = scales
        self.device = device
        self.bridge_cache = {}

    @compile_function_fixed(dynamic=True, use_global_cache=True)
    def compute_multiscale_transport(self, x: torch.Tensor, drift: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Compute transport at multiple scales"""
        s = 12.0
        gate = torch.sigmoid(((1 - alpha_bar_t).mean() - 0.5) * s).to(x.dtype)

        transports = []
        weights = []

        for scale in self.scales:
            if scale < 0.5 and x.shape[-1] > 512:
                continue

            if scale < 1.0:
                x_scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
                drift_scaled = F.interpolate(drift, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                x_scaled, drift_scaled = x, drift

            transport = self._compute_scale_transport(x_scaled, drift_scaled, scale)

            if scale < 1.0:
                transport = F.interpolate(transport, size=x.shape[-2:], mode='bilinear', align_corners=False)

            transports.append(transport)

            weight = self._compute_scale_weight(x, transport, scale, alpha_bar_t)
            weights.append(weight)

        if not transports:
            multiscale = x + drift
        else:
            weights = torch.stack(weights, dim=0)
            weights = F.softmax(weights / 0.1, dim=0).to(x.dtype)
            multiscale = sum(w * tr for w, tr in zip(weights, transports))

        return (1 - gate) * (x + drift) + gate * multiscale

    def _compute_scale_transport(self, x: torch.Tensor, drift: torch.Tensor, scale: float) -> torch.Tensor:
        """Transport computation at specific scale"""
        if scale < 1.0:
            kernel_size = int(3 / scale)
            if kernel_size % 2 == 0:
                kernel_size += 1
            drift = F.avg_pool2d(drift, kernel_size, stride=1, padding=kernel_size//2)

        return x + drift

    def _compute_scale_weight(self, x: torch.Tensor, transport: torch.Tensor,
                            scale: float, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Compute importance weight for each scale"""
        x_fft = torch.fft.rfft2(x)
        t_fft = torch.fft.rfft2(transport)

        try:
            freq_y = torch.fft.fftfreq(x.shape[-2], device=x.device)
        except TypeError:
            freq_y = torch.fft.fftfreq(x.shape[-2]).to(x.device)
        try:
            freq_x = torch.fft.rfftfreq(x.shape[-1], device=x.device)
        except TypeError:
            freq_x = torch.fft.rfftfreq(x.shape[-1]).to(x.device)

        freq_mag = torch.sqrt(freq_y[:, None]**2 + freq_x[None, :]**2)

        if scale >= 1.0:
            freq_weight = freq_mag
        else:
            freq_weight = 1.0 - freq_mag

        energy = (torch.abs(t_fft - x_fft) * freq_weight[None, None, :, :]).mean()

        time_factor = 1.0 - (alpha_bar_t if torch.is_tensor(alpha_bar_t) else x.new_tensor(alpha_bar_t))

        return energy * (1.0 + time_factor * scale)





class TransportModule(nn.Module):
    """Optimized transport operations module with sliced OT"""

    def __init__(self, config: FastSBOTConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # Pass generator to SlicedOptimalTransport
        self.sliced_ot = SlicedOptimalTransport(
            config.memory_limit_ot_mb,
            config._sinkhorn_iterations,
            config.sinkhorn_tolerance,
            config.sliced_ot_projection_fn,
            generator=config.generator
        )



