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
        """Determine if we should use sliced OT based on memory.

        Expects ``x_batch`` shaped (B, N, d).
        """
        if x_batch.dim() != 3:
            raise ValueError(f"Expected (B, N, d) tensor, got shape {tuple(x_batch.shape)}")

        B, N, _ = x_batch.shape
        element_size = x_batch.element_size()
        cost_matrix_memory = B * N * N * element_size

        return cost_matrix_memory > self.memory_limit_bytes

    def _reshape_to_points(self, tensor: torch.Tensor):
        """Flatten arbitrary inputs to (B, N, d) and return a restore hook."""
        if tensor.dim() == 2:
            # Treat feature dimension as points with scalar features
            points = tensor.unsqueeze(-1)
            def restore(out: torch.Tensor) -> torch.Tensor:
                return out.squeeze(-1)
            return points, restore

        if tensor.dim() == 3:
            return tensor, lambda out: out

        if tensor.dim() < 2:
            raise ValueError(f"Input must have batch dimension, got shape {tuple(tensor.shape)}")

        batch = tensor.shape[0]
        channel = tensor.shape[1]
        spatial = tensor.shape[2:]
        if len(spatial) == 0:
            raise ValueError(f"Input with shape {tuple(tensor.shape)} is not compatible with OT flattening.")

        # Move channel to the last axis then flatten spatial dims into point dimension
        points = tensor.movedim(1, -1).reshape(batch, -1, channel)

        def restore(out: torch.Tensor) -> torch.Tensor:
            return out.reshape(batch, *spatial, channel).movedim(-1, 1)

        return points, restore

    def transport(self, x_batch: torch.Tensor, y_batch: torch.Tensor,
                  eps: Union[float, torch.Tensor], n_projections: int = 100) -> torch.Tensor:
        """Choose between full and sliced OT based on memory"""
        x_points, restore = self._reshape_to_points(x_batch)
        y_points, _ = self._reshape_to_points(y_batch)

        if x_points.shape != y_points.shape:
            raise ValueError(f"Shape mismatch: x={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")

        # Convert eps to float if tensor
        if torch.is_tensor(eps):
            eps_val = eps.item()
        else:
            eps_val = eps

        if self.should_use_sliced(x_points):
            N = x_points.shape[1]
            if self.projection_fn:
                n_proj = self.projection_fn(N, n_projections)
            elif N > 256 * 256:
                n_proj = min(n_projections, max(16, int(256**2 / N)))
            else:
                n_proj = n_projections
            result = self._sliced_ot_fixed(x_points, y_points, n_proj)
        else:
            result = self._full_ot(x_points, y_points, eps_val)

        return restore(result)

    def _sliced_ot_fixed(self, x: torch.Tensor, y: torch.Tensor,
                        n_projections: int) -> torch.Tensor:
        """Sliced OT with deterministic projections

        CRITICAL GRADIENT LIMITATION:
        This implementation uses scatter_ with integer indices from argsort, which
        breaks gradient flow with respect to x positions during training.

        - Gradients flow through y_sorted (values) ✓
        - Gradients DO NOT flow through x_indices (integer sort order) ✗

        Consequence:
        - For INFERENCE (sampling): Mathematically valid approximation of OT
        - For TRAINING: Loss gradient w.r.t. x positions is ZERO or incorrect

        If you need gradients for training position-dependent losses, use:
        1. Soft-sort (differentiable sorting) instead of hard argsort
        2. Direct Wasserstein distance: torch.sum((x_sorted - y_sorted)**2)
        3. Full Sinkhorn (_full_ot) for small problems
        """
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

        for i in range(n_projections):
            # Generate and normalize theta in FP32 for stability
            theta = torch.randn(d, device=device, dtype=torch.float32, generator=self.generator)
            theta = F.normalize(theta, dim=0)
            theta = theta.to(dtype)  # Cast back to data dtype

            # Use matmul without out= to preserve gradients
            x_proj = torch.matmul(x.view(B, N, d), theta)
            y_proj = torch.matmul(y.view(B, N, d), theta)

            _, x_indices = torch.sort(x_proj, dim=1)
            y_sorted, _ = torch.sort(y_proj, dim=1)

            # WARNING: scatter_ breaks gradient flow w.r.t. x positions
            # but gradients DO flow through y_sorted values
            transported_proj = transported_proj.scatter(1, x_indices, y_sorted)

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
        # Barycentric projection requires dividing by row mass (uniform = 1/N). Without
        # this scaling the map is biased toward zero by roughly a factor of 1/N.
        row_sums = P_fp32.sum(dim=2, keepdim=True).clamp_min(1e-12)
        out_fp32 = torch.bmm(P_fp32, y_fp32) / row_sums
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
        """Support for custom marginals, defaults to uniform

        Includes stability checks and fallback to prevent gradient explosions.
        """
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

        # CRITICAL: Check for NaN or degenerate coupling matrix
        # This prevents gradient explosions in barycentric projection
        if torch.isnan(P).any() or torch.isinf(P).any():
            logger.warning(f"Sinkhorn produced NaN/Inf (eps={eps:.2e}). Falling back to uniform coupling.")
            # Fallback to uniform coupling (identity-like for square matrices)
            P = torch.eye(n, m, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) / max(n, m)
            return P

        # Check row sum deviation from uniform (should be ~1/n for each row)
        row_sums = P.sum(dim=2)
        expected_row_sum = 1.0 / n
        max_deviation = torch.abs(row_sums - expected_row_sum).max()

        # If coupling is severely degenerate (e.g. rows sum to near-zero), fallback
        if max_deviation > 0.5 / n:  # Allow 50% deviation
            logger.warning(
                f"Sinkhorn coupling degenerate (max row sum deviation: {max_deviation:.2e}, "
                f"expected: {expected_row_sum:.2e}, eps={eps:.2e}). Falling back to uniform coupling."
            )
            P = torch.eye(n, m, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) / max(n, m)
            return P

        return P





class MomentumTransport(nn.Module):
    """Transport with momentum for accelerated convergence

    Uses shape-based velocity caching to handle varying batch sizes and
    multi-scale/patch-based processing without destroying momentum state.
    """

    def __init__(self, beta: float = 0.9, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.beta = beta
        self.device = device
        # Use dict cache instead of single buffer to support multiple shapes
        # Do NOT register as buffer - causes issues with state_dict and JIT
        self._velocity_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def reset_velocity(self, shape: Optional[Tuple[int, ...]] = None):
        """Reset momentum velocity for specific shape or all shapes"""
        if shape is not None:
            if shape in self._velocity_cache:
                del self._velocity_cache[shape]
                logger.debug(f"Reset velocity for shape {shape}")
        else:
            self._velocity_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.debug("Reset all velocity cache")

    def apply_transport(self, x: torch.Tensor, drift: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """Apply transport with momentum (handles shape changes via caching)"""
        shape_key = tuple(x.shape)

        # Get or create velocity for this shape
        if shape_key not in self._velocity_cache:
            self._velocity_cache[shape_key] = torch.zeros_like(x)
            self._cache_misses += 1
            if self._cache_misses == 1:
                logger.debug(f"Created velocity cache for shape {shape_key}")
            elif self._cache_misses > 1:
                logger.warning(
                    f"Momentum cache miss (shape {shape_key}). "
                    f"Frequent misses ({self._cache_misses} shapes) may indicate "
                    f"momentum is being reset too often (e.g., varying batch size)."
                )
        else:
            self._cache_hits += 1

        velocity = self._velocity_cache[shape_key]

        # Update velocity with exponential moving average
        velocity = (self.beta * velocity.detach() + (1 - self.beta) * drift).detach()
        self._velocity_cache[shape_key] = velocity

        lookahead = x + self.beta * velocity

        transport_weight = self._compute_adaptive_weight(lookahead, alpha_bar_t)

        return x + transport_weight * velocity

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

        alpha_tensor = torch.as_tensor(
            alpha_bar_t,
            device=x.device,
            dtype=x.dtype if x.dtype.is_floating_point else torch.float32
        )
        alpha_clamped = torch.clamp(alpha_tensor, 0.0, 1.0)
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
        """Compute transport at multiple scales with anti-aliased downsampling"""
        s = 12.0
        gate = torch.sigmoid(((1 - alpha_bar_t).mean() - 0.5) * s).to(x.dtype)

        transports = []
        weights = []

        for scale in self.scales:
            if scale < 0.5 and x.shape[-1] > 512:
                continue

            if scale < 1.0:
                # CRITICAL: Use 'area' mode for downsampling to prevent aliasing
                # 'bilinear' causes high-frequency artifacts to fold into low frequencies
                x_scaled = F.interpolate(x, scale_factor=scale, mode='area')
                drift_scaled = F.interpolate(drift, scale_factor=scale, mode='area')
            else:
                x_scaled, drift_scaled = x, drift

            transport = self._compute_scale_transport(x_scaled, drift_scaled, scale)

            if scale < 1.0:
                # Upsampling uses bilinear (no aliasing risk)
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



