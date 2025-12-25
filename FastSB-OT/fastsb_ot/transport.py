"""Transport modules for FastSB-OT."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common
from .config import FastSBOTConfig

compile_function_fixed = common.compile_function_fixed
log_sum_exp_stabilized = common.log_sum_exp_stabilized
check_tensor_finite = common.check_tensor_finite
runtime_asserts_enabled = common.runtime_asserts_enabled
nan_checks_enabled = common.nan_checks_enabled

__all__ = [
    "SlicedOptimalTransport",
    "MomentumTransport",
    "HierarchicalBridge",
    "TransportModule",
]


class SlicedOptimalTransport:
    """Memory-efficient sliced OT with proper N-point handling"""

    def __init__(
        self,
        memory_limit_mb: int = 100,
        sinkhorn_iters: int = 50,
        sinkhorn_tol: float = 1e-5,
        projection_fn: Optional[Callable[[int, int], int]] = None,
        generator: Optional[torch.Generator] = None,
        runtime_asserts: bool = True,
        nan_checks: bool = True,
        sinkhorn_mass_tolerance: float = 2e-2,
    ) -> None:
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tol = sinkhorn_tol
        self.projection_fn = projection_fn
        self.generator = generator
        self.runtime_asserts = runtime_asserts
        self.nan_checks = nan_checks
        self.sinkhorn_mass_tolerance = sinkhorn_mass_tolerance

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

    def _reshape_to_points(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Flatten arbitrary inputs to (B, N, d) and return a restore hook."""
        if tensor.dim() == 2:
            # Treat feature dimension as points with scalar features
            points = tensor.unsqueeze(-1)  # (B, N) -> (B, N, 1)
            restore_fn: Callable[[torch.Tensor], torch.Tensor] = lambda out: out.squeeze(-1)
            return points, restore_fn

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
        points = tensor.movedim(1, -1).reshape(batch, -1, channel)  # (B, C, *S) -> (B, N, C)

        restore_fn = lambda out: out.reshape(batch, *spatial, channel).movedim(-1, 1)  # (B, N, C) -> (B, C, *S)

        return points, restore_fn

    def transport(self, x_batch: torch.Tensor, y_batch: torch.Tensor,
                  eps: Union[float, torch.Tensor], n_projections: int = 100) -> torch.Tensor:
        """Choose between full and sliced OT based on memory.

        Shapes:
            - x_batch, y_batch: (B, N, d) or (B, C, H, W) or (B, N)
            - return: same shape as inputs
        """
        if self.nan_checks:
            check_tensor_finite("x_batch", x_batch, enabled=True)
            check_tensor_finite("y_batch", y_batch, enabled=True)
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

    def _projection_chunk_size(self, batch: int, points: int, element_size: int, max_projections: int) -> int:
        bytes_per_proj = batch * points * (3 * element_size + 8)
        if bytes_per_proj <= 0:
            return max_projections
        budget = max(self.memory_limit_bytes // 2, bytes_per_proj)
        return max(1, min(max_projections, budget // bytes_per_proj))

    def _sliced_ot_fixed(self, x: torch.Tensor, y: torch.Tensor,
                        n_projections: int) -> torch.Tensor:
        """Sliced OT with deterministic projections.

        Shapes:
            - x, y: (B, N, d)
            - return: (B, N, d)
        """
        B, N, d = x.shape
        device = x.device
        dtype = x.dtype

        if self.runtime_asserts:
            if x.shape != y.shape:
                raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor (B,N,d), got {x.dim()}D")

        if self.nan_checks:
            check_tensor_finite("x", x, enabled=True)
            check_tensor_finite("y", y, enabled=True)

        transported = torch.zeros_like(x)
        gen = self.generator if self.generator is not None else torch.Generator(device=device)

        chunk = self._projection_chunk_size(B, N, x.element_size(), n_projections)
        done = 0
        while done < n_projections:
            current = min(chunk, n_projections - done)
            theta = common._randn_like_compat(
                torch.empty((current, d), device=device, dtype=torch.float32),
                gen
            )  # (P, d)
            theta = F.normalize(theta, dim=1, eps=1e-8).to(dtype)  # (P, d) -> (P, d)

            x_proj = torch.matmul(x, theta.transpose(0, 1))  # (B, N, d) @ (d, P) -> (B, N, P)
            y_proj = torch.matmul(y, theta.transpose(0, 1))  # (B, N, d) @ (d, P) -> (B, N, P)

            _, x_indices = torch.sort(x_proj, dim=1)  # (B, N, P) -> (B, N, P)
            y_sorted, _ = torch.sort(y_proj, dim=1)  # (B, N, P) -> (B, N, P)

            transported_proj = torch.empty_like(x_proj)  # (B, N, P)
            transported_proj.scatter_(1, x_indices, y_sorted)  # (B, N, P) scatter -> (B, N, P)

            diff = transported_proj - x_proj  # (B, N, P)
            transported += torch.einsum("bnp,pd->bnd", diff, theta)  # (B, N, P) x (P, d) -> (B, N, d)
            done += current

        return x + transported / n_projections

    def _full_ot(self, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        """Full OT with FP32 Sinkhorn and matmul for numeric stability.

        Shapes:
            - x, y: (B, N, d)
            - return: (B, N, d)
        """
        B, N, d = x.shape

        if self.runtime_asserts:
            if x.shape != y.shape:
                raise ValueError(f"Shape mismatch: x={x.shape}, y={y.shape}")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
        if self.nan_checks:
            check_tensor_finite("x", x, enabled=True)
            check_tensor_finite("y", y, enabled=True)

        x_expanded = x.unsqueeze(2)  # (B, N, d) -> (B, N, 1, d)
        y_expanded = y.unsqueeze(1)  # (B, N, d) -> (B, 1, N, d)

        diff = x_expanded - y_expanded  # (B, N, 1, d) - (B, 1, N, d) broadcast -> (B, N, N, d)
        C = torch.sum(diff ** 2, dim=-1)  # (B, N, N, d) -> (B, N, N)

        P_fp32 = self._sinkhorn_batch_fixed(C.float(), eps)
        y_fp32 = y.float() if y.dtype != torch.float32 else y
        # Barycentric projection requires dividing by row mass (uniform = 1/N). Without
        # this scaling the map is biased toward zero by roughly a factor of 1/N.
        row_sums = P_fp32.sum(dim=2, keepdim=True).clamp_min(1e-12)  # (B, N, N) -> (B, N, 1)
        out_fp32 = torch.bmm(P_fp32, y_fp32) / row_sums  # (B, N, N) @ (B, N, d) -> (B, N, d); / (B, N, 1)
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
        """Support for custom marginals, defaults to uniform.

        Shapes:
            - C_batch: (B, n, m)
            - row_marginals: (B, n), col_marginals: (B, m)
            - return: (B, n, m)
        """
        B, n, m = C_batch.shape

        if max_iter is None:
            max_iter = self.sinkhorn_iters
        if tol is None:
            tol = self.sinkhorn_tol

        C_min = C_batch.reshape(B, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)  # (B, n*m) -> (B, 1, 1)
        C_normalized = C_batch - C_min  # (B, n, m) - (B, 1, 1) broadcast -> (B, n, m)

        if row_marginals is None:
            log_a = C_batch.new_full((B, n), -math.log(n))
        else:
            log_a = torch.log(row_marginals + 1e-10)

        if col_marginals is None:
            log_b = C_batch.new_full((B, m), -math.log(m))
        else:
            log_b = torch.log(col_marginals + 1e-10)

        K_log = -C_normalized / eps

        log_u = torch.zeros_like(log_a)  # (B, n)
        log_v = torch.zeros_like(log_b)  # (B, m)

        for iteration in range(max_iter):
            log_u_prev = log_u.clone()
            log_v_prev = log_v.clone()

            log_v = log_b - log_sum_exp_stabilized(K_log + log_u.unsqueeze(-1), dim=1)  # (B, n, m) -> (B, m)
            log_u = log_a - log_sum_exp_stabilized(K_log + log_v.unsqueeze(1), dim=2)  # (B, n, m) -> (B, n)

            check_every = 5 if not C_batch.is_cuda else 10
            if iteration % check_every == 0:
                err_u = torch.abs(log_u - log_u_prev).max()
                err_v = torch.abs(log_v - log_v_prev).max()
                if max(err_u.item(), err_v.item()) < tol:
                    break

        log_v = log_b - log_sum_exp_stabilized(K_log + log_u.unsqueeze(-1), dim=1)  # (B, n, m) -> (B, m)

        def _compute_plan() -> torch.Tensor:
            log_P = log_u.unsqueeze(-1) + K_log + log_v.unsqueeze(1)  # (B, n, 1)+(B, n, m)+(B, 1, m) -> (B, n, m)
            return torch.exp(log_P)  # (B, n, m)

        P = _compute_plan()

        if self.nan_checks:
            check_tensor_finite("sinkhorn_plan", P, enabled=True)
        if self.runtime_asserts:
            if row_marginals is None:
                expected_rows = C_batch.new_full((B, n), 1.0 / n)
            else:
                expected_rows = row_marginals
            if col_marginals is None:
                expected_cols = C_batch.new_full((B, m), 1.0 / m)
            else:
                expected_cols = col_marginals
            row_err = (P.sum(dim=2) - expected_rows).abs().max()  # (B, n) - (B, n) -> scalar
            col_err = (P.sum(dim=1) - expected_cols).abs().max()  # (B, m) - (B, m) -> scalar
            tol_mass = max(self.sinkhorn_mass_tolerance, self.sinkhorn_tol * 10)
            if max(row_err.item(), col_err.item()) > tol_mass:
                # Extra refinement for marginal satisfaction; bounded to avoid runaway compute.
                refine_iters = max(20, max_iter * 5)
                for _ in range(refine_iters):
                    log_v = log_b - log_sum_exp_stabilized(K_log + log_u.unsqueeze(-1), dim=1)  # (B, n, m) -> (B, m)
                    log_u = log_a - log_sum_exp_stabilized(K_log + log_v.unsqueeze(1), dim=2)  # (B, n, m) -> (B, n)
                log_v = log_b - log_sum_exp_stabilized(K_log + log_u.unsqueeze(-1), dim=1)  # (B, n, m) -> (B, m)
                P = _compute_plan()
                if self.nan_checks:
                    check_tensor_finite("sinkhorn_plan", P, enabled=True)
                row_err = (P.sum(dim=2) - expected_rows).abs().max()  # (B, n) - (B, n) -> scalar
                col_err = (P.sum(dim=1) - expected_cols).abs().max()  # (B, m) - (B, m) -> scalar
                if max(row_err.item(), col_err.item()) > tol_mass:
                    raise ValueError(
                        f"Sinkhorn mass conservation failed (row_err={row_err.item():.3e}, "
                        f"col_err={col_err.item():.3e}, tol={tol_mass:.3e})."
                    )

        return P





class MomentumTransport(nn.Module):
    """Transport with momentum for accelerated convergence"""

    def __init__(self, beta: float = 0.9, device: torch.device = torch.device('cpu')) -> None:
        super().__init__()
        self.beta = beta
        self.device = device
        self.register_buffer('velocity', None, persistent=False)
        self.velocity: Optional[torch.Tensor]
        self.velocity_shape: Optional[torch.Size] = None

    def reset_velocity(self) -> None:
        """Reset momentum velocity"""
        self.velocity = None
        self.velocity_shape = None

    def apply_transport(
        self, x: torch.Tensor, drift: torch.Tensor, alpha_bar_t: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Apply transport with momentum (handles shape changes)"""
        if self.velocity is not None and x.shape != self.velocity_shape:
            self.reset_velocity()

        if self.velocity is None:
            self.velocity = torch.zeros_like(x)
            self.velocity_shape = x.shape

        self.velocity = (self.beta * self.velocity.detach() + (1 - self.beta) * drift).detach()  # (B, ...) -> (B, ...)

        lookahead = x + self.beta * self.velocity  # (B, ...) -> (B, ...)

        transport_weight = self._compute_adaptive_weight(lookahead, alpha_bar_t)

        return x + transport_weight * self.velocity  # (B, ...) + (B, 1, ..., 1) broadcast -> (B, ...)

    def _compute_adaptive_weight(
        self, x: torch.Tensor, alpha_bar_t: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Compute adaptive weight for transport"""
        if x.dim() == 4 and x.shape[-1] > 1 and x.shape[-2] > 1:
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]  # (B, C, H, W-1)
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # (B, C, H-1, W)
            dx_center = dx[:, :, :-1, :]  # (B, C, H-1, W-1)
            dy_center = dy[:, :, :, :-1]  # (B, C, H-1, W-1)
            grad_mag = torch.sqrt(dx_center**2 + dy_center**2 + 1e-8)  # (B, C, H-1, W-1)
            smoothness = 1.0 / (1.0 + grad_mag.mean(dim=(1, 2, 3), keepdim=True))  # (B, 1, 1, 1)
        else:
            smoothness = x.new_ones(x.shape[0], 1)  # (B, 1)

        alpha_tensor = torch.as_tensor(
            alpha_bar_t,
            device=x.device,
            dtype=x.dtype if x.dtype.is_floating_point else torch.float32
        )  # scalar or (B,) -> tensor
        alpha_clamped = torch.clamp(alpha_tensor, 0.0, 1.0)  # same shape as alpha_tensor
        time_weight = 1.0 - alpha_clamped  # scalar or (B,)
        weight = smoothness * (0.5 + 0.5 * time_weight)  # (B,1,1,1) * (B,) broadcast

        for _ in range(x.dim() - len(weight.shape)):
            weight = weight.unsqueeze(-1)  # (B, 1) -> (B, 1, 1, 1)

        return weight





class HierarchicalBridge(nn.Module):
    """Multi-scale Schrodinger bridge decomposition"""

    def __init__(
        self, scales: List[float] = [1.0, 0.5, 0.25], device: torch.device = torch.device('cpu')
    ) -> None:
        super().__init__()
        self.scales = scales
        self.device = device
        self.bridge_cache: Dict[str, torch.Tensor] = {}

    @compile_function_fixed(dynamic=True, use_global_cache=True)
    def compute_multiscale_transport(
        self, x: torch.Tensor, drift: torch.Tensor, alpha_bar_t: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Compute transport at multiple scales"""
        s = 12.0
        alpha_tensor = torch.as_tensor(alpha_bar_t, device=x.device, dtype=x.dtype)  # scalar or (B,) -> tensor
        gate = torch.sigmoid(((1 - alpha_tensor).mean() - 0.5) * s).to(x.dtype)  # () -> ()

        transports = []
        weights = []

        for scale in self.scales:
            if scale < 0.5 and x.shape[-1] > 512:
                continue

            if scale < 1.0:
                x_scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)  # (B, C, H, W) -> (B, C, Hs, Ws)
                drift_scaled = F.interpolate(drift, scale_factor=scale, mode='bilinear', align_corners=False)  # (B, C, H, W) -> (B, C, Hs, Ws)
            else:
                x_scaled, drift_scaled = x, drift

            transport = self._compute_scale_transport(x_scaled, drift_scaled, scale)

            if scale < 1.0:
                transport = F.interpolate(transport, size=x.shape[-2:], mode='bilinear', align_corners=False)  # (B, C, Hs, Ws) -> (B, C, H, W)

            transports.append(transport)

            weight = self._compute_scale_weight(x, transport, scale, alpha_bar_t)
            weights.append(weight)

        if not transports:
            multiscale = x + drift
        else:
            # CRITICAL FIX: Explicit type handling to prevent confusion between list and tensor
            weights_tensor = torch.stack(weights, dim=0)  # (S,) -> (S,)
            weights_tensor = F.softmax(weights_tensor / 0.1, dim=0).to(x.dtype)  # (S,) -> (S,)
            # Unstack back to list for weighted sum (more explicit than zip iteration)
            weights_list = [weights_tensor[i] for i in range(len(transports))]
            weighted = torch.stack([w * tr for w, tr in zip(weights_list, transports)], dim=0)  # (S, B, C, H, W)
            multiscale = weighted.sum(dim=0)  # (S, B, C, H, W) -> (B, C, H, W)

        return (1 - gate) * (x + drift) + gate * multiscale  # scalar gate broadcast -> (B, C, H, W)

    def _compute_scale_transport(self, x: torch.Tensor, drift: torch.Tensor, scale: float) -> torch.Tensor:
        """Transport computation at specific scale"""
        if scale < 1.0:
            kernel_size = int(3 / scale)
            if kernel_size % 2 == 0:
                kernel_size += 1
            drift = F.avg_pool2d(drift, kernel_size, stride=1, padding=kernel_size//2)  # (B, C, H, W) -> (B, C, H, W)

        return x + drift

    def _compute_scale_weight(
        self, x: torch.Tensor, transport: torch.Tensor, scale: float, alpha_bar_t: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """Compute importance weight for each scale"""
        x_fft = torch.fft.rfft2(x)  # (B, C, H, W) -> (B, C, H, W//2+1)
        t_fft = torch.fft.rfft2(transport)  # (B, C, H, W) -> (B, C, H, W//2+1)

        try:
            freq_y = torch.fft.fftfreq(x.shape[-2], device=x.device)  # (H,)
        except TypeError:
            freq_y = torch.fft.fftfreq(x.shape[-2]).to(x.device)  # (H,)
        try:
            freq_x = torch.fft.rfftfreq(x.shape[-1], device=x.device)  # (W//2+1,)
        except TypeError:
            freq_x = torch.fft.rfftfreq(x.shape[-1]).to(x.device)  # (W//2+1,)

        freq_mag = torch.sqrt(freq_y[:, None]**2 + freq_x[None, :]**2)  # (H, 1) + (1, Wf) -> (H, Wf)

        if scale >= 1.0:
            freq_weight = freq_mag
        else:
            freq_weight = 1.0 - freq_mag

        energy = (torch.abs(t_fft - x_fft) * freq_weight[None, None, :, :]).mean()  # (B, C, H, Wf) * (1, 1, H, Wf) -> scalar

        alpha_tensor = torch.as_tensor(alpha_bar_t, device=x.device, dtype=x.dtype)  # scalar or (B,) -> tensor
        time_factor = 1.0 - alpha_tensor  # scalar or (B,)

        return energy * (1.0 + time_factor * scale)  # scalar * scalar -> scalar





class TransportModule(nn.Module):
    """Optimized transport operations module with sliced OT"""

    def __init__(self, config: FastSBOTConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.device = device

        # Pass generator to SlicedOptimalTransport
        self.sliced_ot = SlicedOptimalTransport(
            config.memory_limit_ot_mb,
            config._sinkhorn_iterations,
            config.sinkhorn_tolerance,
            config.sliced_ot_projection_fn,
            generator=config.generator,
            runtime_asserts=runtime_asserts_enabled(config),
            nan_checks=nan_checks_enabled(config),
            sinkhorn_mass_tolerance=config.sinkhorn_mass_tolerance,
        )



