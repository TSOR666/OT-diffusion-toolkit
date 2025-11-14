"""FFT-based optimal transport utilities."""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class FFTOptimalTransport:
    """FFT-based Optimal Transport implementation."""

    def __init__(
        self,
        epsilon: float = 0.01,
        max_iter: int = 50,
        tol: float = 1e-5,
        kernel_type: str = "gaussian",
        device: torch.device | None = None,
        multiscale: bool = True,
        scale_levels: int = 3,
        fallback_block_size: int = 2048,
    ) -> None:
        # Input validation
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol must be positive, got {tol}")
        if scale_levels < 1:
            raise ValueError(f"scale_levels must be at least 1, got {scale_levels}")
        if fallback_block_size < 1:
            raise ValueError(f"fallback_block_size must be at least 1, got {fallback_block_size}")
        if kernel_type not in ["gaussian", "laplacian"]:
            raise ValueError(
                f"kernel_type must be one of ['gaussian', 'laplacian'], "
                f"got '{kernel_type}'"
            )

        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_type = kernel_type
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multiscale = multiscale
        self.scale_levels = scale_levels
        self.fallback_block_size = fallback_block_size

    def _select_interpolation_mode(self, ndim: int) -> Tuple[str, Optional[bool]]:
        if ndim <= 0:
            raise ValueError(f"Expected positive dimensionality for interpolation, got {ndim}")

        if ndim == 1:
            return "linear", False
        if ndim == 2:
            return "bilinear", False
        if ndim == 3:
            return "trilinear", False

        # PyTorch does not support higher-order linear interpolation directly; fall back to nearest.
        return "nearest", None

    def _resize_density(self, tensor: torch.Tensor, shape: List[int]) -> torch.Tensor:
        mode, align_corners = self._select_interpolation_mode(len(shape))
        interpolate_kwargs = {"mode": mode}
        if align_corners is not None:
            interpolate_kwargs["align_corners"] = align_corners

        resized = F.interpolate(
            tensor.unsqueeze(0).unsqueeze(0), size=shape, **interpolate_kwargs
        )
        return resized.squeeze(0).squeeze(0)

    def _is_grid_structured(self, x: torch.Tensor) -> Tuple[bool, Optional[List[int]]]:
        if x.dim() > 2:
            return True, list(x.shape[1:])

        if x.dim() == 2:
            ndim = x.size(1)
            if ndim <= 3:
                for dim in range(ndim):
                    unique_coords = torch.unique(x[:, dim])
                    if len(unique_coords) > 100:
                        return False, None
                grid_shape = [len(torch.unique(x[:, dim])) for dim in range(ndim)]
                if int(np.prod(grid_shape)) == x.size(0):
                    return True, grid_shape
        return False, None

    def _reshape_to_grid(self, x: torch.Tensor, grid_shape: List[int]) -> torch.Tensor:
        batch_size = x.size(0)
        channels = x.size(1) if x.dim() > 1 else 1

        if x.dim() == 1:
            x = x.unsqueeze(1)

        if int(np.prod(grid_shape)) != batch_size:
            raise ValueError(
                f"Product of grid shape {grid_shape} does not match batch size {batch_size}"
            )

        return x.transpose(0, 1).reshape(channels, *grid_shape)

    def _apply_kernel_fft(self, u: torch.Tensor, kernel_fft: torch.Tensor) -> torch.Tensor:
        u_fft = torch.fft.rfftn(u)
        result_fft = u_fft * kernel_fft
        return torch.fft.irfftn(result_fft, s=u.shape)

    def _compute_kernel_fft(self, shape: List[int], epsilon: float) -> torch.Tensor:
        coords = []
        for idx, size in enumerate(shape):
            coord = torch.arange(size, device=self.device) - size // 2
            coord = coord.reshape([1 if j != idx else size for j in range(len(shape))])
            coords.append(coord)
        dist_sq = sum(coord ** 2 for coord in coords)

        if self.kernel_type == "gaussian":
            kernel = torch.exp(-dist_sq / (epsilon + 1e-30))
        elif self.kernel_type == "laplacian":
            kernel = torch.exp(-torch.sqrt(dist_sq + 1e-10) / epsilon)
        else:
            raise ValueError(
                f"Unsupported kernel type: '{self.kernel_type}'. "
                f"Supported types are: 'gaussian', 'laplacian'"
            )

        kernel = torch.fft.ifftshift(kernel)
        return torch.fft.rfftn(kernel)

    def _sinkhorn_fft(
        self,
        mu: torch.Tensor,
        nu: torch.Tensor,
        epsilon: float,
        kernel_fft: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = mu / mu.sum()
        nu = nu / nu.sum()

        if kernel_fft is None:
            kernel_fft = self._compute_kernel_fft(list(mu.shape), epsilon)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        for _ in range(self.max_iter):
            u_prev = u.clone()
            Ku = self._apply_kernel_fft(torch.exp(u), kernel_fft)
            v = torch.log(nu + 1e-15) - torch.log(Ku + 1e-15)
            Kv = self._apply_kernel_fft(torch.exp(v), kernel_fft)
            u = torch.log(mu + 1e-15) - torch.log(Kv + 1e-15)
            if torch.max(torch.abs(u - u_prev)) < self.tol:
                break

        a_vec = torch.exp(u)
        b_vec = torch.exp(v)
        Kv = self._apply_kernel_fft(b_vec, kernel_fft)
        objective = (u * mu).sum() + (v * nu).sum() - epsilon * (a_vec * Kv).sum()
        return objective, u, v

    def _multiscale_sinkhorn_fft(
        self, mu: torch.Tensor, nu: torch.Tensor, epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.multiscale:
            return self._sinkhorn_fft(mu, nu, epsilon)

        shape = mu.shape
        current_shape = [max(4, size // (2 ** self.scale_levels)) for size in shape]

        mu_coarse = self._resize_density(mu, current_shape)
        nu_coarse = self._resize_density(nu, current_shape)

        _, u_coarse, v_coarse = self._sinkhorn_fft(mu_coarse, nu_coarse, epsilon)

        for level in range(self.scale_levels, -1, -1):
            current_shape = [max(4, size // (2 ** level)) for size in shape]
            if level == self.scale_levels:
                continue

            u_upsampled = self._resize_density(u_coarse, current_shape)
            v_upsampled = self._resize_density(v_coarse, current_shape)

            mu_current = self._resize_density(mu, current_shape)
            nu_current = self._resize_density(nu, current_shape)

            kernel_fft = self._compute_kernel_fft(current_shape, epsilon)
            _, u_current, v_current = self._sinkhorn_fft(mu_current, nu_current, epsilon, kernel_fft)
            u_coarse, v_coarse = u_current, v_current

        objective, u_final, v_final = self._sinkhorn_fft(mu, nu, epsilon)
        return objective, u_final, v_final

    def _compute_gradient_on_grid(
        self, u: torch.Tensor, grid_spacing: Optional[List[float]] = None
    ) -> List[torch.Tensor]:
        grid_shape = u.shape
        ndim = len(grid_shape)
        if grid_spacing is None:
            grid_spacing = [1.0] * ndim

        gradients: List[torch.Tensor] = []
        for dim in range(ndim):
            grad_d = torch.zeros_like(u)
            interior = [slice(1, -1) if idx == dim else slice(None) for idx in range(ndim)]
            forward = [slice(2, None) if idx == dim else slice(None) for idx in range(ndim)]
            backward = [slice(0, -2) if idx == dim else slice(None) for idx in range(ndim)]
            grad_d[tuple(interior)] = (
                u[tuple(forward)] - u[tuple(backward)]
            ) / (2 * grid_spacing[dim])

            left = [slice(0, 1) if idx == dim else slice(None) for idx in range(ndim)]
            left_forward = [slice(1, 2) if idx == dim else slice(None) for idx in range(ndim)]
            grad_d[tuple(left)] = (
                u[tuple(left_forward)] - u[tuple(left)]
            ) / grid_spacing[dim]

            right = [slice(-1, None) if idx == dim else slice(None) for idx in range(ndim)]
            right_backward = [slice(-2, -1) if idx == dim else slice(None) for idx in range(ndim)]
            grad_d[tuple(right)] = (
                u[tuple(right)] - u[tuple(right_backward)]
            ) / grid_spacing[dim]

            gradients.append(grad_d)
        return gradients

    def optimal_transport(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights_x: Optional[torch.Tensor] = None,
        weights_y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_grid_x, grid_shape_x = self._is_grid_structured(x)
        is_grid_y, grid_shape_y = self._is_grid_structured(y)

        if is_grid_x and is_grid_y and grid_shape_x == grid_shape_y:
            grid_shape = grid_shape_x
            if x.dim() > 2:
                mu = x.sum(0) if x.dim() > len(grid_shape) else x
            else:
                values = (
                    weights_x
                    if weights_x is not None
                    else torch.ones(x.size(0), device=x.device)
                )
                mu = self._reshape_to_grid(values.unsqueeze(1), grid_shape).squeeze(0)

            if y.dim() > 2:
                nu = y.sum(0) if y.dim() > len(grid_shape) else y
            else:
                values = (
                    weights_y
                    if weights_y is not None
                    else torch.ones(y.size(0), device=y.device)
                )
                nu = self._reshape_to_grid(values.unsqueeze(1), grid_shape).squeeze(0)

            objective, u_pot, v_pot = self._multiscale_sinkhorn_fft(mu, nu, self.epsilon)
            return objective, u_pot, v_pot

        warnings.warn(
            "Data is not grid-structured; using chunked Sinkhorn fallback",
            RuntimeWarning,
        )
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        if y.dim() > 2:
            y = y.reshape(y.size(0), -1)

        weights_x = (
            weights_x
            if weights_x is not None
            else torch.ones(x.size(0), device=self.device) / max(1, x.size(0))
        )
        weights_y = (
            weights_y
            if weights_y is not None
            else torch.ones(y.size(0), device=self.device) / max(1, y.size(0))
        )
        weights_x = (weights_x / weights_x.sum()).to(self.device)
        weights_y = (weights_y / weights_y.sum()).to(self.device)

        cost, u_dual, v_dual = self._sinkhorn_blockwise_fallback(
            x.to(self.device),
            y.to(self.device),
            weights_x,
            weights_y,
            self.epsilon,
            self.max_iter,
            self.tol,
            self.fallback_block_size,
        )
        return cost, u_dual, v_dual

    def _sinkhorn_blockwise_fallback(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
        epsilon: float,
        max_iter: int,
        tol: float,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        dtype = torch.float32

        n_size, m_size = x.size(0), y.size(0)
        log_u = torch.zeros(n_size, device=device, dtype=dtype)
        log_v = torch.zeros(m_size, device=device, dtype=dtype)

        log_a = torch.log(weights_x.to(device=device, dtype=dtype) + 1e-10)
        log_b = torch.log(weights_y.to(device=device, dtype=dtype) + 1e-10)

        row_block = max(1, block_size // max(1, int(math.sqrt(m_size) + 1)))
        col_block = block_size

        for _ in range(max_iter):
            log_u_prev = log_u.clone()
            log_u = self._sinkhorn_block_update(
                x, y, log_v, epsilon, log_a, row_block, col_block, update_rows=True
            )
            log_v = self._sinkhorn_block_update(
                y, x, log_u, epsilon, log_b, col_block, row_block, update_rows=False
            )
            if torch.max(torch.abs(log_u - log_u_prev)) < tol:
                break

        total_cost = torch.tensor(0.0, device=device, dtype=dtype)
        for i_start in range(0, n_size, row_block):
            i_end = min(i_start + row_block, n_size)
            x_chunk = x[i_start:i_end]
            log_u_chunk = log_u[i_start:i_end]
            for j_start in range(0, m_size, col_block):
                j_end = min(j_start + col_block, m_size)
                y_chunk = y[j_start:j_end]
                log_v_chunk = log_v[j_start:j_end]

                cost_chunk = torch.cdist(x_chunk, y_chunk, p=2).pow(2)
                log_plan = (
                    log_u_chunk[:, None]
                    + log_v_chunk[None, :]
                    - cost_chunk / epsilon
                )
                plan = torch.exp(log_plan)
                total_cost += torch.sum(plan * cost_chunk)

        return total_cost.to(x.dtype), log_u.to(x.dtype), log_v.to(x.dtype)

    def _sinkhorn_block_update(
        self,
        x_primary: torch.Tensor,
        x_secondary: torch.Tensor,
        log_dual_secondary: torch.Tensor,
        epsilon: float,
        log_marginal: torch.Tensor,
        primary_block: int,
        secondary_block: int,
        update_rows: bool,
    ) -> torch.Tensor:
        device = x_primary.device
        dtype = log_dual_secondary.dtype
        size_primary = x_primary.size(0)
        updated = torch.empty(size_primary, device=device, dtype=dtype)

        for i_start in range(0, size_primary, primary_block):
            i_end = min(i_start + primary_block, size_primary)
            primary_chunk = x_primary[i_start:i_end]
            log_marginal_chunk = log_marginal[i_start:i_end]

            accum = torch.full(
                (i_end - i_start,),
                -float("inf"),
                device=device,
                dtype=dtype,
            )

            for j_start in range(0, x_secondary.size(0), secondary_block):
                j_end = min(j_start + secondary_block, x_secondary.size(0))
                secondary_chunk = x_secondary[j_start:j_end]
                log_dual_chunk = log_dual_secondary[j_start:j_end]

                cost_chunk = torch.cdist(primary_chunk, secondary_chunk, p=2).pow(2)
                log_kernel = -cost_chunk / epsilon + log_dual_chunk[None, :]
                if update_rows:
                    accum = torch.logaddexp(accum, torch.logsumexp(log_kernel, dim=1))
                else:
                    accum = torch.logaddexp(accum, torch.logsumexp(log_kernel, dim=0))

            updated[i_start:i_end] = log_marginal_chunk - accum

        return updated
