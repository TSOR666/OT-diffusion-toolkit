"""Sinkhorn kernel implementations for SPOT."""
from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np
import torch

from .constants import EPSILON_CLAMP
from .logger import logger

# Try to import Triton acceleration
try:
    from .triton_kernels import TRITON_AVAILABLE, triton_sinkhorn_update
except ImportError:
    TRITON_AVAILABLE = False
    triton_sinkhorn_update = None

__all__ = ["OptimizedSinkhornKernel"]


class OptimizedSinkhornKernel:
    """Production-grade Sinkhorn implementation."""

    def __init__(self, device: torch.device, dtype: torch.dtype, config):
        self.device = device
        self.dtype = dtype
        self.config = config
        self.backends = ["torch_native"]
        self.last_backend_used = "torch_native"

        # Add Triton backend if available
        self.use_triton = TRITON_AVAILABLE and device.type == "cuda" and getattr(config, "use_triton", True)
        if self.use_triton:
            self.backends.insert(0, "triton")
            logger.debug("Triton backend available for GPU acceleration")

        try:
            import ot

            self.ot_sinkhorn = ot.sinkhorn
            self.backends.insert(0, "pot")
            self.use_pot = config.use_pot_library
        except ImportError:  # pragma: no cover - optional dependency
            self.use_pot = False

        # Deterministic mode must not use POT (non-deterministic CPU/GPU kernels)
        if getattr(config, "deterministic", False):
            self.use_pot = False
            if "pot" in self.backends:
                self.backends.remove("pot")

    def sinkhorn_log_stabilized(
        self, x: torch.Tensor, y: torch.Tensor, eps: float, n_iter: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Sinkhorn dual variables in log-space for numerical stability."""

        N, M = x.size(0), y.size(0)
        required_cost_elems = N * M

        max_elements = self.config.max_tensor_size_elements
        max_dense_elems = self.config.max_dense_matrix_elements

        if (
            x.numel() > max_elements
            or y.numel() > max_elements
            or required_cost_elems > max_dense_elems
        ):
            logger.debug(
                "Sinkhorn input too large for dense solve (features=%s/%s, cost=%s > %s); entering blockwise streaming mode",
                x.numel(),
                y.numel(),
                required_cost_elems,
                max_dense_elems,
            )
            return self._sinkhorn_blockwise(x, y, eps, n_iter)

        x = x.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        # Try Triton backend for large problems where tiling provides benefits
        # Triton is designed for medium-large problems (>= 1M elements) where GPU tiling helps
        use_triton = self.use_triton and not self.config.deterministic
        n_elements = x.size(0) * y.size(0)

        # Use Triton for problems >= 1M elements (kernel launch overhead worth it)
        # Upper bound enforced by max_tensor_size_elements check above
        if use_triton and n_elements >= 1_000_000:
            try:
                self.last_backend_used = "triton"
                return self._sinkhorn_triton(x, y, eps, n_iter)
            except Exception as exc:
                logger.debug("Triton backend failed: %s, falling back to native PyTorch", exc)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Triton traceback:", exc_info=True)

        # Try POT backend for small-medium problems (< 1M elements)
        # POT has lower overhead but doesn't scale as well to large problems
        use_pot = self.use_pot and not self.config.deterministic

        if use_pot and n_elements < 1_000_000:
            try:
                self.last_backend_used = "pot"
                return self._sinkhorn_pot(x, y, eps, n_iter)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.debug("POT backend failed: %s, falling back to native PyTorch", exc)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("POT traceback:", exc_info=True)

        self.last_backend_used = "torch_native"
        return self._sinkhorn_native(x, y, eps, n_iter)

    def _sinkhorn_native(self, x: torch.Tensor, y: torch.Tensor, eps: float, n_iter: int):
        """Native PyTorch Sinkhorn implementation with memory-efficient computation.

        IMPORTANT: This method should only be called for problems that fit in memory.
        The caller (sinkhorn_log_stabilized) enforces this via max_tensor_size_elements.
        """
        N, M = x.size(0), y.size(0)

        # Handle degenerate case where N=0 or M=0
        if N == 0 or M == 0:
            logger.warning("Sinkhorn called with empty tensor (N=%d, M=%d), returning NaN duals", N, M)
            log_u = torch.full((N,), float("nan"), device=self.device, dtype=torch.float32)
            log_v = torch.full((M,), float("nan"), device=self.device, dtype=torch.float32)
            return log_u, log_v

        # Safety check: Verify we won't OOM on the cost matrix
        # C is N×M fp32 matrix = 4 * N * M bytes
        matrix_bytes = 4 * N * M
        matrix_mb = matrix_bytes / (1024 ** 2)
        if matrix_mb > 500:  # Warn if cost matrix > 500MB
            logger.warning(
                f"Large cost matrix ({N}×{M} = {matrix_mb:.0f}MB). "
                f"Consider lowering max_tensor_size_elements to force blockwise mode earlier."
            )

        if self.config.deterministic_cdist_cpu and self.device.type == "cuda":
            with torch.amp.autocast(device_type="cpu", enabled=False):
                x_cpu = x.float().cpu()
                y_cpu = y.float().cpu()
                C = torch.cdist(x_cpu, y_cpu, p=2).pow(2).to(self.device)
        else:
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                C = torch.cdist(x.float(), y.float(), p=2).pow(2)

        log_a = torch.full((N,), -math.log(N), device=self.device, dtype=torch.float32)
        log_b = torch.full((M,), -math.log(M), device=self.device, dtype=torch.float32)

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            # MEMORY OPTIMIZATION: We still materialize log_K here, but at least
            # we've gated entry to this function via max_tensor_size_elements check.
            # For truly large problems, the blockwise path should be taken.
            log_K = -C / float(eps)

            lu = torch.zeros(N, device=self.device, dtype=torch.float32)
            lv = torch.zeros(M, device=self.device, dtype=torch.float32)

            for _ in range(n_iter):
                lu = log_a - torch.logsumexp(log_K + lv[None, :], dim=1)
                lv = log_b - torch.logsumexp(log_K.T + lu[None, :], dim=1)

        if not (torch.isfinite(lu).all() and torch.isfinite(lv).all()):
            logger.warning(
                "Non-finite Sinkhorn duals (eps=%s), using identity fallback - may affect quality", eps
            )
            log_u = torch.full((N,), float("nan"), device=self.device, dtype=torch.float32)
            log_v = torch.full((M,), float("nan"), device=self.device, dtype=torch.float32)
            return log_u, log_v

        return lu, lv

    def _sinkhorn_blockwise(self, x: torch.Tensor, y: torch.Tensor, eps: float, n_iter: int):
        x = x.to(device=self.device, dtype=torch.float32)
        y = y.to(device=self.device, dtype=torch.float32)

        N, M = x.size(0), y.size(0)
        log_a = torch.full((N,), -math.log(max(N, 1)), device=self.device, dtype=torch.float32)
        log_b = torch.full((M,), -math.log(max(M, 1)), device=self.device, dtype=torch.float32)

        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)

        threshold = max(self.config.blockwise_threshold, 1)
        row_block = max(128, min(N, int(math.sqrt(threshold))))
        col_block = max(128, min(M, threshold // max(row_block, 1)))

        for _ in range(n_iter):
            log_u = self._sinkhorn_block_update(
                x, y, log_v, eps, log_a, row_block, col_block, update_rows=True
            )
            log_v = self._sinkhorn_block_update(
                y, x, log_u, eps, log_b, col_block, row_block, update_rows=False
            )

        return log_u, log_v

    def _sinkhorn_block_update(
        self,
        x_primary: torch.Tensor,
        x_secondary: torch.Tensor,
        log_dual_secondary: torch.Tensor,
        eps: float,
        log_marginal: torch.Tensor,
        primary_block: int,
        secondary_block: int,
        update_rows: bool = True,
    ) -> torch.Tensor:
        device = x_primary.device
        dtype = torch.float32
        size_primary = x_primary.size(0)

        updated = torch.empty(size_primary, device=device, dtype=dtype)

        for i_start in range(0, size_primary, primary_block):
            i_end = min(i_start + primary_block, size_primary)
            primary_chunk = x_primary[i_start:i_end]
            marginal_chunk = log_marginal[i_start:i_end]

            accumulator = torch.full((i_end - i_start,), -float("inf"), device=device, dtype=dtype)

            for j_start in range(0, x_secondary.size(0), secondary_block):
                j_end = min(j_start + secondary_block, x_secondary.size(0))
                secondary_chunk = x_secondary[j_start:j_end]
                dual_chunk = log_dual_secondary[j_start:j_end]

                C_chunk = torch.cdist(primary_chunk, secondary_chunk, p=2).pow(2)
                log_kernel = -C_chunk / eps + dual_chunk[None, :]

                accumulator = torch.logaddexp(accumulator, torch.logsumexp(log_kernel, dim=1))

            updated[i_start:i_end] = marginal_chunk - accumulator

        return updated

    def _sinkhorn_triton(self, x: torch.Tensor, y: torch.Tensor, eps: float, n_iter: int):
        """Triton-accelerated Sinkhorn with log-stabilization."""
        N, M = x.size(0), y.size(0)

        # Handle degenerate case where N=0 or M=0
        if N == 0 or M == 0:
            logger.warning("Triton Sinkhorn called with empty tensor (N=%d, M=%d), returning NaN duals", N, M)
            log_u = torch.full((N,), float("nan"), device=self.device, dtype=torch.float32)
            log_v = torch.full((M,), float("nan"), device=self.device, dtype=torch.float32)
            return log_u, log_v

        # Initialize log marginals
        log_a = torch.full((N,), -math.log(N), device=self.device, dtype=torch.float32)
        log_b = torch.full((M,), -math.log(M), device=self.device, dtype=torch.float32)

        # Initialize dual variables in log space
        log_u = torch.zeros(N, device=self.device, dtype=torch.float32)
        log_v = torch.zeros(M, device=self.device, dtype=torch.float32)

        # Precompute squared norms for efficiency
        x_sq = (x.float() * x.float()).sum(dim=1)
        y_sq = (y.float() * y.float()).sum(dim=1)

        # Sinkhorn iterations using Triton
        for _ in range(n_iter):
            # Update log_u
            log_u = triton_sinkhorn_update(
                x, y, log_v, log_a, eps,
                tile_size=64,
                precomputed_sq=(x_sq, y_sq)
            )

            # Update log_v (swap x and y)
            log_v = triton_sinkhorn_update(
                y, x, log_u, log_b, eps,
                tile_size=64,
                precomputed_sq=(y_sq, x_sq)
            )

        # Check for numerical issues
        if not (torch.isfinite(log_u).all() and torch.isfinite(log_v).all()):
            logger.warning("Non-finite Sinkhorn duals with Triton (eps=%s), using identity fallback", eps)
            log_u = torch.full((N,), float("nan"), device=self.device, dtype=torch.float32)
            log_v = torch.full((M,), float("nan"), device=self.device, dtype=torch.float32)
            return log_u, log_v

        return log_u, log_v

    def _sinkhorn_pot(self, x: torch.Tensor, y: torch.Tensor, eps: float, n_iter: int):
        import ot  # Local import to ensure optional dependency is available when needed

        N, M = x.size(0), y.size(0)

        # Handle degenerate case where N=0 or M=0
        if N == 0 or M == 0:
            logger.warning("POT Sinkhorn called with empty tensor (N=%d, M=%d), returning NaN duals", N, M)
            log_u = torch.full((N,), float("nan"), device=self.device, dtype=torch.float32)
            log_v = torch.full((M,), float("nan"), device=self.device, dtype=torch.float32)
            return log_u, log_v

        x_np = x.detach().cpu().float().numpy()
        y_np = y.detach().cpu().float().numpy()

        a = np.ones(N, dtype=np.float32) / N
        b = np.ones(M, dtype=np.float32) / M

        C = np.sum((x_np[:, None, :] - y_np[None, :, :]) ** 2, axis=2)

        P, log = ot.sinkhorn(a, b, C, eps, numItermax=n_iter, log=True)

        log_u_np = log.get("logu")
        log_v_np = log.get("logv")

        if log_u_np is None:
            u_np = log.get("u")
            if u_np is None:
                u_np = np.ones_like(a)
            log_u_np = np.log(np.clip(u_np, np.finfo(np.float32).tiny, None))
        if log_v_np is None:
            v_np = log.get("v")
            if v_np is None:
                v_np = np.ones_like(b)
            log_v_np = np.log(np.clip(v_np, np.finfo(np.float32).tiny, None))

        log_u = torch.from_numpy(log_u_np.astype(np.float32)).to(self.device)
        log_v = torch.from_numpy(log_v_np.astype(np.float32)).to(self.device)

        return log_u, log_v

    def cleanup_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
