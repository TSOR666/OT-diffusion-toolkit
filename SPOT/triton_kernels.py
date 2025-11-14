"""Triton-accelerated GPU kernels for SPOT solver.

This module provides optimized GPU kernels for:
- Sinkhorn iterations with fused operations
- Fast cost matrix computation
- Optimized softmax and logsumexp operations
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

__all__ = [
    "TRITON_AVAILABLE",
    "triton_sinkhorn_update",
    "fused_cost_softmax",
]

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


NUMERICAL_STABILITY_EPSILON = 1e-12


if TRITON_AVAILABLE:
    @triton.jit
    def _sinkhorn_logsumexp_kernel(
        x_ptr,
        y_ptr,
        log_other_ptr,
        log_target_ptr,
        out_ptr,
        x_sq_ptr,
        y_sq_ptr,
        n_rows,
        n_cols,
        dim,
        eps,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """Fused Sinkhorn logsumexp kernel with cost computation."""
        row_pid = tl.program_id(axis=0)
        row_start = row_pid * BLOCK_ROWS
        row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = row_offsets < n_rows

        log_max = tl.full((BLOCK_ROWS,), float("-inf"), dtype=tl.float32)
        sum_exp = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

        x_sq_vals = tl.load(x_sq_ptr + row_offsets, mask=row_mask, other=0.0)
        log_target_vals = tl.load(log_target_ptr + row_offsets, mask=row_mask, other=0.0)

        for col_start in range(0, n_cols, BLOCK_COLS):
            col_offsets = col_start + tl.arange(0, BLOCK_COLS)
            col_mask = col_offsets < n_cols

            y_sq_vals = tl.load(y_sq_ptr + col_offsets, mask=col_mask, other=0.0)
            log_other_vals = tl.load(
                log_other_ptr + col_offsets, mask=col_mask, other=float("-inf")
            )

            # Compute dot product for cost matrix
            acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)

            for dim_start in range(0, dim, BLOCK_DIM):
                dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
                dim_mask = dim_offsets < dim

                x_ptrs = x_ptr + row_offsets[:, None] * dim + dim_offsets[None, :]
                x_tile = tl.load(
                    x_ptrs,
                    mask=row_mask[:, None] & dim_mask[None, :],
                    other=0.0,
                )
                x_tile = x_tile.to(tl.float32)

                y_ptrs = y_ptr + col_offsets[None, :] * dim + dim_offsets[:, None]
                y_tile = tl.load(
                    y_ptrs,
                    mask=dim_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                y_tile = y_tile.to(tl.float32)

                acc += tl.dot(x_tile, y_tile)

            # Compute cost and kernel in one go
            cost_tile = x_sq_vals[:, None] + y_sq_vals[None, :] - 2.0 * acc
            log_kernel = -cost_tile / eps + log_other_vals[None, :]

            # Numerically stable logsumexp reduction
            tile_max = tl.max(log_kernel, axis=1)
            new_log_max = tl.maximum(log_max, tile_max)

            rescale = tl.exp(log_max - new_log_max)
            sum_exp *= rescale
            sum_exp += tl.sum(tl.exp(log_kernel - new_log_max[:, None]), axis=1)
            log_max = new_log_max

        log_sum = log_max + tl.log(sum_exp + NUMERICAL_STABILITY_EPSILON)
        output = log_target_vals - log_sum
        tl.store(out_ptr + row_offsets, output, mask=row_mask)

    @triton.jit
    def _fused_cost_softmax_kernel(
        x_ptr,
        y_ptr,
        x_sq_ptr,
        y_sq_ptr,
        out_ptr,
        n_rows,
        n_cols,
        dim,
        eps,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """Fused cost computation and softmax kernel."""
        row_pid = tl.program_id(axis=0)
        col_pid = tl.program_id(axis=1)

        row_start = row_pid * BLOCK_ROWS
        col_start = col_pid * BLOCK_COLS

        row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
        col_offsets = col_start + tl.arange(0, BLOCK_COLS)

        row_mask = row_offsets < n_rows
        col_mask = col_offsets < n_cols

        x_sq_vals = tl.load(x_sq_ptr + row_offsets, mask=row_mask, other=0.0)
        y_sq_vals = tl.load(y_sq_ptr + col_offsets, mask=col_mask, other=0.0)

        # Compute dot product
        acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)

        for dim_start in range(0, dim, BLOCK_DIM):
            dim_offsets = dim_start + tl.arange(0, BLOCK_DIM)
            dim_mask = dim_offsets < dim

            x_ptrs = x_ptr + row_offsets[:, None] * dim + dim_offsets[None, :]
            x_tile = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            x_tile = x_tile.to(tl.float32)

            y_ptrs = y_ptr + col_offsets[None, :] * dim + dim_offsets[:, None]
            y_tile = tl.load(
                y_ptrs,
                mask=dim_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            y_tile = y_tile.to(tl.float32)

            acc += tl.dot(x_tile, y_tile)

        # Compute cost and softmax
        cost_tile = x_sq_vals[:, None] + y_sq_vals[None, :] - 2.0 * acc
        S = -cost_tile / eps

        # Numerically stable softmax across columns
        S_max = tl.max(S, axis=1, keep_dims=True)
        S_shifted = S - S_max
        exp_S = tl.exp(S_shifted)
        sum_exp = tl.sum(exp_S, axis=1, keep_dims=True) + NUMERICAL_STABILITY_EPSILON
        W = exp_S / sum_exp

        # Store output
        out_ptrs = out_ptr + row_offsets[:, None] * n_cols + col_offsets[None, :]
        tl.store(out_ptrs, W, mask=row_mask[:, None] & col_mask[None, :])


def triton_sinkhorn_update(
    x: torch.Tensor,
    y: torch.Tensor,
    log_other: torch.Tensor,
    log_target: torch.Tensor,
    eps: float,
    tile_size: int = 64,
    *,
    precomputed_sq: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Compute a single Sinkhorn update using optimized Triton kernel.

    Args:
        x: Source points [N, D]
        y: Target points [M, D]
        log_other: Log dual variable for other marginal [M]
        log_target: Log marginal for target [N]
        eps: Entropic regularization parameter
        tile_size: Tile size for blocking
        precomputed_sq: Optional precomputed squared norms (x_sq, y_sq)

    Returns:
        Updated log dual variable [N]
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for accelerated Sinkhorn update")

    if x.device.type != "cuda":
        raise ValueError("Triton Sinkhorn update requires CUDA tensors")

    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {x.shape[1]} vs {y.shape[1]}")

    # Ensure contiguous memory layout
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    # Convert to float32 for numerical stability
    x = x.float()
    y = y.float()
    log_other = log_other.float().contiguous()
    log_target = log_target.float().contiguous()

    # Precompute squared norms if not provided
    if precomputed_sq is None:
        x_sq = (x * x).sum(dim=1)
        y_sq = (y * y).sum(dim=1)
    else:
        x_sq, y_sq = precomputed_sq

    out = torch.empty_like(log_target, dtype=torch.float32)

    n_rows = x.shape[0]
    n_cols = y.shape[0]
    dim = x.shape[1]

    # Adaptive block sizes based on problem size
    BLOCK_ROWS = min(128, max(32, tile_size))
    BLOCK_COLS = min(128, max(32, tile_size))
    BLOCK_DIM = 64 if dim >= 64 else 32

    grid = (triton.cdiv(n_rows, BLOCK_ROWS),)

    eps_value = torch.tensor(float(eps), device=x.device, dtype=torch.float32)

    _sinkhorn_logsumexp_kernel[grid](
        x,
        y,
        log_other,
        log_target,
        out,
        x_sq,
        y_sq,
        n_rows,
        n_cols,
        dim,
        eps_value,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        BLOCK_DIM=BLOCK_DIM,
    )

    return out


def fused_cost_softmax(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    tile_size: int = 64,
) -> torch.Tensor:
    """Compute cost matrix and softmax in a single fused kernel.

    Args:
        x: Source points [N, D]
        y: Target points [M, D]
        eps: Entropic regularization parameter
        tile_size: Tile size for blocking

    Returns:
        Softmax weights [N, M]
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused cost-softmax")

    if x.device.type != "cuda":
        raise ValueError("Triton kernels require CUDA tensors")

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    x = x.float()
    y = y.float()

    x_sq = (x * x).sum(dim=1)
    y_sq = (y * y).sum(dim=1)

    n_rows = x.shape[0]
    n_cols = y.shape[0]
    dim = x.shape[1]

    out = torch.empty((n_rows, n_cols), device=x.device, dtype=torch.float32)

    BLOCK_ROWS = min(64, max(16, tile_size))
    BLOCK_COLS = min(64, max(16, tile_size))
    BLOCK_DIM = 64 if dim >= 64 else 32

    grid = (
        triton.cdiv(n_rows, BLOCK_ROWS),
        triton.cdiv(n_cols, BLOCK_COLS),
    )

    eps_value = torch.tensor(float(eps), device=x.device, dtype=torch.float32)

    _fused_cost_softmax_kernel[grid](
        x,
        y,
        x_sq,
        y_sq,
        out,
        n_rows,
        n_cols,
        dim,
        eps_value,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_COLS=BLOCK_COLS,
        BLOCK_DIM=BLOCK_DIM,
    )

    return out
