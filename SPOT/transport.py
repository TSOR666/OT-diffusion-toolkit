"""Transport-related utilities for SPOT."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from .constants import MAX_BLOCK_SIZE
from .logger import logger

__all__ = ["make_grid_patch_transport", "blockwise_soft_assignment"]


def make_grid_patch_transport(solver, patch_size: int = 64, stride: int | None = None) -> Callable:
    """Create a patch-based transport map factory bound to a solver instance."""

    stride = stride or patch_size // 2

    unfold_cache: Dict[Tuple[int, int, int], torch.nn.Unfold] = {}
    fold_cache: Dict[Tuple[int, int, int], torch.nn.Fold] = {}
    norm_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
    y_patches_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def factory(x_flat: torch.Tensor, y_flat: torch.Tensor, eps: float):
        if x_flat.dim() != 2 or y_flat.dim() != 2:
            raise ValueError("Patch transport expects flattened inputs")

        B, N = x_flat.shape
        C = solver.config.patch_size
        if B != 1:
            logger.debug("Patch transport only supports batch=1; falling back to identity")
            return lambda z: z

        C = solver.config.patch_size
        H = W = int((N / C) ** 0.5)
        if H * W * C != N:
            logger.debug("Patch transport input size mismatch, using identity")
            return lambda z: z

        ps = patch_size
        st = stride

        cache_key = (ps, st, N)

        def transport(z: torch.Tensor):
            if z.dim() != 4:
                logger.debug("Patch transport expects BCHW tensors; falling back to identity")
                return z

            B, C_channels, H_img, W_img = z.shape
            if (H_img, W_img) != (H, W) or C_channels != C:
                logger.debug("Patch transport shape mismatch, using identity")
                return z

            if cache_key not in unfold_cache:
                unfold_cache[cache_key] = torch.nn.Unfold(kernel_size=ps, stride=st)
                fold_cache[cache_key] = torch.nn.Fold(output_size=(H, W), kernel_size=ps, stride=st)

            unfold = unfold_cache[cache_key]
            fold = fold_cache[cache_key]

            patches = unfold(z)
            B_, D_, L = patches.shape

            if cache_key not in norm_cache:
                ones = torch.ones((B, D_, L), device=z.device, dtype=z.dtype)
                norm_cache[cache_key] = fold(ones).clamp_min(1e-6)
            norm = norm_cache[cache_key]

            if cache_key not in y_patches_cache:
                y_img = y_flat.view(1, C, H, W).to(z.device, z.dtype)
                y_p = unfold(y_img).transpose(1, 2)
                y_patches_cache[cache_key] = y_p.reshape(-1, y_p.size(-1))

            y_pf = y_patches_cache[cache_key]

            x_p = patches.transpose(1, 2)
            x_pf = x_p.reshape(-1, x_p.size(-1))

            u, v = solver.sinkhorn_kernel.sinkhorn_log_stabilized(
                x_pf, y_pf, eps, n_iter=solver.config.sinkhorn_iterations
            )

            if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
                logger.debug("Patch OT produced non-finite values, using identity")
                return z

            tm = solver._create_transport_map(x_pf, y_pf, u, v, eps)

            z_p_next = tm(x_pf).reshape(B, L, D_).transpose(1, 2)
            z_next = fold(z_p_next) / norm

            return z_next

        return transport

    return factory


def blockwise_soft_assignment(
    zf: torch.Tensor,
    xf: torch.Tensor,
    eps: float,
    Ybar: torch.Tensor,
    block_size: int = MAX_BLOCK_SIZE,
    deterministic_cdist_cpu: bool = False,
) -> torch.Tensor:
    """Compute W @ Ybar in blocks to handle large N×M."""

    n_z = zf.size(0)
    result = torch.zeros_like(zf)

    xf_cpu = xf.float().cpu() if (deterministic_cdist_cpu and zf.is_cuda) else None

    for i in range(0, n_z, block_size):
        end_i = min(i + block_size, n_z)
        zf_block = zf[i:end_i]

        with torch.no_grad():
            device_type = "cuda" if zf.is_cuda else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                if xf_cpu is not None:
                    zf_block_cpu = zf_block.float().cpu()
                    C_block = torch.cdist(zf_block_cpu, xf_cpu, p=2).pow(2).to(zf.device)
                else:
                    C_block = torch.cdist(zf_block.float(), xf.float(), p=2).pow(2)
                S_block = -C_block / eps
                S_block = S_block - S_block.max(dim=1, keepdim=True)[0]
                W_block = torch.softmax(S_block, dim=1)

        result[i:end_i] = (W_block @ Ybar).to(zf.dtype)

    return result
