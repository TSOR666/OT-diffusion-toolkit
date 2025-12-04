"""Transport-related utilities for SPOT."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from .constants import MAX_BLOCK_SIZE
from .logger import logger

__all__ = ["make_grid_patch_transport", "blockwise_soft_assignment"]


def make_grid_patch_transport(solver, patch_size: int = 64, stride: int | None = None) -> Callable:
    """Create a patch-based transport map factory bound to a solver instance.

    Args:
        solver: The solver instance
        patch_size: Size of each patch (will be adjusted if too large for image)
        stride: Stride between patches (default: patch_size // 2 for 50% overlap)

    Returns:
        A factory function that creates patch-based transport maps for BCHW image pairs
    """

    # Ensure stride is at least 1 to avoid division by zero
    stride = stride or max(1, patch_size // 2)

    unfold_cache: Dict[Tuple[int, int, int, int, torch.device, torch.dtype], torch.nn.Unfold] = {}
    fold_cache: Dict[Tuple[int, int, int, int, torch.device, torch.dtype], torch.nn.Fold] = {}
    # Store norm in fp32 to avoid precision loss when normalizing overlapping patches
    norm_cache: Dict[Tuple[int, int, int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def factory(x_img: torch.Tensor, y_img: torch.Tensor, eps: float):
        if x_img.dim() != 4 or y_img.dim() != 4:
            logger.debug("Patch transport expects BCHW tensors; using identity")
            return lambda z: z

        if x_img.shape != y_img.shape:
            logger.debug("Patch transport requires matching shapes; using identity")
            return lambda z: z

        B, C, H, W = x_img.shape
        if B != 1:
            logger.debug("Patch transport currently supports batch size 1; using identity")
            return lambda z: z

        if patch_size > min(H, W):
            logger.debug("Patch size %d exceeds spatial dimensions %dx%d; using identity", patch_size, H, W)
            return lambda z: z

        # Adjust patch size to ensure at least 2x2 patches for meaningful patch-based OT
        ps = patch_size
        st = stride

        # Calculate number of patches with current settings
        num_patches_h = ((H - ps) // st) + 1
        num_patches_w = ((W - ps) // st) + 1
        potential_num_patches = num_patches_h * num_patches_w

        # If only 1 patch, reduce patch_size to get at least 4 patches (2x2 grid)
        # This ensures patch-based OT provides meaningful spatial transport
        if potential_num_patches == 1 and min(H, W) > ps:
            # Aim for 2x2 patches with 50% overlap
            ps = max(min(H, W) // 2, 16)  # Minimum 16x16 patches
            st = max(1, ps // 2)  # Ensure stride >= 1
            logger.debug(
                "Adjusted patch_size from %d to %d to avoid single-patch case (image=%dx%d)",
                patch_size, ps, H, W
            )

        # Heuristic memory guard: bail out if unfolding would exceed configured tensor budget
        patch_dim_est = C * ps * ps
        num_patches_est = ((H - ps) // st + 1) * ((W - ps) // st + 1)
        total_elements_est = B * patch_dim_est * num_patches_est
        if total_elements_est > solver.config.max_tensor_size_elements:
            logger.debug(
                "Patch transport unfolded tensor would have %d elements (cap=%d); using identity",
                total_elements_est,
                solver.config.max_tensor_size_elements,
            )
            return lambda z: z

        cache_key = (C, H, W, ps, st, x_img.device, x_img.dtype)
        if cache_key not in unfold_cache:
            unfold_cache[cache_key] = torch.nn.Unfold(kernel_size=ps, stride=st)
            fold_cache[cache_key] = torch.nn.Fold(output_size=(H, W), kernel_size=ps, stride=st)

        unfold = unfold_cache[cache_key]
        fold = fold_cache[cache_key]

        patches_x = unfold(x_img)
        patches_y = unfold(y_img)
        _, patch_dim, num_patches = patches_x.shape

        logger.debug(
            "Patch transport: patch_size=%d, stride=%d, image=%dx%d, num_patches=%d",
            ps, st, H, W, num_patches
        )

        norm_key = (C, H, W, ps, st, x_img.device, torch.float32)
        if norm_key not in norm_cache:
            ones = torch.ones((B, patch_dim, num_patches), device=x_img.device, dtype=torch.float32)
            norm_cache[norm_key] = fold(ones).clamp_min(1e-6)
        norm = norm_cache[norm_key]

        x_vec = patches_x.transpose(1, 2).reshape(-1, patch_dim)
        y_vec = patches_y.transpose(1, 2).reshape(-1, patch_dim)

        log_u, log_v = solver.sinkhorn_kernel.sinkhorn_log_stabilized(
            x_vec, y_vec, eps, n_iter=solver.config.sinkhorn_iterations
        )

        if not (torch.isfinite(log_u).all() and torch.isfinite(log_v).all()):
            logger.debug("Patch Sinkhorn produced non-finite values; using identity transport")
            return lambda z: z

        # Allow single-patch case (N=M=1) for patch-based transport
        transport_map = solver._create_transport_map(x_vec, y_vec, log_u, log_v, eps, allow_single_point=True)

        def apply_transport(z: torch.Tensor) -> torch.Tensor:
            if z.dim() != 4:
                logger.debug("Patch transport expects BCHW tensors; returning input unchanged")
                return z

            if z.shape[1:] != (C, H, W):
                logger.debug("Patch transport received mismatched shape %s; returning input unchanged", tuple(z.shape))
                return z

            patches_z = unfold(z)
            z_vec = patches_z.transpose(1, 2).reshape(-1, patch_dim)
            transported_vec = transport_map(z_vec).reshape(B, num_patches, patch_dim).transpose(1, 2)
            transported = fold(transported_vec.to(norm.dtype)) / norm
            return transported.to(z.dtype)

        return apply_transport

    return factory


def blockwise_soft_assignment(
    zf: torch.Tensor,
    xf: torch.Tensor,
    eps: float,
    Ybar: torch.Tensor,
    block_size: int = MAX_BLOCK_SIZE,
    deterministic_cdist_cpu: bool = False,
) -> torch.Tensor:
    """Compute W @ Ybar in blocks to handle large NM."""

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

