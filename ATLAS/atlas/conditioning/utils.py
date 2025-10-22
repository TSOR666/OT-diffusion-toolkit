from typing import Any, Dict, Optional

import torch


def safe_expand_tensor(tensor: torch.Tensor, target_batch: int, base_batch: int) -> torch.Tensor:
    """Broadcast a tensor to the desired batch size while keeping semantics intact."""

    if tensor.dim() == 0 or tensor.size(0) == target_batch:
        return tensor
    if tensor.size(0) == 1:
        return tensor.expand(target_batch, *tensor.shape[1:])
    if tensor.size(0) == base_batch and target_batch % base_batch == 0:
        repeat = target_batch // base_batch
        return tensor.repeat_interleave(repeat, dim=0)
    if target_batch % tensor.size(0) == 0:
        repeat = target_batch // tensor.size(0)
        return tensor.repeat_interleave(repeat, dim=0)
    raise ValueError(
        f"Cannot expand conditioning batch of size {tensor.size(0)} to {target_batch}."
    )


def expand_condition_dict(
    cond_dict: Optional[Dict[str, Any]],
    target_batch: int,
    base_batch: int,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Ensure all tensor entries in a conditioning payload match the target batch size."""

    if cond_dict is None:
        return None
    expanded: Dict[str, Any] = {}
    for key, value in cond_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.dim() > 0:
                value = safe_expand_tensor(value, target_batch, base_batch)
            expanded[key] = value
        else:
            expanded[key] = value
    return expanded
