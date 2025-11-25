import math
from typing import List, Tuple
import warnings
import logging

import torch.nn as nn

from ..config.conditioning_config import LoRAConfig


class LoRALinear(nn.Module):
    """Wrapper that adds a LoRA branch to an existing linear layer."""

    def __init__(self, layer: nn.Linear, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base = layer
        for param in self.base.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.lora_down = nn.Linear(layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_hidden = self.lora_down(x)
        if self.dropout is not None:
            lora_hidden = self.dropout(lora_hidden)
        lora_out = self.lora_up(lora_hidden) * self.scale
        return base_out + lora_out

    def merge_lora_weights(self) -> None:
        """Merge LoRA weights into the base layer for inference."""
        with torch.no_grad():
            delta = (self.lora_up.weight @ self.lora_down.weight) * self.scale
            self.base.weight.data += delta
            self.lora_down.weight.zero_()
            self.lora_up.weight.zero_()


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a nested module specified by its dot-separated path."""
    parts = module_name.split(".")
    parent = root
    try:
        for name in parts[:-1]:
            parent = getattr(parent, name)
        if not hasattr(parent, parts[-1]):
            raise AttributeError(f"Module '{parts[-1]}' not found in {type(parent).__name__}")
        setattr(parent, parts[-1], new_module)
    except AttributeError as exc:
        raise ValueError(f"Failed to replace module '{module_name}': {exc}") from exc


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> None:
    """Inject LoRA adapters into selected linear submodules by name pattern."""

    if not config.enabled:
        return

    target_patterns = set(config.target_modules)
    replacements: List[Tuple[str, nn.Linear]] = []

    if not target_patterns:
        warnings.warn("LoRA is enabled but target_modules is empty; no adapters applied.", RuntimeWarning)
        return

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(pattern in name for pattern in target_patterns):
            replacements.append((name, module))

    for name, module in replacements:
        wrapped = LoRALinear(module, config.rank, config.alpha, config.dropout)
        _replace_module(model, name, wrapped)

    if not replacements:
        logging.warning("LoRA enabled but no modules matched patterns: %s", target_patterns)
