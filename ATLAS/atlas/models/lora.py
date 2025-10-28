import math
from typing import List, Tuple

import torch.nn as nn

from ..config.conditioning_config import LoRAConfig


class LoRALinear(nn.Module):
    """Wrapper that adds a LoRA branch to an existing linear layer."""

    def __init__(self, layer: nn.Linear, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)

        self.lora_down = nn.Linear(layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_up(self.dropout(self.lora_down(x))) * self.scale
        return base_out + lora_out


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """Replace a nested module specified by its dot-separated path."""

    parts = module_name.split(".")
    parent = root
    for name in parts[:-1]:
        parent = getattr(parent, name)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> None:
    """Inject LoRA adapters into selected linear submodules by name pattern."""

    if not config.enabled:
        return

    target_patterns = set(config.target_modules)
    replacements: List[Tuple[str, nn.Linear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not target_patterns:
            replacements.append((name, module))
            continue
        if any(pattern in name for pattern in target_patterns):
            replacements.append((name, module))

    for name, module in replacements:
        wrapped = LoRALinear(module, config.rank, config.alpha, config.dropout)
        _replace_module(model, name, wrapped)
