from dataclasses import dataclass
from typing import Tuple


@dataclass
class LoRAConfig:
    """Configuration options for low-rank adaptation adapters."""

    enabled: bool = False
    rank: int = 4
    alpha: int = 8
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "proj",
        "time_mlp",
        "context_proj",
    )


@dataclass
class ConditioningConfig:
    """Settings for textual or visual conditioning interfaces."""

    use_clip: bool = True
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    context_dim: int = 768
    max_length: int = 77
    guidance_scale: float = 7.5
    use_fp16: bool = False
    cache_encodings: bool = True
    cache_max_entries: int = 32
