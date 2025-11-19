from dataclasses import dataclass
from typing import Tuple


@dataclass
class LoRAConfig:
    """Configuration options for low-rank adaptation adapters."""

    enabled: bool = False
    rank: int = 4
    alpha: int = 4
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

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"LoRA dropout must be in [0, 1], got {self.dropout}.")
        scaling = self.alpha / self.rank
        if scaling > 2.0:
            raise ValueError(
                f"LoRA scaling alpha/rank = {scaling:.2f} is too large; "
                "set alpha closer to rank for stability."
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

    def __post_init__(self) -> None:
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
