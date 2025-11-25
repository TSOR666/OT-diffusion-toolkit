from dataclasses import dataclass
from typing import Tuple
import warnings

# CLIP model embedding dimensions
CLIP_DIMS = {
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024,
    "ViT-B-32": 512,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-L-14-336": 768,
    "ViT-H-14": 1024,
}
VALID_CLIP_PRETRAINED = {"openai", "laion400m", "laion2b"}


@dataclass(frozen=True)
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
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}.")
        if self.enabled and not self.target_modules:
            raise ValueError(
                "target_modules cannot be empty when LoRA is enabled; specify module name patterns."
            )
        if self.enabled:
            scaling = self.alpha / self.rank
            if scaling > 4.0:
                warnings.warn(
                    f"LoRA scaling alpha/rank = {scaling:.2f} is unusually large and may cause instability.",
                    UserWarning,
                    stacklevel=2,
                )


@dataclass(frozen=True)
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
        if self.guidance_scale < 1.0:
            raise ValueError(f"guidance_scale must be >= 1.0, got {self.guidance_scale}.")
        if self.guidance_scale > 20.0:
            warnings.warn(
                f"Very high guidance_scale ({self.guidance_scale}) may cause artifacts.",
                UserWarning,
                stacklevel=2,
            )
        if self.cache_encodings and self.cache_max_entries <= 0:
            raise ValueError(
                f"cache_max_entries must be positive when caching is enabled, got {self.cache_max_entries}."
            )
        if self.use_clip and self.clip_model in CLIP_DIMS:
            expected_dim = CLIP_DIMS[self.clip_model]
            if self.context_dim != expected_dim:
                warnings.warn(
                    f"CLIP model '{self.clip_model}' typically uses context_dim={expected_dim}, "
                    f"but got {self.context_dim}. This may cause dimension mismatches.",
                    UserWarning,
                    stacklevel=2,
                )
        if self.use_clip and self.clip_pretrained not in VALID_CLIP_PRETRAINED:
            warnings.warn(
                f"Unknown clip_pretrained '{self.clip_pretrained}'. Valid options: {VALID_CLIP_PRETRAINED}.",
                UserWarning,
                stacklevel=2,
            )
