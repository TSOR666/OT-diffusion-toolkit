from dataclasses import dataclass, field
from typing import Tuple

from .conditioning_config import ConditioningConfig, LoRAConfig


@dataclass
class HighResModelConfig:
    """Configuration for the high-resolution latent score model.

    Defaults target latent diffusion on 1024x1024 images while keeping the network
    feasible on modern GPUs.
    """

    in_channels: int = 4
    out_channels: int = 4
    latent_downsampling_factor: int = 8  # VAE downsampling (8 for SD VAE, 4 for some others)
    base_channels: int = 192
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_levels: Tuple[int, ...] = (1, 2)
    num_heads: int = 4
    dropout: float = 0.0
    time_emb_dim: int = 768
    conditional: bool = True
    conditioning_dim: int = 0
    model_variant: str = "large"
    use_clip_conditioning: bool = True
    context_dim: int = 768
    cross_attention_levels: Tuple[int, ...] = (1, 2)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
