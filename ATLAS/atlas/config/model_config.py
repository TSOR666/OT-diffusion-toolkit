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
    conditioning_dim: int = 768
    model_variant: str = "large"
    use_clip_conditioning: bool = True
    context_dim: int = 768
    cross_attention_levels: Tuple[int, ...] = (1, 2)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}.")
        if self.conditional and self.conditioning_dim <= 0:
            raise ValueError(
                "conditioning_dim must be positive when conditional=True."
            )
        depth = len(self.channel_mult)
        max_level = depth - 1
        invalid_attn = [lvl for lvl in self.attention_levels if lvl > max_level or lvl < 0]
        if invalid_attn:
            raise ValueError(
                f"attention_levels {invalid_attn} exceed available depth {max_level}."
            )
        invalid_cross = [
            lvl for lvl in self.cross_attention_levels if lvl > max_level or lvl < 0
        ]
        if invalid_cross:
            raise ValueError(
                f"cross_attention_levels {invalid_cross} exceed available depth {max_level}."
            )
