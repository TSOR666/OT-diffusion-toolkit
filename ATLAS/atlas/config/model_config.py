from dataclasses import dataclass, field
from typing import Tuple
import warnings

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
    model_variant: str = "custom"
    conditional: bool = True
    # Optional vector conditioning dimension (e.g., class embeddings). Set to 0 to disable.
    conditioning_dim: int = 0
    cross_attention_levels: Tuple[int, ...] = (1, 2)
    conditioning_dim: int = 768
    # Legacy fields retained for backward compatibility; kept in sync with conditioning config.
    use_clip_conditioning: bool = True
    context_dim: int = 768
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.base_channels < 32:
            warnings.warn(
                f"Very small base_channels ({self.base_channels}) may limit capacity.",
                UserWarning,
                stacklevel=2,
            )
        if self.base_channels > 512:
            warnings.warn(
                f"Large base_channels ({self.base_channels}) may cause memory issues.",
                ResourceWarning,
                stacklevel=2,
            )
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}.")
        if self.time_emb_dim <= 0:
            raise ValueError("time_emb_dim must be positive.")
        if self.conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative.")
        if self.time_emb_dim < 128:
            warnings.warn(
                f"Small time_emb_dim ({self.time_emb_dim}) may limit temporal modeling",
                UserWarning,
                stacklevel=2,
            )
        if not self.model_variant:
            warnings.warn("model_variant is empty; defaulting to 'custom'.", UserWarning, stacklevel=2)
            self.model_variant = "custom"
        if self.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.latent_downsampling_factor <= 0:
            raise ValueError("latent_downsampling_factor must be positive.")
        if self.latent_downsampling_factor & (self.latent_downsampling_factor - 1) != 0:
            warnings.warn(
                f"latent_downsampling_factor {self.latent_downsampling_factor} is not a power of 2.",
                UserWarning,
                stacklevel=2,
            )
        if not self.channel_mult:
            raise ValueError("channel_mult cannot be empty.")
        if any(mult <= 0 for mult in self.channel_mult):
            raise ValueError("All channel_mult values must be positive.")
        if self.channel_mult[0] != 1:
            warnings.warn(
                f"channel_mult[0]={self.channel_mult[0]} (typically 1).",
                UserWarning,
                stacklevel=2,
            )
        max_channels = self.base_channels * max(self.channel_mult)
        if max_channels > 2048:
            warnings.warn(
                f"Maximum channels {max_channels} may cause memory issues.",
                ResourceWarning,
                stacklevel=2,
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
        for lvl in self.attention_levels:
            channels_at_level = self.base_channels * self.channel_mult[lvl]
            if channels_at_level % self.num_heads != 0:
                raise ValueError(
                    f"Channels at attention level {lvl} ({channels_at_level}) must be divisible by num_heads ({self.num_heads})."
                )
        if self.conditioning_dim < 0:
            raise ValueError("conditioning_dim must be non-negative.")
        if self.conditional:
            if self.conditioning.context_dim <= 0:
                raise ValueError("conditioning.context_dim must be positive when conditional=True.")
            if self.context_dim != self.conditioning.context_dim:
                warnings.warn(
                    f"context_dim ({self.context_dim}) differs from conditioning.context_dim ({self.conditioning.context_dim}); "
                    "these must match for CLIP conditioning.",
                    UserWarning,
                    stacklevel=2,
                )
            if self.use_clip_conditioning != self.conditioning.use_clip:
                warnings.warn(
                    "use_clip_conditioning differs from conditioning.use_clip; using conditioning.use_clip.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_clip_conditioning = self.conditioning.use_clip
            if self.conditioning_dim == 0:
                self.conditioning_dim = self.context_dim
        else:
            self.use_clip_conditioning = False
            self.conditioning_dim = 0

    @property
    def depth(self) -> int:
        return len(self.channel_mult)

    @property
    def max_channels(self) -> int:
        return self.base_channels * max(self.channel_mult)

    @property
    def total_downsampling(self) -> int:
        return int(2 ** self.depth)

    def __repr__(self) -> str:
        return (
            f"HighResModelConfig(depth={self.depth}, base_ch={self.base_channels}, "
            f"max_ch={self.max_channels}, heads={self.num_heads}, conditional={self.conditional})"
        )
