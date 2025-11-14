from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import HighResModelConfig
from .attention import ContextualAttention2D
from .blocks import DownsampleBlock, ResnetBlock2D, UpsampleBlock, make_group_norm
from .embeddings import SinusoidalTimeEmbedding
from .lora import apply_lora_to_model


class HighResLatentScoreModel(nn.Module):
    """
    Hierarchical UNet-style score model tailored for high-resolution latent diffusion.

    The architecture combines sinusoidal time embeddings, residual bottlenecks, and
    attention at medium resolutions to balance fidelity with efficiency. It also
    supports classifier-free guidance through learned unconditional/conditional
    embeddings.
    """

    def __init__(self, config: HighResModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model_size = config.model_variant
        self.conditional = config.conditional
        self.conditioning_config = config.conditioning
        cond_cfg = config.conditioning
        self.use_context = (
            cond_cfg.use_clip if cond_cfg is not None else config.use_clip_conditioning
        )
        self.context_dim = (
            cond_cfg.context_dim if cond_cfg is not None else config.context_dim
        )
        self.default_guidance_scale = (
            cond_cfg.guidance_scale if cond_cfg is not None else 1.0
        )
        self.cross_attention_levels = set(config.cross_attention_levels)

        base_width = config.base_channels
        channel_mult = config.channel_mult

        self.time_embed = SinusoidalTimeEmbedding(config.time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_emb_dim * 4, config.time_emb_dim),
        )

        if self.conditional:
            if config.conditioning_dim > 0:
                self.condition_encoder = nn.Linear(
                    config.conditioning_dim, config.time_emb_dim
                )
            else:
                self.condition_encoder = None
            self.uncond_embedding = nn.Parameter(torch.zeros(config.time_emb_dim))
            self.cond_embedding = nn.Parameter(torch.zeros(config.time_emb_dim))
            nn.init.normal_(self.uncond_embedding, std=0.02)
            nn.init.normal_(self.cond_embedding, std=0.02)
        else:
            self.condition_encoder = None

        self.input_proj = nn.Conv2d(
            config.in_channels, base_width, kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_channels: List[int] = []

        in_channels = base_width
        num_levels = len(channel_mult)

        for level, mult in enumerate(channel_mult):
            out_channels = base_width * mult
            use_attention = level in config.attention_levels
            downsample = level < num_levels - 1
            context_dim = (
                self.context_dim
                if (self.use_context and level in self.cross_attention_levels)
                else None
            )
            block = DownsampleBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_emb_dim=config.time_emb_dim,
                dropout=config.dropout,
                add_attention=use_attention,
                num_heads=config.num_heads,
                downsample=downsample,
                num_res_blocks=config.num_res_blocks,
                context_dim=context_dim,
            )
            self.down_blocks.append(block)
            self.skip_channels.append(out_channels)
            in_channels = out_channels

        self.mid_block1 = ResnetBlock2D(
            in_channels, in_channels, config.time_emb_dim, config.dropout
        )
        mid_context_dim = self.context_dim if self.use_context else None
        self.mid_attention = ContextualAttention2D(
            in_channels, config.num_heads, context_dim=mid_context_dim, dropout=config.dropout
        )
        self.mid_block2 = ResnetBlock2D(
            in_channels, in_channels, config.time_emb_dim, config.dropout
        )

        for level in reversed(range(num_levels)):
            skip_ch = self.skip_channels[level]
            use_attention = level in config.attention_levels
            upsample = level > 0
            context_dim = (
                self.context_dim
                if (self.use_context and level in self.cross_attention_levels)
                else None
            )
            block = UpsampleBlock(
                in_channels=in_channels,
                skip_channels=skip_ch,
                out_channels=skip_ch,
                time_emb_dim=config.time_emb_dim,
                dropout=config.dropout,
                add_attention=use_attention,
                num_heads=config.num_heads,
                upsample=upsample,
                num_res_blocks=config.num_res_blocks,
                context_dim=context_dim,
            )
            self.up_blocks.append(block)
            in_channels = skip_ch

        self.output_norm = make_group_norm(in_channels)
        self.output_conv = nn.Conv2d(
            in_channels, config.out_channels, kernel_size=3, padding=1
        )

        apply_lora_to_model(self, config.lora)

    def _parse_condition(
        self,
        condition: Optional[Union[bool, torch.Tensor, Dict[str, Any]]],
        batch: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        context = None
        context_mask = None
        cond_emb = None

        if isinstance(condition, dict):
            context = condition.get("context")
            context_mask = condition.get("context_mask") or condition.get("mask")
            if context is not None and context.dim() == 2:
                context = context.unsqueeze(1)
            emb = condition.get("embedding")
            if (
                emb is None
                and condition.get("conditioning") is not None
                and self.condition_encoder is not None
            ):
                cond_vec = condition["conditioning"]
                if cond_vec.dim() == 1:
                    cond_vec = cond_vec.unsqueeze(0)
                emb = self.condition_encoder(cond_vec.to(device))
            if emb is not None:
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                cond_emb = emb.to(device)
        elif isinstance(condition, torch.Tensor):
            if condition.dim() == 3:
                context = condition
            elif (
                self.condition_encoder is not None
                and condition.dim() == 2
                and condition.size(1) == self.condition_encoder.in_features
            ):
                cond_emb = self.condition_encoder(condition.to(device))
            else:
                cond_emb = condition.to(device)
        elif isinstance(condition, bool):
            cond_emb = (
                self.cond_embedding.expand(batch, -1)
                if condition
                else self.uncond_embedding.expand(batch, -1)
            )
        elif condition is None and self.conditional:
            cond_emb = self.uncond_embedding.expand(batch, -1)

        if cond_emb is not None and cond_emb.dim() == 1:
            cond_emb = cond_emb.unsqueeze(0)
        if cond_emb is not None and cond_emb.size(0) != batch:
            cond_emb = cond_emb.expand(batch, -1)

        return context, context_mask, cond_emb

    def _apply_conditioning(
        self,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cond_emb is None:
            return time_emb
        return time_emb + cond_emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[Union[bool, torch.Tensor, Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        if timesteps.dim() == 1:
            timesteps = timesteps.to(x.device)
        elif timesteps.dim() > 1:
            timesteps = timesteps.squeeze()

        batch = x.size(0)
        context, context_mask, cond_emb = self._parse_condition(
            condition, batch, x.device
        )

        time_emb = self.time_embed(timesteps)
        time_emb = self.time_mlp(time_emb)
        time_emb = self._apply_conditioning(time_emb, cond_emb)

        h = self.input_proj(x)
        skips: List[torch.Tensor] = []

        for block in self.down_blocks:
            h, skip = block(
                h, time_emb, context=context, context_mask=context_mask
            )
            skips.append(skip)

        h = self.mid_block1(h, time_emb)
        h = self.mid_attention(h, context=context, context_mask=context_mask)
        h = self.mid_block2(h, time_emb)

        for block in self.up_blocks:
            skip = skips.pop()
            h = block(
                h, skip, time_emb, context=context, context_mask=context_mask
            )

        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_conv(h)


def build_highres_score_model(
    config: Optional[HighResModelConfig] = None,
) -> HighResLatentScoreModel:
    """Convenience helper for constructing the high-resolution score model."""

    config = config or HighResModelConfig()
    return HighResLatentScoreModel(config)
