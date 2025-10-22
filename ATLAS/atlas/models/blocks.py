from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ContextualAttention2D


class ResnetBlock2D(nn.Module):
    """Residual block with time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_out = self.time_proj(self.act(time_emb))
        h = h + time_out[:, :, None, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class DownsampleBlock(nn.Module):
    """Downsampling block with residual processing and optional contextual attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float,
        add_attention: bool,
        num_heads: int,
        downsample: bool,
        num_res_blocks: int,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            block_in = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResnetBlock2D(
                    block_in,
                    out_channels,
                    time_emb_dim,
                    dropout,
                )
            )
        self.attention = (
            ContextualAttention2D(
                out_channels, num_heads, context_dim=context_dim, dropout=dropout
            )
            if add_attention
            else None
        )
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            if downsample
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.res_blocks:
            x = block(x, time_emb)
        if self.attention is not None:
            x = self.attention(x, context=context, context_mask=context_mask)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class UpsampleBlock(nn.Module):
    """Upsampling block that merges skip features with optional context attention."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float,
        add_attention: bool,
        num_heads: int,
        upsample: bool,
        num_res_blocks: int,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.upsample = (
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
            if upsample
            else None
        )
        self.res_blocks = nn.ModuleList()
        in_ch = out_channels + skip_channels
        for i in range(num_res_blocks):
            block_in = in_ch if i == 0 else out_channels
            self.res_blocks.append(
                ResnetBlock2D(
                    block_in,
                    out_channels,
                    time_emb_dim,
                    dropout,
                )
            )
        self.attention = (
            ContextualAttention2D(
                out_channels, num_heads, context_dim=context_dim, dropout=dropout
            )
            if add_attention
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, time_emb)
        if self.attention is not None:
            x = self.attention(x, context=context, context_mask=context_mask)
        return x
