from typing import Optional

import torch
import torch.nn as nn


class ContextualAttention2D(nn.Module):
    """Multi-head attention operating on flattened spatial tokens with optional context."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.context_dim = context_dim
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.to_out = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout)

        if self.context_dim is not None and self.context_dim != channels:
            self.context_proj = nn.Linear(self.context_dim, channels, bias=False)
        else:
            self.context_proj = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        hidden = x.view(b, c, h * w).permute(0, 2, 1)
        hidden_norm = self.norm(hidden)

        if context is None:
            context_tokens = hidden_norm
        else:
            if context.dim() == 2:
                context_tokens = context.unsqueeze(1)
            elif context.dim() == 3:
                context_tokens = context
            else:
                raise ValueError("Context tensor must be 2D or 3D.")
            context_tokens = context_tokens.to(hidden_norm.dtype)

            if self.context_proj is not None:
                context_tokens = self.context_proj(context_tokens)

        q = self.to_q(hidden_norm)
        k = self.to_k(context_tokens)
        v = self.to_v(context_tokens)

        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if context_mask is not None:
            mask = context_mask.to(dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            # Mask out padded (False) positions; assume True=valid
            invalid_mask = ~mask
            attn_scores = attn_scores.masked_fill(invalid_mask, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.channels)
        out = self.to_out(out)
        out = out + hidden
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return out
