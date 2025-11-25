import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time/positional embedding as used in diffusion models."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError(f"Embedding dimension must be at least 2, got {dim}.")
        if max_period <= 1.0:
            raise ValueError(f"max_period must be greater than 1.0, got {max_period}.")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            raise ValueError(
                f"Input timesteps must be 0D (scalar) or 1D (batch); got shape {tuple(t.shape)}."
            )
        if t.dim() == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        device = t.device
        if not t.is_floating_point():
            t = t.float()
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=t.dtype)
            * -(math.log(self.max_period) / half_dim)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
