import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding used in diffusion models."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device
        t = t.float()
        if half_dim == 0:
            raise ValueError("Embedding dimension must be at least 2 for sinusoidal embeddings.")
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=t.dtype)
            * -(math.log(self.max_period) / half_dim)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb[:, : self.dim]
