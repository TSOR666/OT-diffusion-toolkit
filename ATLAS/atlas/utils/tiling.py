"""Spatial tiling helpers for high-resolution sampling."""

from __future__ import annotations

from typing import Dict, Tuple, Optional
import warnings

import torch
import torch.nn as nn


def _build_window(height: int, width: int, device: torch.device, dtype: torch.dtype, mode: str) -> torch.Tensor:
    if mode == "none":
        return torch.ones((1, 1, height, width), device=device, dtype=dtype)

    if mode == "linear":
        y = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        win_y = torch.minimum(y, 1 - y) * 2.0
        win_x = torch.minimum(x, 1 - x) * 2.0
        window = torch.outer(win_y, win_x)
    else:  # default hann
        win_y = torch.hann_window(height, periodic=False, dtype=dtype, device=device)
        win_x = torch.hann_window(width, periodic=False, dtype=dtype, device=device)
        window = torch.outer(win_y, win_x)

    window = window / window.max().clamp_(min=1e-6)
    return window.view(1, 1, height, width)


class TiledModelWrapper(nn.Module):
    """Wrap a model to evaluate via overlapping spatial tiles."""

    def __init__(
        self,
        model: nn.Module,
        tile_size: int,
        *,
        stride: int | None = None,
        overlap: float = 0.125,
        blending: str = "hann",
    ) -> None:
        super().__init__()
        self.model = model
        self.tile_size = tile_size
        self.explicit_stride = stride
        self.overlap = float(overlap)
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError(f"overlap must be in [0,1); got {self.overlap}")
        if self.overlap > 0.5:
            warnings.warn(
                f"Large overlap ({self.overlap:.2f}) may cause excessive memory usage.",
                ResourceWarning,
                stacklevel=2,
            )
        self.blending = blending
        self._window_cache: Dict[Tuple[int, int, str, Optional[int], torch.dtype], torch.Tensor] = {}
        self.predicts_score = getattr(model, "predicts_score", True)
        self.predicts_noise = getattr(model, "predicts_noise", False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.tile_size is None:
            return self.model(x, t)

        b, c, h, w = x.shape
        if self.tile_size >= min(h, w):
            return self.model(x, t)

        tile_h = min(self.tile_size, h)
        tile_w = min(self.tile_size, w)
        stride = self.explicit_stride or int(max(1, self.tile_size * (1.0 - self.overlap)))

        out_acc = torch.zeros((b, c, h, w), device=x.device, dtype=torch.float32)
        weight = torch.zeros((1, 1, h, w), device=x.device, dtype=torch.float32)

        for y_start in range(0, h, stride):
            y_end = min(y_start + tile_h, h)
            y_start = max(0, y_end - tile_h)
            for x_start in range(0, w, stride):
                x_end = min(x_start + tile_w, w)
                x_start = max(0, x_end - tile_w)

                tiles = x[:, :, y_start:y_end, x_start:x_end]
                scores = self.model(tiles, t)
                scores_f32 = scores.to(dtype=torch.float32)

                window = self._get_window(y_end - y_start, x_end - x_start, x.device, torch.float32)
                out_acc[:, :, y_start:y_end, x_start:x_end] += scores_f32 * window
                weight[:, :, y_start:y_end, x_start:x_end] += window

        weight = weight.clamp_(min=1e-6)
        out = (out_acc / weight).to(dtype=x.dtype)
        return out

    def _get_window(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (
            height,
            width,
            device.type,
            device.index if device.type == "cuda" else None,
            dtype,
        )
        window = self._window_cache.get(key)
        if window is None:
            window = _build_window(height, width, device, dtype, self.blending)
            self._window_cache[key] = window
        return window
