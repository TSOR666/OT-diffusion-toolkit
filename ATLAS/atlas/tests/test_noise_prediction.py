from typing import Any, Optional

import torch
import torch.nn as nn
import pytest

import atlas
from atlas.schedules.noise import karras_noise_schedule
from atlas.utils import NoisePredictionAdapter


class _ShapeMismatchNoiseModel(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        channels = max(1, x.shape[1] - 1)
        return torch.zeros(x.shape[0], channels, *x.shape[2:], device=x.device, dtype=x.dtype)


class _GuidedNoiseModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,  # type: ignore[override]
        conditioning: Optional[Any] = None,  # type: ignore[override]
        **kwargs: Any,
    ) -> torch.Tensor:
        if condition == "cond":
            return torch.full_like(x, 1.0)
        if condition == "uncond":
            return torch.full_like(x, 0.0)
        if conditioning is not None:
            return torch.full_like(x, 3.0)
        return torch.full_like(x, 2.0)


def test_noise_adapter_rejects_shape_mismatch() -> None:
    adapter = NoisePredictionAdapter(_ShapeMismatchNoiseModel())
    x = torch.randn(2, 3, 4)

    with pytest.raises(ValueError):
        adapter.predict_noise(x, 0.5)


def test_noise_adapter_guidance_combines_cond_uncond() -> None:
    adapter = NoisePredictionAdapter(_GuidedNoiseModel())
    x = torch.zeros(1, 2, 4)
    conditioning = {"cond": "cond", "uncond": "uncond", "guidance_scale": 2.0}

    noise = adapter.predict_noise(x, 0.1, conditioning)
    assert torch.allclose(noise, torch.full_like(x, 2.0))


def test_karras_noise_schedule_is_valid() -> None:
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        alpha = karras_noise_schedule(t)
        assert 0.0 < alpha <= 1.0
    tensor_alpha = karras_noise_schedule(torch.tensor(0.5))
    assert torch.isfinite(tensor_alpha).item()
    alpha_value = float(tensor_alpha)
    assert 0.0 < alpha_value <= 1.0


def test_atlas_exports_version_and_utils() -> None:
    assert isinstance(atlas.__version__, str)
    assert "SchroedingerBridgeSolver" in atlas.__all__
    from atlas import utils as atlas_utils

    assert hasattr(atlas_utils, "NoisePredictionAdapter")
