import torch

from atlas.config import HighResModelConfig
from atlas.models import HighResLatentScoreModel


def test_score_model_forward_shapes() -> None:
    config = HighResModelConfig()
    model = HighResLatentScoreModel(config)

    x = torch.randn(2, config.in_channels, 64, 64)
    timesteps = torch.linspace(0.0, 1.0, x.size(0))

    out = model(x, timesteps)

    assert out.shape == x.shape
