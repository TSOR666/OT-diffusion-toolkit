"""Tests covering dataset presets and utilities introduced for training scripts."""

from __future__ import annotations

import pytest
import torch

from atlas.config import presets
from atlas.config.training_config import DatasetConfig
from atlas.utils import create_dataloader


@pytest.mark.parametrize(
    "name,expected_resolution,batch_size",
    [
        ("experiment:lsun256", 256, 48),
        ("experiment:celeba1024", 1024, 12),
        ("experiment:ffhq128", 128, 64),
        ("experiment:imagenet64", 64, 256),
    ],
)
def test_experiment_presets_structure(name: str, expected_resolution: int, batch_size: int) -> None:
    bundle = presets.load_preset(name)
    dataset_cfg = bundle["dataset"]
    training_cfg = bundle["training"]
    inference_cfg = bundle["inference"]

    assert dataset_cfg.resolution == expected_resolution
    assert training_cfg.batch_size == batch_size
    assert inference_cfg.sampler_steps > 0
    assert "model" in bundle and "kernel" in bundle and "sampler" in bundle


def test_fake_dataset_dataloader_shapes(tmp_path) -> None:
    pytest.importorskip("torchvision")
    cfg = DatasetConfig(
        name="fake",
        root=str(tmp_path),
        resolution=64,
        channels=3,
        batch_size=4,
        fake_size=8,
        center_crop=False,
        random_flip=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    dataloader = create_dataloader(cfg, shuffle=False)
    batch = next(iter(dataloader))
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    assert images.shape == (cfg.batch_size, cfg.channels, cfg.resolution, cfg.resolution)
    assert torch.all(torch.isfinite(images))
