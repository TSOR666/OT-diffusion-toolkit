"""Sanity checks for training loop behavior on small synthetic inputs."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from atlas.config.model_config import HighResModelConfig
from atlas.examples.training_pipeline import _save_checkpoint
from atlas.models.score_network import build_highres_score_model


def _tiny_config() -> HighResModelConfig:
    return HighResModelConfig(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(),
        cross_attention_levels=(),
        time_emb_dim=128,
        conditional=False,
    )


def test_single_batch_overfit() -> None:
    torch.manual_seed(0)
    model = build_highres_score_model(_tiny_config())
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(2, 1, 8, 8)
    t = torch.full((2,), 0.5)
    target = torch.randn_like(x)

    with torch.no_grad():
        initial_loss = F.mse_loss(model(x, t), target).item()

    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, t)
        loss = F.mse_loss(pred, target)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    with torch.no_grad():
        final_loss = F.mse_loss(model(x, t), target).item()

    assert final_loss < initial_loss * 0.5


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_highres_score_model(_tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(2, 1, 8, 8)
    t = torch.rand(2)
    target = torch.randn_like(x)

    loss = F.mse_loss(model(x, t), target)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()

    ema_model = copy.deepcopy(model)

    checkpoint_path = tmp_path / "training_sanity.pt"
    _save_checkpoint(
        checkpoint_path,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        step=1,
        epoch=0,
        bundle={"model": _tiny_config()},
    )

    checkpoint = torch.load(checkpoint_path)
    model_loaded = build_highres_score_model(_tiny_config())
    model_loaded.load_state_dict(checkpoint["model"])
    optimizer_loaded = torch.optim.AdamW(model_loaded.parameters(), lr=1e-3)
    optimizer_loaded.load_state_dict(checkpoint["optimizer"])

    with torch.no_grad():
        ref = model(x, t)
        out = model_loaded(x, t)

    assert torch.allclose(ref, out, atol=1e-6)
