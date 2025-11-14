"""Pytest-based verification of ATLAS trainability and sampling.

The original repository shipped a standalone script that exercised the
trainability pipeline and called ``sys.exit`` on completion. Pytest treats a
module-level ``sys.exit`` as an error during collection, which prevented the
repository test suite from running.  This module keeps the same coverage but
exposes each check as an idempotent unit test so that CI systems and local
contributors can rely on ``pytest``.
"""

from __future__ import annotations

import copy
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F

from atlas.config import HighResModelConfig, KernelConfig, SamplerConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import (
    AdvancedHierarchicalDiffusionSampler,
    SchroedingerBridgeSolver,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    """Use CUDA when available; otherwise default to CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def model_config() -> HighResModelConfig:
    """Baseline configuration shared across the tests."""

    return HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mult=(1, 2),
        num_res_blocks=2,
        attention_levels=(1,),
        conditional=False,
        use_clip_conditioning=False,
    )


@pytest.fixture
def make_model(model_config: HighResModelConfig, device: torch.device):
    """Factory that yields a freshly initialised model on the requested device."""

    def _factory() -> HighResLatentScoreModel:
        torch.manual_seed(0)
        model = HighResLatentScoreModel(model_config)
        return model.to(device)

    return _factory


def _noisy_batch(
    device: torch.device,
    *,
    batch_size: int = 2,
    spatial_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a noisy batch compatible with DDPM-style updates."""

    x = torch.randn(batch_size, 3, spatial_size, spatial_size, device=device)
    t = torch.rand(batch_size, device=device)
    noise = torch.randn_like(x)

    alpha = karras_noise_schedule(t).view(-1, 1, 1, 1)
    sigma = torch.sqrt(torch.clamp((1.0 - alpha) / alpha, min=1e-8))
    noisy_x = alpha.sqrt() * x + sigma * noise
    return noisy_x, noise, t


def test_model_forward(make_model, device):
    model = make_model()
    model.eval()

    with torch.no_grad():
        x = torch.randn(2, 3, 64, 64, device=device)
        t = torch.rand(2, device=device)
        output = model(x, t)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_gradient_flow(make_model, device):
    model = make_model()
    model.train()

    noisy_x, noise, t = _noisy_batch(device)
    loss = F.mse_loss(model(noisy_x, t), noise)
    loss.backward()

    params_with_grad = 0
    params_without_grad = 0

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            params_without_grad += 1
            continue
        params_with_grad += 1
        assert torch.isfinite(param.grad).all(), "Encountered non-finite gradients"

    assert params_with_grad > 0, "Expected at least one parameter with gradients"
    assert (
        params_without_grad <= 4
    ), "Unexpected number of parameters without gradients"


def test_optimizer_step(make_model, device):
    model = make_model()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4
    )

    noisy_x, noise, t = _noisy_batch(device)
    loss = F.mse_loss(model(noisy_x, t), noise)
    loss.backward()

    before = [param.detach().clone() for param in model.parameters()]
    optimizer.step()

    changed = False
    for param, previous in zip(model.parameters(), before):
        if not torch.allclose(param, previous):
            changed = True
            break

    assert changed, "Optimizer step did not update any parameters"


def test_training_loop(make_model, device):
    model = make_model()
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-4
    )

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        noisy_x, noise, t = _noisy_batch(device)
        loss = F.mse_loss(model(noisy_x, t), noise)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    loss_tensor = torch.tensor(losses)
    assert torch.isfinite(loss_tensor).all()
    assert losses[-1] <= losses[0] + 0.2


@pytest.fixture
def solver_components(make_model, device):
    model = make_model()
    model.eval()

    kernel_config = KernelConfig(
        epsilon=0.1,
        solver_type="direct",
        kernel_type="gaussian",
    )
    sampler_config = SamplerConfig(
        sb_iterations=1,
        hierarchical_sampling=False,
        verbose_logging=False,
    )

    return model, kernel_config, sampler_config


def test_solver_sampling(device, solver_components):
    model, kernel_config, sampler_config = solver_components

    solver = SchroedingerBridgeSolver(
        score_model=model,
        noise_schedule=karras_noise_schedule,
        device=device,
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    with torch.no_grad():
        samples = solver.sample(shape=(2, 3, 32, 32), timesteps=[1.0, 0.5, 0.1], verbose=False)

    assert samples.shape == (2, 3, 32, 32)
    assert torch.isfinite(samples).all()


def test_hierarchical_sampler(device, solver_components):
    model, kernel_config, sampler_config = solver_components

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=model,
        noise_schedule=karras_noise_schedule,
        device=device,
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    with torch.no_grad():
        samples = sampler.sample(
            shape=(2, 3, 32, 32), timesteps=[1.0, 0.5, 0.1], verbose=False
        )

    assert samples.shape == (2, 3, 32, 32)
    assert torch.isfinite(samples).all()


def test_ema_update(make_model, device):
    model = make_model()
    ema_model = copy.deepcopy(model)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    noisy_x, noise, t = _noisy_batch(device)
    loss = F.mse_loss(model(noisy_x, t), noise)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(0.9999).add_(param, alpha=1.0 - 0.9999)

    differences = [torch.max(torch.abs(e - p)).item() for e, p in zip(ema_model.parameters(), model.parameters())]
    assert any(diff > 0.0 for diff in differences)


@pytest.mark.parametrize("solver_type", ["direct", "fft", "rff"])
def test_kernel_variants(make_model, device, solver_type):
    model = make_model()
    model.eval()

    kernel_config = KernelConfig(
        epsilon=0.1,
        solver_type=solver_type,
        kernel_type="gaussian",
        rff_features=64,
    )
    sampler_config = SamplerConfig(
        sb_iterations=1,
        hierarchical_sampling=False,
        verbose_logging=False,
    )

    solver = SchroedingerBridgeSolver(
        score_model=model,
        noise_schedule=karras_noise_schedule,
        device=device,
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    with torch.no_grad():
        samples = solver.sample(shape=(1, 3, 16, 16), timesteps=[1.0, 0.5, 0.1], verbose=False)

    assert samples.shape == (1, 3, 16, 16)
    assert torch.isfinite(samples).all()


def test_mixed_precision(device, make_model):
    if device.type != "cuda":
        pytest.skip("Mixed precision test requires CUDA")

    model = make_model()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    from torch.cuda.amp import GradScaler, autocast

    scaler = GradScaler()

    optimizer.zero_grad()
    noisy_x, noise, t = _noisy_batch(device)
    with autocast():
        loss = F.mse_loss(model(noisy_x, t), noise)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert torch.isfinite(loss.detach()).all()

