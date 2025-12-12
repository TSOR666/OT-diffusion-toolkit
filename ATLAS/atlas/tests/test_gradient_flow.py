"""Gradient flow validation tests."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import cast, overload

import pytest
import torch

from atlas.config.kernel_config import KernelConfig
from atlas.config.model_config import HighResModelConfig
from atlas.config.sampler_config import SamplerConfig
from atlas.models.score_network import build_highres_score_model
from atlas.solvers.schrodinger_bridge import SchroedingerBridgeSolver


@overload
def _linear_schedule(t: float) -> float: ...


@overload
def _linear_schedule(t: torch.Tensor) -> torch.Tensor: ...


def _linear_schedule(t: float | torch.Tensor) -> float | torch.Tensor:
    """Simple alpha schedule in (0, 1] for testing."""
    if isinstance(t, torch.Tensor):
        return torch.ones_like(t) * 0.9
    return 0.9


def _backward(tensor: torch.Tensor) -> None:
    cast(Callable[[], None], tensor.backward)()


def _tiny_config() -> HighResModelConfig:
    """Construct a lightweight model config for fast tests."""
    return HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_mult=(1,),
        num_res_blocks=1,
        attention_levels=(),
        cross_attention_levels=(),
        time_emb_dim=128,
        conditional=False,
    )


def _build_solver(model: torch.nn.Module) -> SchroedingerBridgeSolver:
    kernel_cfg = KernelConfig(
        solver_type="direct",
        rff_features=256,
        n_landmarks=8,
        max_kernel_cache_size=2,
        epsilon=0.1,  # Increased for better conditioning
    )
    sampler_cfg = SamplerConfig(
        sb_iterations=50,  # Increased for robust convergence
        error_tolerance=1e-3,  # Relaxed for test stability
        marginal_constraint_threshold=5e-2,  # Relaxed threshold for tests
        use_linear_solver=False,
        use_mixed_precision=False,
        verbose_logging=False,
    )
    return SchroedingerBridgeSolver(
        model,
        _linear_schedule,
        device=torch.device("cpu"),
        kernel_config=kernel_cfg,
        sampler_config=sampler_cfg,
    )


def test_score_model_gradient_flow() -> None:
    torch.manual_seed(0)
    model = build_highres_score_model(_tiny_config())

    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    t = torch.rand(2)

    output = model(x, t)
    assert isinstance(output, torch.Tensor)
    loss = output.mean()
    _backward(loss)

    allowed_missing = {"mid_attention.context_proj.weight", "mid_attention.context_proj.bias"}
    params_without_grad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None and name not in allowed_missing
    ]
    assert not params_without_grad, f"Parameters without gradients: {params_without_grad}"

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None and name in allowed_missing:
            continue
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_solver_score_gradient_flow() -> None:
    torch.manual_seed(0)
    model = build_highres_score_model(_tiny_config())
    solver = _build_solver(model)

    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    x_next = solver.solve_once(x, t_curr=0.8, t_next=0.7)
    assert isinstance(x_next, torch.Tensor)

    loss = (x_next ** 2).mean()
    _backward(loss)

    assert x.grad is not None and torch.isfinite(x.grad).all(), "No gradient or non-finite gradient on input"

    allowed_missing = {"mid_attention.context_proj.weight", "mid_attention.context_proj.bias"}
    params_with_grad = sum(
        1
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is not None or name in allowed_missing)
    )
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert params_with_grad == total_params, (
        f"Only {params_with_grad}/{total_params} parameters received gradients (allowed missing: {allowed_missing})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed-precision test")
def test_mixed_precision_gradient_stability() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = build_highres_score_model(_tiny_config()).to(device)

    x = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
    t = torch.rand(2, device=device)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(x, t)
        assert isinstance(output, torch.Tensor)

    loss = output.float().mean()
    _backward(loss)

    grad_magnitudes = [
        param.grad.abs().max().item()
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]
    assert grad_magnitudes, "No gradients recorded"
    assert all(math.isfinite(g) for g in grad_magnitudes), "Non-finite gradients detected"
    assert max(grad_magnitudes) > 0.0, "Gradients underflowed to zero"


def test_no_gradient_detachment() -> None:
    torch.manual_seed(0)
    model = build_highres_score_model(_tiny_config())

    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    t = torch.rand(2)
    output = model(x, t)
    assert isinstance(output, torch.Tensor)

    assert output.requires_grad, "Model output was detached from computation graph"
    _backward(output.mean())
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Input gradient missing or non-finite"
