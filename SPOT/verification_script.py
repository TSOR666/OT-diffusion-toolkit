"""Minimal verification script for SPOT correctness checks."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from SPOT.config import SolverConfig
from SPOT.schedules import CosineSchedule
from SPOT.solver import ProductionSPOTSolver
from SPOT.utils import NoisePredictorToScoreWrapper


class DummyScore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.net(x)  # (B, 3, H, W) -> (B, 3, H, W)


class DummyNoise(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Conv2d(3, 3, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.net(x)  # (B, 3, H, W) -> (B, 3, H, W)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SolverConfig(
        integrator="ddim",
        use_mixed_precision=False,
        use_patch_based_ot=True,
        patch_size=4,
        sinkhorn_iterations=10,
        deterministic=False,
        use_corrector=False,
    )

    schedule = CosineSchedule(device=device, dtype=torch.float32)
    solver = ProductionSPOTSolver(
        score_model=DummyScore(),
        noise_schedule=schedule,
        config=config,
        device=device,
        compute_dtype=torch.float32,
    )

    shape = (1, 3, 8, 8)
    samples = solver.sample_enhanced(shape, num_steps=3, verbose=False, seed=0)
    if isinstance(samples, tuple):
        samples = samples[0]
    if hasattr(samples, "samples"):
        samples = samples.samples

    assert tuple(samples.shape) == shape, f"Unexpected output shape: {tuple(samples.shape)}"
    assert torch.isfinite(samples).all(), "Non-finite values in solver output"

    noise_model = DummyNoise().to(device)
    wrapper = NoisePredictorToScoreWrapper(noise_model, schedule, clamp=1e-8, device=device)
    x = torch.randn(2, 3, 8, 8, device=device, dtype=torch.float32)
    t = torch.full((2,), 0.5, device=device, dtype=torch.float32)
    out = wrapper(x, t)

    assert out.shape == x.shape, f"Wrapper output shape mismatch: {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "Non-finite values in wrapper output"

    loss = out.mean()
    loss.backward()
    for param in noise_model.parameters():
        assert param.grad is not None, "Missing gradient in noise model"
        assert torch.isfinite(param.grad).all(), "Non-finite gradient in noise model"

    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
