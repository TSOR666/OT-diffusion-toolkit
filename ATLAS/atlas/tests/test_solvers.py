import torch
import torch.nn as nn

from atlas.config import KernelConfig, SamplerConfig
from atlas.solvers import SchroedingerBridgeSolver


class _ZeroScore(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning=None) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros_like(x)


def _constant_noise(t):
    if isinstance(t, torch.Tensor):
        return torch.full_like(t, 0.9)
    return 0.9


def test_schrodinger_bridge_solver_step() -> None:
    solver = SchroedingerBridgeSolver(
        score_model=_ZeroScore(),
        noise_schedule=_constant_noise,
        device=torch.device("cpu"),
        kernel_config=KernelConfig(kernel_type="gaussian", epsilon=0.1, solver_type="direct"),
        sampler_config=SamplerConfig(sb_iterations=2, use_linear_solver=False, verbose_logging=False),
    )

    x = torch.randn(2, 4)
    next_x = solver.solve_once(x, 1.0, 0.5)

    assert next_x.shape == x.shape
