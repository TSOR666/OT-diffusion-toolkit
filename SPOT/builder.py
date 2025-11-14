"""Builder utilities and factory functions for SPOT solvers."""
from __future__ import annotations

import torch

from .config import SolverConfig
from .constants import DEFAULT_DPM_ORDER
from .solver import ProductionSPOTSolver

__all__ = [
    "SolverBuilder",
    "create_balanced_solver",
    "create_fast_solver",
    "create_repro_solver",
]


class SolverBuilder:
    """Builder for ergonomic solver configuration."""

    def __init__(self, score_model):
        self.score_model = score_model
        self.config = SolverConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_schedule = None
        self.compute_dtype = None

    def with_device(self, device):
        self.device = device
        return self

    def with_compute_dtype(self, dtype):
        self.compute_dtype = dtype
        return self

    def with_deterministic(self, enabled: bool = True, cdist_cpu: bool = False):
        self.config.deterministic = enabled
        self.config.deterministic_cdist_cpu = cdist_cpu
        return self

    def with_tf32(self, enabled: bool = True):
        self.config.enable_tf32 = enabled
        return self

    def with_dpm_solver_order(self, order: int):
        if order not in [1, 2, 3]:
            raise ValueError(f"DPM-Solver order must be 1-3, got {order}")
        self.config.dpm_solver_order = order
        return self

    def with_richardson_extrapolation(self, enabled: bool = True, threshold: float = 0.1, max_overhead: float = 0.5):
        self.config.richardson_extrapolation = enabled
        self.config.richardson_threshold = threshold
        self.config.richardson_max_overhead = max_overhead
        return self

    def with_timestep_shape_b1(self, enabled: bool = False):
        self.config.timestep_shape_b1 = enabled
        return self

    def with_patch_based_ot(self, enabled: bool = True, patch_size: int = 64):
        self.config.use_patch_based_ot = enabled
        self.config.patch_size = patch_size
        return self

    def with_noise_schedule(self, schedule):
        self.noise_schedule = schedule
        return self

    def with_blockwise_threshold(self, threshold: int):
        self.config.blockwise_threshold = threshold
        return self

    def with_max_dense_matrix_elements(self, max_elements: int):
        self.config.max_dense_matrix_elements = max_elements
        return self

    def with_adaptive_eps_scale(self, mode: str = "sigma"):
        if mode not in ["sigma", "data", "none"]:
            raise ValueError("adaptive_eps_scale must be 'sigma', 'data', or 'none'")
        self.config.adaptive_eps_scale = mode
        return self

    def build(self) -> ProductionSPOTSolver:
        return ProductionSPOTSolver(
            score_model=self.score_model,
            noise_schedule=self.noise_schedule,
            config=self.config,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )


def create_balanced_solver(score_model, device=None, compute_dtype=None) -> ProductionSPOTSolver:
    """Create a solver with balanced speed/quality (recommended)."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_dtype is None:
        compute_dtype = torch.float32

    return (
        SolverBuilder(score_model)
        .with_device(device)
        .with_compute_dtype(compute_dtype)
        .with_dpm_solver_order(DEFAULT_DPM_ORDER)
        .with_richardson_extrapolation(True)
        .with_patch_based_ot(True)
        .with_blockwise_threshold(1_000_000)
        .with_adaptive_eps_scale("sigma")
        .with_deterministic(False)
        .build()
    )


def create_fast_solver(score_model, device=None) -> ProductionSPOTSolver:
    """Create a fast solver with reduced quality."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SolverConfig()
    config.sinkhorn_iterations = 5
    config.dpm_solver_order = 2
    config.use_mixed_precision = device.type == "cuda"
    config.deterministic = False
    config.force_per_pixel_b1 = False
    config.adaptive_eps_scale = "sigma"

    return ProductionSPOTSolver(
        score_model=score_model,
        config=config,
        device=device,
        compute_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )


def create_repro_solver(score_model, device=None) -> ProductionSPOTSolver:
    """Create a reproducible solver with bit-exact determinism."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SolverConfig()
    config.deterministic = True
    config.deterministic_cdist_cpu = True
    config.use_mixed_precision = False
    config.richardson_extrapolation = False

    return ProductionSPOTSolver(
        score_model=score_model,
        config=config,
        device=device,
        compute_dtype=torch.float32,
    )
