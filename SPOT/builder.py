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

    def with_sinkhorn_iterations(self, iterations: int):
        """Set the number of Sinkhorn iterations for OT computation.

        Args:
            iterations: Number of Sinkhorn-Knopp iterations. Should be >= 20 for
                       mathematically sound transport plans. Lower values (< 10)
                       will fail to satisfy marginal constraints.
        """
        if iterations < 1:
            raise ValueError(f"sinkhorn_iterations must be positive, got {iterations}")
        self.config.sinkhorn_iterations = iterations
        return self

    def with_mixed_precision(self, enabled: bool):
        """Enable mixed precision computation (FP16/BF16 on CUDA).

        Args:
            enabled: If True, use mixed precision for faster computation on CUDA.
        """
        self.config.use_mixed_precision = enabled
        return self

    def build(self) -> ProductionSPOTSolver:
        """Build the ProductionSPOTSolver with the configured parameters.

        Note: If noise_schedule is not set via with_noise_schedule(), the solver
        will default to CosineSchedule internally.
        """
        return ProductionSPOTSolver(
            score_model=self.score_model,
            noise_schedule=self.noise_schedule,
            config=self.config,
            device=self.device,
            compute_dtype=self.compute_dtype,
        )


def create_balanced_solver(score_model, noise_schedule=None, device=None, compute_dtype=None) -> ProductionSPOTSolver:
    """Create a solver with balanced speed/quality (recommended).

    Args:
        score_model: Neural network score model
        noise_schedule: Noise schedule (optional, defaults to CosineSchedule if None)
        device: Compute device (optional, auto-detects if None)
        compute_dtype: Compute precision (optional, defaults to fp32)

    Returns:
        ProductionSPOTSolver configured for balanced speed/quality trade-off
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if compute_dtype is None:
        compute_dtype = torch.float32

    builder = SolverBuilder(score_model)
    if noise_schedule is not None:
        builder = builder.with_noise_schedule(noise_schedule)

    return (
        builder
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


def create_fast_solver(score_model, noise_schedule=None, device=None) -> ProductionSPOTSolver:
    """Create a fast solver with reduced quality.

    Args:
        score_model: Neural network score model
        noise_schedule: Noise schedule (optional, defaults to CosineSchedule if None)
        device: Compute device (optional, auto-detects if None)

    Returns:
        ProductionSPOTSolver configured for speed (reduced quality)

    Note:
        Uses 30 Sinkhorn iterations (down from default 20 in config, but up from
        the previous mathematically insufficient 5). While this is still on the
        lower end, it provides a reasonable balance between speed and transport
        quality. For production use, consider create_balanced_solver() instead.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    builder = SolverBuilder(score_model)
    if noise_schedule is not None:
        builder = builder.with_noise_schedule(noise_schedule)

    return (
        builder
        .with_device(device)
        .with_compute_dtype(dtype)
        .with_mixed_precision(device.type == "cuda")
        .with_dpm_solver_order(2)
        .with_sinkhorn_iterations(30)  # Fixed: increased from 5 to 30 for mathematical soundness
        .with_adaptive_eps_scale("sigma")
        .with_deterministic(False)
        .build()
    )


def create_repro_solver(score_model, noise_schedule=None, device=None) -> ProductionSPOTSolver:
    """Create a reproducible solver with bit-exact determinism.

    Args:
        score_model: Neural network score model
        noise_schedule: Noise schedule (optional, defaults to CosineSchedule if None)
        device: Compute device (optional, auto-detects if None)

    Returns:
        ProductionSPOTSolver configured for reproducible, bit-exact deterministic sampling

    Note:
        Disables mixed precision, Richardson extrapolation, and uses CPU for cdist
        operations to ensure bit-exact reproducibility across runs. This may be
        slower than create_balanced_solver() or create_fast_solver().
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    builder = SolverBuilder(score_model)
    if noise_schedule is not None:
        builder = builder.with_noise_schedule(noise_schedule)

    return (
        builder
        .with_device(device)
        .with_compute_dtype(torch.float32)
        .with_deterministic(enabled=True, cdist_cpu=True)
        .with_mixed_precision(False)
        .with_richardson_extrapolation(False)
        .build()
    )
