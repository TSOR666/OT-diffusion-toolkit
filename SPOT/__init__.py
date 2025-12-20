"""SPOT solver package interface."""
from __future__ import annotations

from ._version import __version__
from .builder import (
    SolverBuilder,
    create_balanced_solver,
    create_fast_solver,
    create_repro_solver,
)
from .config import SolverConfig
from .results import SamplingResult
from .schedules import CosineSchedule, LinearSchedule
from .selftest import main, self_test, selftest, validate_install
from .solver import ProductionSPOTSolver

# Optional advanced features (gracefully degrade if not available)
# These imports are re-exported for the public API
try:
    from .integrators import (  # noqa: F401
        HeunIntegrator,
        DDIMIntegrator,
        AdaptiveIntegrator,
        ExponentialIntegrator,
    )
    __all_integrators__ = [
        "HeunIntegrator",
        "DDIMIntegrator",
        "AdaptiveIntegrator",
        "ExponentialIntegrator",
    ]
except ImportError:
    __all_integrators__ = []

try:
    from .corrector import (  # noqa: F401
        LangevinCorrector,
        TweedieCorrector,
        AdaptiveCorrector,
        PredictorCorrectorSampler,
    )
    __all_correctors__ = [
        "LangevinCorrector",
        "TweedieCorrector",
        "AdaptiveCorrector",
        "PredictorCorrectorSampler",
    ]
except ImportError:
    __all_correctors__ = []

try:
    from .triton_kernels import (  # noqa: F401
        TRITON_AVAILABLE,
        triton_sinkhorn_update,
        fused_cost_softmax,
    )
    __all_triton__ = [
        "TRITON_AVAILABLE",
        "triton_sinkhorn_update",
        "fused_cost_softmax",
    ]
except ImportError:
    __all_triton__ = []

from .utils import NoisePredictorToScoreWrapper, wrap_noise_predictor
__all__ = [
    "ProductionSPOTSolver",
    "SolverConfig",
    "SolverBuilder",
    "CosineSchedule",
    "LinearSchedule",
    "create_balanced_solver",
    "create_fast_solver",
    "create_repro_solver",
    "validate_install",
    "self_test",
    "selftest",
    "SamplingResult",
    "__version__",
    "main",
    "NoisePredictorToScoreWrapper",
    "wrap_noise_predictor",
] + __all_integrators__ + __all_correctors__ + __all_triton__

