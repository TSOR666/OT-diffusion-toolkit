"""FastSB-OT solver package."""

__version__ = "1.0.0-rc1"

from .cache import MemoryEfficientCacheFixed
from .config import FastSBOTConfig, QualityPreset
from .kernels import KernelModule
from .solver import FastSBOTSolver, example_usage, make_schedule
from .transport import HierarchicalBridge, MomentumTransport, SlicedOptimalTransport, TransportModule
from .common import clear_global_compile_cache
from .utils import NoisePredictorToScoreWrapper, wrap_noise_predictor

__all__ = [
    "__version__",
    "FastSBOTConfig",
    "QualityPreset",
    "FastSBOTSolver",
    "make_schedule",
    "example_usage",
    "MemoryEfficientCacheFixed",
    "KernelModule",
    "SlicedOptimalTransport",
    "MomentumTransport",
    "HierarchicalBridge",
    "TransportModule",
    "NoisePredictorToScoreWrapper",
    "wrap_noise_predictor",
    "clear_global_compile_cache",
]
