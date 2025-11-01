"""Self-test utilities for the SPOT solver."""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from ._version import __version__
from .builder import SolverBuilder
from .logger import logger

__all__ = ["selftest", "validate_install", "self_test", "main"]


def selftest(verbose: bool = True) -> Dict[str, Any]:
    """Module-level self-test function using a dummy 1x1 conv score model."""

    class DummyScore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x, t):  # pragma: no cover - simple layer
            return self.net(x)

    solver = (
        SolverBuilder(DummyScore())
        .with_compute_dtype(torch.float32)
        .with_deterministic(True, cdist_cpu=True)
        .with_tf32(False)
        .with_patch_based_ot(True, patch_size=16)
        .with_adaptive_eps_scale("sigma")
        .build()
    )

    results = solver.selftest(verbose=verbose)
    solver.cleanup()
    return results


def validate_install() -> Dict[str, Any]:
    """Validate installation using built-in self-test."""

    logger.info("🔍 Validating SPOT %s installation...", __version__)
    try:
        test_results = selftest(verbose=True)
        if test_results["status"] == "passed":
            logger.info("✅ SPOT %s validation successful!", __version__)
            logger.info("   All critical operations verified")
            return {"status": "success", "version": __version__, "test_results": test_results}
        logger.error("❌ SPOT %s validation failed!", __version__)
        return {"status": "failed", "version": __version__, "test_results": test_results}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("❌ Validation failed with exception: %s", exc)
        return {"status": "error", "error": str(exc)}


def self_test(verbose: bool = False) -> bool:
    """Simple self-test entry point."""

    if verbose:
        result = validate_install()
    else:
        prev_level = logger.level
        logger.setLevel(logging.CRITICAL + 1)
        try:
            result = validate_install()
        finally:
            logger.setLevel(prev_level)
    return result["status"] == "success"


def main() -> int:
    """Console entrypoint mirroring ``python -m solvers.spot`` behaviour."""

    logging.basicConfig(level=logging.INFO)
    result = validate_install()

    if result["status"] == "success":
        print(f"\n✅ SPOT {__version__} PRODUCTION FREEZE - Ready for Deployment!")
        print("All critical tests passed successfully.")
        print("\nKey features validated:")
        print("- Per-pixel transport (no N=M=1 degeneracy)")
        print("- Deterministic sampling with local generators")
        print("- Bit-exact determinism option (deterministic_cdist_cpu)")
        print("- Thread-safe concurrent usage without race conditions")
        print("- Small image support (8x8+)")
        print("- Numerical stability with NaN propagation & soft fallbacks")
        print("- Patch OT with cached norms & provided y_flat")
        print("- Integration correctness (no double-drift)")
        print("- Production presets (balanced/fast/repro)")
        return 0

    print("\n❌ Validation failed. Please check the logs above.")
    return 1
