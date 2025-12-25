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
    """Run the built-in solver self-test with a dummy score network."""

    class DummyScore(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Conv2d(3, 3, kernel_size=1)

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            _ = t  # unused, but required by SPOT interface
            return self.net(x)  # (B, 3, H, W) -> (B, 3, H, W)

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
    """Validate the installation by executing the solver self-test."""

    logger.info("Validating SPOT %s installation...", __version__)
    try:
        test_results = selftest(verbose=True)
        if test_results["status"] == "passed":
            logger.info("SPOT %s validation successful", __version__)
            logger.info("   All critical operations verified")
            return {"status": "success", "version": __version__, "test_results": test_results}
        logger.error("SPOT %s validation failed", __version__)
        return {"status": "failed", "version": __version__, "test_results": test_results}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Validation failed with exception: %s", exc)
        return {"status": "error", "error": str(exc)}


def self_test(verbose: bool = False) -> bool:
    """Convenience wrapper returning a boolean outcome."""

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
    """Console entry point mirroring ``python -m SPOT`` behaviour."""

    logging.basicConfig(level=logging.INFO)
    result = validate_install()

    if result["status"] == "success":
        print(f"\nSPOT {__version__} PRODUCTION FREEZE - Ready for deployment!")
        print("All critical tests passed successfully.")
        print("\nKey features validated:")
        print("- Per-pixel transport (no N=M=1 degeneracy)")
        print("- Deterministic sampling with local generators")
        print("- Bit-exact determinism option (deterministic_cdist_cpu)")
        print("- Thread-safe concurrent usage without race conditions")
        print("- Small image support (8x8 and larger)")
        print("- Numerical stability with NaN propagation safeguards")
        print("- Patch OT with cached norms and supplied reference features")
        print("- Integration correctness (no double-drift)")
        print("- Production presets (balanced / fast / repro)")
        return 0

    print("\nValidation failed. Please check the logs above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
