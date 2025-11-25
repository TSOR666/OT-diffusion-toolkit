import os
import random
import warnings
from typing import Dict

import numpy as np
import torch


class PerformanceWarning(UserWarning):
    """Warning about performance impact."""
    pass


def set_seed(
    seed: int,
    strict_determinism: bool = True,
    warn_performance: bool = True,
) -> int:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")
    if seed >= 2**32:
        warnings.warn(
            f"Seed {seed} >= 2^32. NumPy will wrap to {seed % (2**32)}.",
            UserWarning,
            stacklevel=2,
        )

    # Python built-in RNG
    random.seed(seed)
    # NumPy legacy RNG
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if strict_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError as exc:
                    warnings.warn(
                        f"Could not enable all deterministic algorithms: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
            if warn_performance:
                warnings.warn(
                    "cuDNN deterministic mode enabled; this may reduce performance.",
                    PerformanceWarning,
                    stacklevel=2,
                )

    # Hash randomization notice
    if strict_determinism:
        hash_seed = os.environ.get("PYTHONHASHSEED")
        if hash_seed not in {"0", "0\n"}:
            warnings.warn(
                "PYTHONHASHSEED not set to 0. For full determinism, run with:\n"
                "    PYTHONHASHSEED=0 python your_script.py",
                UserWarning,
                stacklevel=2,
            )

    return seed


def disable_deterministic_mode() -> None:
    """Disable deterministic settings to restore performance."""
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(False)
        except RuntimeError:
            pass


def get_random_state() -> Dict[str, object]:
    """Capture current random states for reproducibility."""
    state: Dict[str, object] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            torch.cuda.get_rng_state(device)
            for device in range(torch.cuda.device_count())
        ]
    return state


def restore_random_state(state: Dict[str, object]) -> None:
    """Restore random states previously captured by get_random_state()."""
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        for device, rng_state in enumerate(state["torch_cuda"]):  # type: ignore[index]
            torch.cuda.set_rng_state(rng_state, device)


__all__ = [
    "set_seed",
    "disable_deterministic_mode",
    "get_random_state",
    "restore_random_state",
    "PerformanceWarning",
]
