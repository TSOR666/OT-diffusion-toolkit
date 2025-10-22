"""Shared imports and constants for the SBDS solver package."""

from __future__ import annotations

import math
import time
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from pykeops.torch import LazyTensor  # type: ignore
    KEOPS_AVAILABLE = True
except ImportError:
    LazyTensor = None  # type: ignore
    KEOPS_AVAILABLE = False
    warnings.warn("KeOps not available, falling back to standard PyTorch")

__all__ = [
    "Any",
    "Callable",
    "Dict",
    "F",
    "KEOPS_AVAILABLE",
    "LazyTensor",
    "List",
    "Optional",
    "Tuple",
    "Union",
    "contextmanager",
    "math",
    "np",
    "nullcontext",
    "time",
    "torch",
    "tqdm",
    "warnings",
    "nn",
]
