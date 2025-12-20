"""Metrics utilities for the SBDS solver."""

from __future__ import annotations

import json
import time
import warnings
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
import torch


class _MetricsDict(TypedDict):
    """Type definition for metrics dictionary."""

    step_times: List[float]
    timesteps: List[float]
    transport_costs: List[float]
    memory_usage: List[float]
    module_flops: Dict[str, List[float]]
    convergence_rates: List[float]


class MetricsLogger:
    """Logger for tracking performance metrics during sampling."""

    def __init__(self, log_file: Optional[str] = None) -> None:
        """Initialize the metrics logger."""

        self.log_file = log_file
        self.metrics: _MetricsDict = {
            "step_times": [],
            "timesteps": [],
            "transport_costs": [],
            "memory_usage": [],
            "module_flops": {},
            "convergence_rates": [],
        }
        self.step_start_time: Optional[float] = None

    def start_step(self) -> None:
        """Mark the start of a sampling step."""

        if torch.cuda.is_available():
            # Reset to capture per-step peaks and ensure prior work is finished
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.step_start_time = time.time()

    def end_step(self, timestep: float, transport_cost: Optional[float] = None) -> None:
        """Mark the end of a sampling step and record metrics."""

        if self.step_start_time is None:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - self.step_start_time
        self.metrics["step_times"].append(float(step_time))
        self.metrics["timesteps"].append(float(self._to_float(timestep)))

        if transport_cost is not None:
            self.metrics["transport_costs"].append(float(self._to_float(transport_cost)))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.metrics["memory_usage"].append(float(memory_used))

        self.step_start_time = None

    def log_module_flops(self, module_name: str, flops: Union[int, float, np.integer[Any], np.floating[Any]]) -> None:
        """Log approximate FLOPs for a module."""

        if module_name not in self.metrics["module_flops"]:
            self.metrics["module_flops"][module_name] = []

        self.metrics["module_flops"][module_name].append(float(flops))

    def log_convergence_rate(self, rate: float) -> None:
        """Log convergence rate for theoretical analysis."""

        self.metrics["convergence_rates"].append(float(self._to_float(rate)))

    def save_metrics(self) -> None:
        """Save metrics to file if log_file was provided."""

        if self.log_file is None:
            return

        try:
            serializable = self._as_serializable()
            with open(self.log_file, "w", encoding="utf-8") as file:
                json.dump(serializable, file)
        except Exception as exc:  # pragma: no cover - best effort logging
            warnings.warn(f"Failed to save metrics: {exc}")

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of recorded metrics."""

        summary: Dict[str, float] = {}

        if self.metrics["step_times"]:
            summary["avg_step_time"] = float(np.mean(self.metrics["step_times"]))
            summary["total_time"] = float(np.sum(self.metrics["step_times"]))

        if self.metrics["transport_costs"]:
            summary["avg_transport_cost"] = float(np.mean(self.metrics["transport_costs"]))

        if self.metrics["memory_usage"]:
            summary["peak_memory_gb"] = float(np.max(self.metrics["memory_usage"]))

        if self.metrics["convergence_rates"]:
            summary["avg_convergence_rate"] = float(np.mean(self.metrics["convergence_rates"]))

        for module, flops_list in self.metrics["module_flops"].items():
            summary[f"{module}_avg_gflops"] = float(np.mean(flops_list) / 1e9)

        return summary

    def _to_float(self, value: float | int | torch.Tensor | np.generic) -> float:
        """Convert tensors/NumPy scalars to native float for safe logging."""
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        if isinstance(value, np.generic):
            return float(value)
        return float(value)

    def _as_serializable(self) -> Dict[str, object]:
        """Return metrics converted to JSON-serializable Python types."""
        out: Dict[str, object] = {}
        out["step_times"] = [self._to_float(v) for v in self.metrics["step_times"]]
        out["timesteps"] = [self._to_float(v) for v in self.metrics["timesteps"]]
        out["transport_costs"] = [self._to_float(v) for v in self.metrics["transport_costs"]]
        out["memory_usage"] = [self._to_float(v) for v in self.metrics["memory_usage"]]
        out["convergence_rates"] = [self._to_float(v) for v in self.metrics["convergence_rates"]]
        out["module_flops"] = {
            k: [self._to_float(v) for v in vals] for k, vals in self.metrics["module_flops"].items()
        }
        return out
