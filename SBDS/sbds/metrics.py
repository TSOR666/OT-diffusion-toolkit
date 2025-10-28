"""Metrics utilities for the SBDS solver."""

from __future__ import annotations

import json
import time
import warnings
from typing import Dict, Optional

import numpy as np
import torch


class MetricsLogger:
    """Logger for tracking performance metrics during sampling."""

    def __init__(self, log_file: Optional[str] = None):
        """Initialize the metrics logger."""

        self.log_file = log_file
        self.metrics = {
            "step_times": [],
            "timesteps": [],
            "transport_costs": [],
            "memory_usage": [],
            "module_flops": {},
            "convergence_rates": [],
        }
        self.step_start_time = None

    def start_step(self) -> None:
        """Mark the start of a sampling step."""

        self.step_start_time = time.time()

    def end_step(self, timestep: float, transport_cost: Optional[float] = None) -> None:
        """Mark the end of a sampling step and record metrics."""

        if self.step_start_time is None:
            return

        step_time = time.time() - self.step_start_time
        self.metrics["step_times"].append(step_time)
        self.metrics["timesteps"].append(timestep)

        if transport_cost is not None:
            self.metrics["transport_costs"].append(transport_cost)

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.metrics["memory_usage"].append(memory_used)

        self.step_start_time = None

    def log_module_flops(self, module_name: str, flops: float) -> None:
        """Log approximate FLOPs for a module."""

        if module_name not in self.metrics["module_flops"]:
            self.metrics["module_flops"][module_name] = []

        self.metrics["module_flops"][module_name].append(flops)

    def log_convergence_rate(self, rate: float) -> None:
        """Log convergence rate for theoretical analysis."""

        self.metrics["convergence_rates"].append(rate)

    def save_metrics(self) -> None:
        """Save metrics to file if log_file was provided."""

        if self.log_file is None:
            return

        try:
            with open(self.log_file, "w", encoding="utf-8") as file:
                json.dump(self.metrics, file)
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
