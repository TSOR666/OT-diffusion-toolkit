"""Metrics logging utilities for the SBDS solver."""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from . import common

warnings = common.warnings

__all__ = ["MetricsLogger"]


class MetricsLogger:
    """
    Logger for tracking performance metrics during sampling.
    Tracks computation time, memory usage, and optional transport costs.
    """
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the metrics logger.
        
        Args:
            log_file: Path to log file (optional)
        """
        self.log_file = log_file
        self.metrics = {
            'step_times': [],
            'timesteps': [],
            'transport_costs': [],
            'memory_usage': [],
            'module_flops': {},
            'convergence_rates': []
        }
        self.step_start_time = None
        
    def start_step(self):
        """Mark the start of a sampling step."""
        self.step_start_time = time.time()
        
    def end_step(self, timestep: float, transport_cost: Optional[float] = None):
        """
        Mark the end of a sampling step and record metrics.
        
        Args:
            timestep: Current diffusion timestep
            transport_cost: Optional transport cost for this step
        """
        if self.step_start_time is None:
            return
            
        step_time = time.time() - self.step_start_time
        self.metrics['step_times'].append(step_time)
        self.metrics['timesteps'].append(timestep)
        
        if transport_cost is not None:
            self.metrics['transport_costs'].append(transport_cost)
            
        # Record peak memory usage if using CUDA
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            self.metrics['memory_usage'].append(memory_used)
            
        self.step_start_time = None
        
    def log_module_flops(self, module_name: str, flops: float):
        """
        Log approximate FLOPs for a module.
        
        Args:
            module_name: Name of the module
            flops: Approximate FLOPs for this module
        """
        if module_name not in self.metrics['module_flops']:
            self.metrics['module_flops'][module_name] = []
            
        self.metrics['module_flops'][module_name].append(flops)
        
    def log_convergence_rate(self, rate: float):
        """Log convergence rate for theoretical analysis."""
        self.metrics['convergence_rates'].append(rate)
        
    def save_metrics(self):
        """Save metrics to file if log_file was provided."""
        if self.log_file is not None:
            try:
                import json
                with open(self.log_file, 'w') as f:
                    json.dump(self.metrics, f)
            except Exception as e:
                warnings.warn(f"Failed to save metrics: {e}")
                
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of recorded metrics.
        
        Returns:
            Dictionary of summary statistics
        """
        summary = {}
        
        if self.metrics['step_times']:
            summary['avg_step_time'] = np.mean(self.metrics['step_times'])
            summary['total_time'] = np.sum(self.metrics['step_times'])
            
        if self.metrics['transport_costs']:
            summary['avg_transport_cost'] = np.mean(self.metrics['transport_costs'])
            
        if self.metrics['memory_usage']:
            summary['peak_memory_gb'] = np.max(self.metrics['memory_usage'])
            
        if self.metrics['convergence_rates']:
            summary['avg_convergence_rate'] = np.mean(self.metrics['convergence_rates'])
            
        for module, flops_list in self.metrics['module_flops'].items():
            summary[f'{module}_avg_gflops'] = np.mean(flops_list) / 1e9
            
        return summary



