"""CUDA graph utilities for accelerating repeated sampler calls."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


@dataclass
class _GraphHandle:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_t: torch.Tensor
    static_out: torch.Tensor

    def copy_inputs(self, x: torch.Tensor, t: torch.Tensor) -> None:
        self.static_x.copy_(x)
        self.static_t.copy_(t)

    def replay(self) -> None:
        self.graph.replay()

    def output(self) -> torch.Tensor:
        return self.static_out.clone()


class CUDAGraphModelWrapper(nn.Module):
    """Wrap a model to execute forward passes via CUDA graphs with LRU cache."""

    def __init__(self, model: nn.Module, warmup_iters: int = 2, max_cache_size: int = 16) -> None:
        super().__init__()
        self.model = model
        self.warmup_iters = int(max(0, warmup_iters))
        self.max_cache_size = int(max(1, max_cache_size))
        self._graphs: "OrderedDict[Tuple, _GraphHandle]" = OrderedDict()
        self._cuda_available = torch.cuda.is_available()
        self._graphs_supported = self._cuda_available and hasattr(torch.cuda, "CUDAGraph")
        self._graphs_disabled = False
        self.predicts_score = getattr(model, "predicts_score", True)
        self.predicts_noise = getattr(model, "predicts_noise", False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if (
            not self._graphs_supported
            or self._graphs_disabled
            or x.device.type != "cuda"
            or t.device.type != "cuda"
        ):
            return self.model(x, t)

        key = (
            tuple(x.shape),
            x.dtype,
            tuple(t.shape),
            t.dtype,
            x.device.index if x.device.type == "cuda" else "cpu",
        )

        handle = self._graphs.get(key)
        if handle is None:
            try:
                handle = self._create_graph(x, t)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "Disabling CUDA graphs after capture failure: %s",
                    exc,
                )
                self._graphs_disabled = True
                self._graphs.clear()
                return self.model(x, t)
            self._graphs[key] = handle
            # LRU eviction: remove oldest entry if cache exceeds max size
            while len(self._graphs) > self.max_cache_size:
                evicted_key = next(iter(self._graphs))
                del self._graphs[evicted_key]
        else:
            # Move to end to mark as recently used
            self._graphs.move_to_end(key)

        handle.copy_inputs(x, t)
        handle.replay()
        return handle.output()

    @torch.no_grad()
    def _create_graph(self, x: torch.Tensor, t: torch.Tensor) -> _GraphHandle:
        static_x = x.clone()
        static_t = t.clone()

        # Warmup to populate caches for deterministic graph capture
        device_index = x.device.index if x.device.type == "cuda" else torch.cuda.current_device()

        if self.warmup_iters > 0:
            for _ in range(self.warmup_iters):
                _ = self.model(x, t)

        torch.cuda.synchronize(device_index)

        static_out = self.model(static_x, static_t).clone()
        graph = torch.cuda.CUDAGraph()

        torch.cuda.synchronize(device_index)
        with torch.cuda.graph(graph):
            static_out.copy_(self.model(static_x, static_t))

        torch.cuda.synchronize(device_index)
        return _GraphHandle(graph=graph, static_x=static_x, static_t=static_t, static_out=static_out)
