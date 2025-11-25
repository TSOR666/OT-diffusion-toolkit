"""CUDA graph utilities for accelerating repeated sampler calls."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import inspect
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

    def output(self, copy: bool = True) -> torch.Tensor:
        """
        Return captured output; clone by default to avoid overwrites on replay.
        Set copy=False only if you consume the output before the next forward.
        """
        return self.static_out.clone() if copy else self.static_out


class CUDAGraphModelWrapper(nn.Module):
    """Wrap a model to execute forward passes via CUDA graphs with LRU cache."""

    def __init__(self, model: nn.Module, warmup_iters: int = 2, max_cache_size: int = 32) -> None:
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

        try:
            signature = inspect.signature(self.model.forward)
            params = list(signature.parameters.values())
            has_var_positional = any(p.kind == p.VAR_POSITIONAL for p in params)
            has_var_keyword = any(p.kind == p.VAR_KEYWORD for p in params)
            if has_var_positional or has_var_keyword:
                self._graphs_supported = False
                logger.info(
                    "CUDA graphs disabled because the wrapped model accepts *args/**kwargs, "
                    "which are incompatible with static graph capture."
                )
            fixed_params = [
                p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(fixed_params) > 3:  # self + x + t (+ extras)
                self._graphs_supported = False
                logger.info(
                    "CUDA graphs disabled because the wrapped model accepts additional positional "
                    "arguments (likely conditioning)."
                )
        except (ValueError, TypeError):
            pass

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
            x.device.index if x.device.type == "cuda" else None,
            self.model.training,
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
                evicted_key, evicted_handle = self._graphs.popitem(last=False)
                del evicted_handle.static_x
                del evicted_handle.static_t
                del evicted_handle.static_out
                del evicted_handle.graph
            torch.cuda.empty_cache()
        else:
            # Move to end to mark as recently used
            self._graphs.move_to_end(key)

        handle.copy_inputs(x, t)
        handle.replay()
        return handle.output()

    @torch.no_grad()
    def _create_graph(self, x: torch.Tensor, t: torch.Tensor) -> _GraphHandle:
        if x.device.type != "cuda" or t.device.type != "cuda":
            raise ValueError("CUDA graphs require CUDA tensors.")
        device_index = x.device.index if x.device.index is not None else 0

        static_x = x.clone()
        static_t = t.clone()

        # Warmup to populate caches for deterministic graph capture
        if self.warmup_iters > 0:
            for _ in range(self.warmup_iters):
                _ = self.model(static_x, static_t)
            torch.cuda.synchronize(device_index)

        static_out = self.model(static_x, static_t).clone()
        graph = torch.cuda.CUDAGraph()

        torch.cuda.synchronize(device_index)
        try:
            with torch.cuda.graph(graph):
                static_out.copy_(self.model(static_x, static_t))
        except RuntimeError as exc:
            raise RuntimeError(
                f"CUDA graph capture failed: {exc}. Ensure the model graph is static and CUDA-only."
            ) from exc

        torch.cuda.synchronize(device_index)
        return _GraphHandle(graph=graph, static_x=static_x, static_t=static_t, static_out=static_out)

    def enable_graphs(self, enabled: bool = True) -> None:
        """Enable or disable CUDA graph acceleration and clear cache when disabling."""
        self._graphs_disabled = not enabled
        if not enabled:
            self.clear_cache()

    def clear_cache(self) -> None:
        """Clear cached CUDA graphs and free associated GPU memory."""
        for handle in self._graphs.values():
            del handle.static_x
            del handle.static_t
            del handle.static_out
            del handle.graph
        self._graphs.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __getattr__(self, name: str):
        # Forward attribute lookups to wrapped model when not found on wrapper
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.model, name)
