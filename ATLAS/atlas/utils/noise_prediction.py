from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from collections.abc import Callable
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class NoisePredictionAdapter:
    """
    Adapter around arbitrary noise predictors that enforces shape, dtype, and
    conditioning contracts while providing a consistent guidance/conditioning API.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        forward_fn = getattr(model, "forward", None) or model
        if not callable(forward_fn):
            raise TypeError(
                f"Model must be callable or implement forward(); got {type(model)}"
            )
        self._forward_fn = cast(Callable[..., torch.Tensor], forward_fn)
        self._signature = self._safe_signature()
        self._params = (
            self._signature.parameters if self._signature is not None else None
        )

    def predict_noise(
        self,
        x: torch.Tensor,
        t: Union[float, torch.Tensor],
        conditioning: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Run the underlying model and return a validated noise prediction tensor.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input examples must be torch.Tensor instances.")
        if x.ndim == 0:
            raise ValueError("Input tensor must include a batch dimension.")

        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t_tensor = t.expand(x.shape[0]).to(x.device)
            elif t.dim() == 1:
                if t.shape[0] == 1:
                    t_tensor = t.expand(x.shape[0]).to(x.device)
                elif t.shape[0] == x.shape[0]:
                    t_tensor = t.to(x.device)
                else:
                    raise ValueError(
                        f"Timestep batch size {t.shape[0]} does not match input batch {x.shape[0]}."
                    )
            else:
                raise ValueError(f"Timestep tensor must be 0D or 1D, got shape {t.shape}.")
        else:
            t_scalar = float(t)
            t_tensor = torch.full(
                (x.shape[0],), t_scalar, dtype=torch.float32, device=x.device
            )

        noise_pred = self._predict_with_conditioning(x, t_tensor, conditioning)
        return self._ensure_valid_noise(noise_pred, reference=x)

    def _predict_with_conditioning(
        self,
        x: torch.Tensor,
        t_tensor: torch.Tensor,
        conditioning: Optional[Any],
    ) -> torch.Tensor:
        fallback_conditioning = conditioning
        noise_pred: Optional[torch.Tensor] = None

        if isinstance(conditioning, Mapping):
            has_cfg_keys = any(
                key in conditioning for key in {"cond", "uncond", "guidance_scale"}
            )
            if has_cfg_keys:
                guidance_value = conditioning.get("guidance_scale", 1.0)
                if isinstance(guidance_value, torch.Tensor):
                    if guidance_value.numel() != 1:
                        raise ValueError(
                            f"guidance_scale must be scalar, got tensor with {guidance_value.numel()} elements."
                        )
                    guidance_value = guidance_value.item()
                try:
                    guidance = float(guidance_value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"guidance_scale must be numeric, got {type(guidance_value)}"
                    ) from exc

                cond_payload = conditioning.get("cond")
                uncond_payload = conditioning.get("uncond")

                noise_cond = self._call_with_condition(x, t_tensor, cond_payload)
                noise_uncond = self._call_with_condition(x, t_tensor, uncond_payload)

                if noise_cond is not None and noise_uncond is not None:
                    noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                elif noise_cond is not None:
                    noise_pred = noise_cond
                elif noise_uncond is not None:
                    noise_pred = noise_uncond

                remaining = {
                    k: v
                    for k, v in conditioning.items()
                    if k
                    not in {
                        "cond",
                        "uncond",
                        "guidance_scale",
                        "base_batch",
                    }
                }
                fallback_conditioning = remaining or None

        if noise_pred is None and fallback_conditioning is not None:
            noise_pred = self._call_with_condition(x, t_tensor, fallback_conditioning)

        if noise_pred is None:
            noise_pred = self._forward_fn(x, t_tensor)

        if noise_pred is None:
            raise ValueError("The noise predictor returned None.")

        return noise_pred

    def _call_with_condition(
        self,
        x: torch.Tensor,
        t_tensor: torch.Tensor,
        payload: Optional[Any],
    ) -> Optional[torch.Tensor]:
        if payload is None:
            return None

        params = self._params

        if params is not None:
            for kw in ("condition", "conditioning"):
                if kw in params:
                    result = self._try_call(x, t_tensor, **{kw: payload})
                    if result is not None:
                        return result

            if isinstance(payload, Mapping):
                accepted_kwargs = {
                    k: v for k, v in payload.items() if k in params
                }
                if accepted_kwargs:
                    result = self._try_call(x, t_tensor, **accepted_kwargs)
                    if result is not None:
                        return result

        # Fallback attempts without relying on signature introspection
        for attempt in (
            self._try_call(x, t_tensor, condition=payload),
            self._try_call(x, t_tensor, conditioning=payload),
        ):
            if attempt is not None:
                return attempt

        if isinstance(payload, Mapping):
            attempt = self._try_call(x, t_tensor, **payload)
            if attempt is not None:
                return attempt

        return None

    def _try_call(
        self, x: torch.Tensor, t_tensor: torch.Tensor, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        if not kwargs:
            return None
        try:
            result = self._forward_fn(x, t_tensor, **kwargs)
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    f"Noise predictor must return torch.Tensor, got {type(result)}"
                )
            return result
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" in message or "positional arguments" in message:
                return None
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug(
                "Model call failed with kwargs %s: %s", list(kwargs.keys()), exc
            )
            return None

    def _ensure_valid_noise(
        self, noise_pred: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(noise_pred, torch.Tensor):
            raise TypeError("Noise predictor did not return a tensor.")
        if noise_pred.shape != reference.shape:
            raise ValueError(
                f"Noise predictor output has shape {noise_pred.shape}, "
                f"but expected {reference.shape}."
            )

        if not torch.isfinite(noise_pred).all():
            raise ValueError("Noise predictor generated non-finite values.")

        return noise_pred.to(device=reference.device, dtype=reference.dtype)

    def _safe_signature(self) -> Optional[inspect.Signature]:
        try:
            return inspect.signature(self._forward_fn)
        except (TypeError, ValueError):
            return None
