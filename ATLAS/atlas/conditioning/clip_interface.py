import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config.conditioning_config import ConditioningConfig


class CLIPConditioningInterface:
    """
    Lightweight interface for obtaining CLIP text embeddings suitable for diffusion conditioning.
    Falls back gracefully if CLIP dependencies are unavailable.
    """

    def __init__(
        self,
        config: ConditioningConfig,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pad_id: Optional[int] = None
        self.eot_id: Optional[int] = None
        self.cache: "OrderedDict[Tuple[str, ...], Dict[str, Any]]" = OrderedDict()
        if self.config.use_clip:
            self._setup_clip()

    def _setup_clip(self) -> None:
        try:
            import open_clip  # type: ignore
        except ImportError:
            logging.warning("open_clip not found; CLIP conditioning disabled.")
            self.config.use_clip = False
            return

        try:
            model, _, _ = open_clip.create_model_and_transforms(
                self.config.clip_model,
                pretrained=self.config.clip_pretrained,
                device=self.device,
            )
            tokenizer = open_clip.get_tokenizer(self.config.clip_model)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(
                "Failed to initialize CLIP model '%s': %s",
                self.config.clip_model,
                exc,
            )
            self.config.use_clip = False
            return

        self.model = model.eval().to(self.device)
        if self.config.use_fp16:
            self.model = self.model.half()
        self.tokenizer = tokenizer

        pad_id = getattr(tokenizer, "pad_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None and hasattr(tokenizer, "tokenizer"):
            inner = getattr(tokenizer, "tokenizer")
            pad_id = getattr(inner, "pad_token_id", None)
        if pad_id is not None:
            if hasattr(pad_id, "item"):
                pad_id = int(pad_id.item())
            else:
                pad_id = int(pad_id)
        self.pad_id = pad_id

        eot_id = getattr(tokenizer, "eot_token_id", None)
        if eot_id is None:
            eot_id = getattr(tokenizer, "eot_token", None)
        if eot_id is None and hasattr(tokenizer, "tokenizer"):
            inner = getattr(tokenizer, "tokenizer")
            eot_id = getattr(inner, "eot_token_id", None)
        if eot_id is not None and hasattr(eot_id, "item"):
            eot_id = int(eot_id.item())
        elif isinstance(eot_id, int):
            eot_id = int(eot_id)
        else:
            eot_id = None
        self.eot_id = eot_id

    def _tokenize(self, prompts: List[str]) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("CLIP tokenizer not initialized.")
        tokens = self.tokenizer(prompts)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(dtype=torch.long, device=self.device)
            return tokens

        if isinstance(tokens, dict):
            tokens = tokens.get("input_ids", tokens)

        return torch.as_tensor(tokens, dtype=torch.long, device=self.device)

    def _encode_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("CLIP model not initialized.")

        x = self.model.token_embedding(tokens)
        if hasattr(self.model, "positional_embedding"):
            pos_emb = self.model.positional_embedding
            seq_len = x.size(1)
            if pos_emb.dim() == 1:
                pos_slice = pos_emb[:seq_len].view(1, seq_len, 1)
            else:
                pos_slice = pos_emb[:seq_len]
                if pos_slice.dim() == 2:
                    pos_slice = pos_slice.unsqueeze(0)
            x = x + pos_slice

        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)

        if hasattr(self.model, "ln_final"):
            x = self.model.ln_final(x)

        if self.eot_id is not None:
            eot_mask = tokens.eq(self.eot_id)
            has_eot = eot_mask.any(dim=-1)
            if has_eot.any():
                eot_positions = eot_mask.float().argmax(dim=-1)
                fallback = tokens.new_full((tokens.size(0),), tokens.size(1) - 1)
                cls_indices = torch.where(has_eot, eot_positions, fallback)
            else:
                cls_indices = tokens.new_full((tokens.size(0),), tokens.size(1) - 1)
        else:
            cls_indices = tokens.new_full((tokens.size(0),), tokens.size(1) - 1)
        pooled = x[torch.arange(x.size(0), device=x.device), cls_indices]
        if (
            hasattr(self.model, "text_projection")
            and self.model.text_projection is not None
        ):
            pooled = pooled @ self.model.text_projection

        if self.config.use_fp16:
            x = x.half()
            pooled = pooled.half()

        return x, pooled

    def encode_text(
        self,
        prompts: List[str],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        if not self.config.use_clip or self.model is None:
            raise RuntimeError("CLIP conditioning requested but model is unavailable.")

        key = tuple(prompts)
        if use_cache and self.config.cache_encodings and key in self.cache:
            cached = self.cache[key]
            self.cache.move_to_end(key)
            return {
                k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                for k, v in cached.items()
            }

        tokens = self._tokenize(prompts)
        with torch.no_grad():
            context, pooled = self._encode_tokens(tokens)

        context = context.detach()
        pooled = pooled.detach()

        attention_mask = None
        if self.pad_id is not None:
            attention_mask = tokens.ne(self.pad_id).float()


        payload = {
            "context": context,
            "mask": attention_mask,
            "pooled": pooled,
        }
        if use_cache and self.config.cache_encodings:
            self._store_cache_entry(key, payload)
        return payload

    def build_conditioning_payload(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        neg = negative_prompts or [""] * len(prompts)
        if len(neg) != len(prompts):
            raise ValueError("Negative prompts must match batch size.")

        cond = self.encode_text(prompts)
        uncond = self.encode_text(neg)

        scale = (
            guidance_scale
            if guidance_scale is not None
            else self.config.guidance_scale
        )
        base_batch = cond["context"].size(0)
        payload: Dict[str, Any] = {
            "cond": {
                "context": cond["context"],
                "context_mask": cond["mask"],
                "embedding": cond["pooled"],
            },
            "uncond": {
                "context": uncond["context"],
                "context_mask": uncond["mask"],
                "embedding": uncond["pooled"],
            },
            "guidance_scale": float(scale),
            "base_batch": int(base_batch),
        }
        return payload

    def clear_cache(self) -> None:
        """Clear all cached conditioning payloads."""
        self.cache.clear()

    def _store_cache_entry(self, key: Tuple[str, ...], payload: Dict[str, Any]) -> None:
        """Store payload in cache with eviction policy."""
        if not self.config.cache_encodings or self.config.cache_max_entries <= 0:
            return
        cached = {
            k: v.clone().detach() if isinstance(v, torch.Tensor) else v
            for k, v in payload.items()
        }
        self.cache[key] = cached
        self.cache.move_to_end(key)
        while len(self.cache) > self.config.cache_max_entries:
            self.cache.popitem(last=False)
