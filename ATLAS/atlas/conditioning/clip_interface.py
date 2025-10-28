import logging
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
        self.pad_id: Optional[int] = 0
        self.cache: Dict[Tuple[str, ...], Dict[str, torch.Tensor]] = {}
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

        self.model = model.eval()
        if self.config.use_fp16:
            self.model = self.model.half()
        self.tokenizer = tokenizer

    def _tokenize(self, prompts: List[str]) -> torch.Tensor:
        if self.tokenizer is None:
            raise RuntimeError("CLIP tokenizer not initialized.")
        tokens = self.tokenizer(prompts)
        if isinstance(tokens, torch.Tensor):
            return tokens.to(self.device)
        return torch.tensor(tokens, device=self.device)

    def _encode_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("CLIP model not initialized.")

        x = self.model.token_embedding(tokens).to(self.device)
        if hasattr(self.model, "positional_embedding"):
            pos_emb = self.model.positional_embedding.to(self.device)
            if pos_emb.dim() == 1:
                x = x + pos_emb.unsqueeze(0)
            else:
                x = x + pos_emb

        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)

        if hasattr(self.model, "ln_final"):
            x = self.model.ln_final(x)

        cls_indices = tokens.argmax(dim=-1)
        pooled = x[torch.arange(x.size(0)), cls_indices]
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
            return {k: v.clone() for k, v in cached.items()}

        tokens = self._tokenize(prompts)
        context, pooled = self._encode_tokens(tokens)
        attention_mask = tokens.eq(self.pad_id) if self.pad_id is not None else None

        payload = {
            "context": context,
            "mask": attention_mask,
            "pooled": pooled,
        }
        if use_cache and self.config.cache_encodings:
            # Only clone non-None values
            self.cache[key] = {
                k: v.clone().detach() if v is not None else None
                for k, v in payload.items()
            }
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
