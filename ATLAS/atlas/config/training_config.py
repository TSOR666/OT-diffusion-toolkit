"""Training and inference configuration dataclasses for ATLAS experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration describing how to load a dataset for training or evaluation."""

    name: str
    root: str
    resolution: int
    channels: int
    center_crop: bool = True
    random_flip: bool = True
    download: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    batch_size: Optional[int] = None
    fake_size: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kwargs: object) -> "DatasetConfig":
        """Return a copy with the provided attribute overrides."""

        return replace(self, **kwargs)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters controlling a training run."""

    batch_size: int = 32
    micro_batch_size: Optional[int] = None
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    epochs: int = 300
    max_steps: Optional[int] = None
    log_interval: int = 50
    checkpoint_interval: int = 5000
    validation_interval: Optional[int] = None
    mixed_precision: bool = True
    gradient_clip_norm: Optional[float] = 1.0
    compile: bool = False
    device: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    seed: Optional[int] = 42

    def with_overrides(self, **kwargs: object) -> "TrainingConfig":
        """Return a copy with the provided attribute overrides."""

        return replace(self, **kwargs)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.micro_batch_size is not None:
            if self.micro_batch_size <= 0:
                raise ValueError("micro_batch_size must be positive when set.")
            if self.micro_batch_size > self.batch_size:
                raise ValueError(
                    "micro_batch_size cannot exceed batch_size."
                )
            if self.batch_size % self.micro_batch_size != 0:
                raise ValueError(
                    "batch_size must be divisible by micro_batch_size for accumulation."
                )
        if not (0 < self.ema_decay < 1):
            raise ValueError(
                f"ema_decay must be in (0, 1), got {self.ema_decay}."
            )
        if not (0 < self.betas[0] < 1 and 0 < self.betas[1] < 1):
            raise ValueError(f"betas must be in (0, 1), got {self.betas}.")


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for running inference / sampling."""

    sampler_steps: int = 30
    guidance_scale: float = 1.0
    batch_size: int = 4
    num_samples: int = 16
    seed: Optional[int] = None
    use_ema: bool = True
    device: Optional[str] = None
    output_dir: str = "outputs"

    def with_overrides(self, **kwargs: object) -> "InferenceConfig":
        """Return a copy with the provided attribute overrides."""

        return replace(self, **kwargs)


__all__ = [
    "DatasetConfig",
    "TrainingConfig",
    "InferenceConfig",
]
