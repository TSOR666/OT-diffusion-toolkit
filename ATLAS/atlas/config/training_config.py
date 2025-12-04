"""Training and inference configuration dataclasses for ATLAS experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import warnings


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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty.")
        if not self.root:
            raise ValueError("root path cannot be empty.")
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}.")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}.")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive when set, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}.")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0.")
        if self.fake_size is not None and self.fake_size <= 0:
            raise ValueError(f"fake_size must be positive when set, got {self.fake_size}.")

    def with_overrides(self, **kwargs: object) -> "DatasetConfig":
        """Return a copy with the provided attribute overrides."""

        new_dict = {**vars(self), **kwargs}
        return DatasetConfig(**new_dict)


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

        new_dict = {**vars(self), **kwargs}
        return TrainingConfig(**new_dict)

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
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}.")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}.")
        if self.max_steps is not None and self.epochs > 1:
            warnings.warn(
                "Both epochs and max_steps are set; training will stop at whichever comes first.",
                UserWarning,
                stacklevel=2,
            )
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}.")
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {self.checkpoint_interval}."
            )
        if self.validation_interval is not None and self.validation_interval <= 0:
            raise ValueError(
                f"validation_interval must be positive when set, got {self.validation_interval}."
            )
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive when set, got {self.gradient_clip_norm}."
            )


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

        new_dict = {**vars(self), **kwargs}
        return InferenceConfig(**new_dict)

    def __post_init__(self) -> None:
        if self.sampler_steps <= 0:
            raise ValueError(f"sampler_steps must be positive, got {self.sampler_steps}.")
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale must be non-negative, got {self.guidance_scale}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}.")


__all__ = [
    "DatasetConfig",
    "TrainingConfig",
    "InferenceConfig",
]
