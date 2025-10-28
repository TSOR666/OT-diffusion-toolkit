"""Dataset utilities for ATLAS training and evaluation scripts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - optional dependency guard
    from torchvision import datasets, transforms
except ModuleNotFoundError as exc:  # pragma: no cover - executed when torchvision is absent
    datasets = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    _TORCHVISION_ERROR = exc
else:  # pragma: no cover - import path when torchvision available
    _TORCHVISION_ERROR = None

from ..config.training_config import DatasetConfig


def _build_transform(config: DatasetConfig) -> transforms.Compose:
    if transforms is None:
        raise ImportError(
            "torchvision is required for dataset transforms. Install torchvision or "
            "provide a custom pipeline."
        ) from _TORCHVISION_ERROR
    ops = [
        transforms.Resize(
            config.resolution, interpolation=transforms.InterpolationMode.BICUBIC
        )
    ]
    if config.center_crop:
        ops.append(transforms.CenterCrop(config.resolution))
    else:
        ops.append(transforms.RandomCrop(config.resolution))
    if config.random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.5] * config.channels, std=[0.5] * config.channels)])
    return transforms.Compose(ops)


def build_dataset(config: DatasetConfig, *, split: Optional[str] = None) -> Dataset:
    """Instantiate a dataset based on the provided configuration."""

    split = split or config.extra.get("split", "train")
    transform = _build_transform(config)
    name = config.name.lower()
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)

    if name in {"lsun", "lsun256"}:
        if datasets is None:
            raise ImportError(
                "torchvision is required for LSUN datasets. Install torchvision to continue."
            ) from _TORCHVISION_ERROR
        classes = config.extra.get("classes") or ["bedroom_train"]
        return datasets.LSUN(
            root=str(root),
            classes=classes,
            transform=transform,
        )
    if name in {"celeba", "celeba1024", "celeba-hq"}:
        if datasets is None:
            raise ImportError(
                "torchvision is required for CelebA datasets. Install torchvision to continue."
            ) from _TORCHVISION_ERROR
        target_type = config.extra.get("target_type", "attr")
        return datasets.CelebA(
            root=str(root),
            split=split,
            transform=transform,
            download=config.download,
            target_type=target_type,
        )
    if name in {"fake", "synthetic"}:
        if datasets is None:
            raise ImportError(
                "torchvision is required for FakeData. Install torchvision or swap in a custom dataset."
            ) from _TORCHVISION_ERROR
        size = config.fake_size or max(config.batch_size or 1, 1)
        image_size = (config.channels, config.resolution, config.resolution)
        return datasets.FakeData(size=size, image_size=image_size, transform=transform)

    raise ValueError(
        f"Unsupported dataset '{config.name}'. "
        "Supported options: lsun, celeba, fake"
    )


def create_dataloader(
    config: DatasetConfig,
    *,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    split: Optional[str] = None,
) -> DataLoader:
    """Build a PyTorch DataLoader respecting the dataset configuration."""

    dataset = build_dataset(config, split=split)
    eff_batch = batch_size or config.batch_size or 1
    return DataLoader(
        dataset,
        batch_size=eff_batch,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        drop_last=True,
    )


def override_dataset_root(config: DatasetConfig, root: str) -> DatasetConfig:
    """Return a new dataset configuration pointing to a different root path."""

    if not root:
        return config
    return replace(config, root=root)


__all__ = [
    "build_dataset",
    "create_dataloader",
    "override_dataset_root",
]
