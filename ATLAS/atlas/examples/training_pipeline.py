"""Reusable helpers for ATLAS dataset-specific training and inference scripts."""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
try:  # pragma: no cover - optional dependency for image IO
    from torchvision.utils import save_image
except ModuleNotFoundError as exc:  # pragma: no cover
    save_image = None  # type: ignore[assignment]
    _TORCHVISION_UTILS_ERROR = exc
else:  # pragma: no cover
    _TORCHVISION_UTILS_ERROR = None

from atlas.config import presets
from atlas.config.training_config import DatasetConfig, InferenceConfig, TrainingConfig
from atlas.models import HighResLatentScoreModel
from atlas.schedules import karras_noise_schedule
from atlas.solvers import AdvancedHierarchicalDiffusionSampler
from atlas.utils import create_dataloader, set_seed


def _expand_alpha(alpha: torch.Tensor) -> torch.Tensor:
    return alpha.view(alpha.shape[0], 1, 1, 1)


def _prepare_images(batch: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        images = batch
    else:
        images = batch[0]
    if images.dtype != torch.float32:
        images = images.float()
    return images


def _update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters(), strict=True):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buf, buf in zip(ema_model.buffers(), model.buffers(), strict=True):
            ema_buf.copy_(buf)


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    bundle: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "bundle": {key: asdict(value) if hasattr(value, "__dataclass_fields__") else value for key, value in bundle.items()},
    }
    torch.save(payload, path)


def run_training(
    preset_name: str,
    *,
    dataset_root: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[str] = None,
    max_steps: Optional[int] = None,
) -> None:
    """Execute a training run based on a named preset."""

    bundle = presets.load_preset(preset_name)
    model_cfg = bundle["model"]
    dataset_cfg: DatasetConfig = bundle["dataset"]
    train_cfg: TrainingConfig = bundle["training"]

    if dataset_root:
        dataset_cfg = dataset_cfg.with_overrides(root=dataset_root)
    if checkpoint_dir:
        train_cfg = train_cfg.with_overrides(checkpoint_dir=checkpoint_dir)
    if device:
        train_cfg = train_cfg.with_overrides(device=device)
    if max_steps is not None:
        train_cfg = train_cfg.with_overrides(max_steps=max_steps)

    actual_device = torch.device(
        train_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    set_seed(train_cfg.seed)

    micro_batch = train_cfg.micro_batch_size or train_cfg.batch_size
    if train_cfg.batch_size % micro_batch != 0:
        raise ValueError(
            f"Batch size {train_cfg.batch_size} is not divisible by micro batch {micro_batch}."
        )
    accum_steps = max(train_cfg.batch_size // micro_batch, 1)

    dataset_cfg = dataset_cfg.with_overrides(batch_size=micro_batch)
    dataloader = create_dataloader(dataset_cfg)

    score_model = HighResLatentScoreModel(model_cfg).to(actual_device)
    ema_model = copy.deepcopy(score_model).to(actual_device)
    for param in ema_model.parameters():
        param.requires_grad_(False)

    if train_cfg.compile and hasattr(torch, "compile"):
        score_model = torch.compile(score_model)  # type: ignore[attr-defined]

    optimizer = torch.optim.AdamW(
        score_model.parameters(),
        lr=train_cfg.learning_rate,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    use_amp = train_cfg.mixed_precision and actual_device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    autocast_ctx = autocast if use_amp else nullcontext

    global_step = 0
    for epoch in range(train_cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        for micro_idx, batch in enumerate(dataloader):
            images = _prepare_images(batch).to(actual_device)
            batch_size = images.shape[0]
            t = torch.rand(batch_size, device=actual_device)
            alpha = karras_noise_schedule(t)
            sigma = torch.sqrt(torch.clamp((1.0 - alpha) / alpha, min=1e-8))
            noise = torch.randn_like(images)
            noisy = _expand_alpha(alpha.sqrt()) * images + _expand_alpha(sigma) * noise

            with autocast_ctx():
                pred_noise = score_model(noisy, t)
                loss = F.mse_loss(pred_noise, noise)
                loss = loss / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (micro_idx + 1) % accum_steps == 0:
                if train_cfg.gradient_clip_norm is not None:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        score_model.parameters(), train_cfg.gradient_clip_norm
                    )
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                _update_ema(score_model, ema_model, train_cfg.ema_decay)
                global_step += 1

                if global_step % train_cfg.log_interval == 0:
                    print(
                        f"[Epoch {epoch+1}] step={global_step} loss={loss.item():.4f}"
                    )

                if global_step % train_cfg.checkpoint_interval == 0:
                    ckpt_path = Path(train_cfg.checkpoint_dir) / f"step_{global_step:07d}.pt"
                    _save_checkpoint(
                        ckpt_path,
                        model=score_model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        step=global_step,
                        epoch=epoch,
                        bundle={
                            "model": model_cfg,
                            "dataset": dataset_cfg,
                            "training": train_cfg,
                        },
                    )

                if train_cfg.max_steps is not None and global_step >= train_cfg.max_steps:
                    return

                optimizer.zero_grad(set_to_none=True)

        if train_cfg.max_steps is not None and global_step >= train_cfg.max_steps:
            break

    final_path = Path(train_cfg.checkpoint_dir) / "latest.pt"
    _save_checkpoint(
        final_path,
        model=score_model,
        ema_model=ema_model,
        optimizer=optimizer,
        step=global_step,
        epoch=train_cfg.epochs,
        bundle={"model": model_cfg, "dataset": dataset_cfg, "training": train_cfg},
    )
def _apply_model_state(
    model: torch.nn.Module,
    checkpoint: Dict[str, object],
    *,
    use_ema: bool,
) -> None:
    state = checkpoint.get("ema") if use_ema and "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state)


def run_inference(
    preset_name: str,
    *,
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Path:
    """Run inference for a preset using a stored checkpoint."""

    bundle = presets.load_preset(preset_name)
    model_cfg = bundle["model"]
    dataset_cfg: DatasetConfig = bundle["dataset"]
    infer_cfg: InferenceConfig = bundle["inference"]
    kernel_cfg = bundle["kernel"]
    sampler_cfg = bundle["sampler"]

    if output_dir:
        infer_cfg = infer_cfg.with_overrides(output_dir=output_dir)
    if device:
        infer_cfg = infer_cfg.with_overrides(device=device)

    actual_device = torch.device(
        infer_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    set_seed(infer_cfg.seed)

    score_model = HighResLatentScoreModel(model_cfg).to(actual_device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _apply_model_state(score_model, checkpoint, use_ema=infer_cfg.use_ema)
    score_model.eval()

    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=score_model,
        noise_schedule=karras_noise_schedule,
        device=actual_device,
        kernel_config=kernel_cfg,
        sampler_config=sampler_cfg,
    )

    timesteps = torch.linspace(1.0, 0.01, infer_cfg.sampler_steps).tolist()
    total = infer_cfg.num_samples
    batch_size = infer_cfg.batch_size
    resolution = dataset_cfg.resolution
    channels = model_cfg.out_channels

    out_dir = Path(infer_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    generated = 0
    batch_index = 0
    while generated < total:
        current_batch = min(batch_size, total - generated)
        samples = sampler.sample(
            (current_batch, channels, resolution, resolution),
            timesteps,
            verbose=True,
        )
        samples = samples.clamp(-1.0, 1.0)
        images = (samples + 1.0) / 2.0
        if save_image is None:
            raise ImportError(
                "torchvision is required to save generated images. Install torchvision "
                "or replace the save logic."
            ) from _TORCHVISION_UTILS_ERROR
        grid_path = out_dir / f"batch_{batch_index:04d}.png"
        nrow = min(4, current_batch)
        save_image(images, grid_path, nrow=nrow)
        saved_paths.append(grid_path)
        generated += current_batch
        batch_index += 1

    index_path = out_dir / "manifest.json"
    index_path.write_text(json.dumps({"images": [p.name for p in saved_paths]}, indent=2))
    return index_path


__all__ = ["run_training", "run_inference"]
