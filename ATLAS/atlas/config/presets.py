"""Common configuration presets used across the ATLAS package."""

from __future__ import annotations

import copy
from typing import Any, Dict

from .conditioning_config import ConditioningConfig, LoRAConfig
from .kernel_config import KernelConfig
from .model_config import HighResModelConfig
from .sampler_config import SamplerConfig
from .training_config import DatasetConfig, InferenceConfig, TrainingConfig


def highres_latent_score_default() -> HighResModelConfig:
    """Baseline configuration targeting 1024x1024 latent diffusion."""

    conditioning_cfg = ConditioningConfig()
    lora_cfg = LoRAConfig()
    return HighResModelConfig(conditioning=conditioning_cfg, lora=lora_cfg)


def gaussian_multiscale_kernel() -> KernelConfig:
    """Balanced kernel configuration that works well for large images."""

    return KernelConfig()


def hierarchical_sampler_default() -> SamplerConfig:
    """Preset tuned for hierarchical high-resolution sampling."""

    return SamplerConfig()


def lsun256_experiment() -> Dict[str, Any]:
    """Preset bundle covering LSUN Bedroom 256x256 training and sampling."""

    model_cfg = HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=160,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_levels=(1, 2),
        num_heads=4,
        dropout=0.1,
        time_emb_dim=640,
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False, context_dim=0),
    )

    dataset_cfg = DatasetConfig(
        name="lsun",
        root="./data/lsun",
        resolution=256,
        channels=3,
        center_crop=True,
        random_flip=True,
        download=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=12,
        extra={"classes": ["bedroom_train"]},
    )

    train_cfg = TrainingConfig(
        batch_size=48,
        micro_batch_size=12,
        learning_rate=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        ema_decay=0.9995,
        epochs=400,
        log_interval=100,
        checkpoint_interval=2500,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=False,
        checkpoint_dir="checkpoints/lsun256",
    )

    infer_cfg = InferenceConfig(
        sampler_steps=36,
        guidance_scale=1.0,
        batch_size=6,
        num_samples=24,
        seed=1234,
        use_ema=True,
        output_dir="outputs/lsun256",
    )

    kernel_cfg = gaussian_multiscale_kernel()
    sampler_cfg = hierarchical_sampler_default().with_overrides(
        sb_iterations=2,
        memory_efficient=True,
        verbose_logging=False,
    )

    return {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "training": train_cfg,
        "inference": infer_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
    }


def celeba1024_experiment() -> Dict[str, Any]:
    """Preset bundle covering CelebA-HQ 1024x1024 training and sampling."""

    model_cfg = HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=208,
        channel_mult=(1, 2, 3, 4, 4),
        num_res_blocks=3,
        attention_levels=(2, 3),
        num_heads=8,
        dropout=0.05,
        time_emb_dim=768,
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False, context_dim=0),
    )

    dataset_cfg = DatasetConfig(
        name="celeba",
        root="./data/celeba_hq",
        resolution=1024,
        channels=3,
        center_crop=True,
        random_flip=True,
        download=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        batch_size=6,
        extra={"split": "train", "target_type": "attr"},
    )

    train_cfg = TrainingConfig(
        batch_size=12,
        micro_batch_size=6,
        learning_rate=1.5e-4,
        betas=(0.9, 0.995),
        weight_decay=5e-5,
        ema_decay=0.999,
        epochs=600,
        log_interval=50,
        checkpoint_interval=2000,
        mixed_precision=True,
        gradient_clip_norm=0.7,
        compile=True,
        checkpoint_dir="checkpoints/celeba1024",
    )

    infer_cfg = InferenceConfig(
        sampler_steps=48,
        guidance_scale=1.0,
        batch_size=4,
        num_samples=12,
        seed=2025,
        use_ema=True,
        output_dir="outputs/celeba1024",
    )

    kernel_cfg = gaussian_multiscale_kernel().with_overrides(
        multi_scale=True,
        scale_factors=(1.0, 0.5, 0.25),
    )
    sampler_cfg = hierarchical_sampler_default().with_overrides(
        sb_iterations=3,
        memory_efficient=True,
        verbose_logging=False,
    )

    return {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "training": train_cfg,
        "inference": infer_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
    }


def ffhq128_experiment() -> Dict[str, Any]:
    """Preset bundle covering FFHQ 128x128 training and sampling."""

    model_cfg = HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=160,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_levels=(1, 2),
        num_heads=4,
        dropout=0.1,
        time_emb_dim=640,
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False, context_dim=0),
    )

    dataset_cfg = DatasetConfig(
        name="ffhq",
        root="./data/ffhq128",
        resolution=128,
        channels=3,
        center_crop=True,
        random_flip=True,
        download=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=16,
        extra={"split": "train"},
    )

    train_cfg = TrainingConfig(
        batch_size=64,
        micro_batch_size=16,
        learning_rate=3e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        ema_decay=0.999,
        epochs=800,
        log_interval=100,
        checkpoint_interval=2000,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=False,
        checkpoint_dir="checkpoints/ffhq128",
    )

    infer_cfg = InferenceConfig(
        sampler_steps=36,
        guidance_scale=1.0,
        batch_size=8,
        num_samples=32,
        seed=2026,
        use_ema=True,
        output_dir="outputs/ffhq128",
    )

    kernel_cfg = gaussian_multiscale_kernel()
    sampler_cfg = hierarchical_sampler_default().with_overrides(
        sb_iterations=2,
        memory_efficient=True,
        verbose_logging=False,
    )

    return {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "training": train_cfg,
        "inference": infer_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
    }


def imagenet64_experiment() -> Dict[str, Any]:
    """Preset bundle covering ImageNet 64x64 training and sampling."""

    model_cfg = HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_levels=(1, 2),
        num_heads=4,
        dropout=0.1,
        time_emb_dim=512,
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False, context_dim=0),
    )

    dataset_cfg = DatasetConfig(
        name="imagenet64",
        root="./data/imagenet64",
        resolution=64,
        channels=3,
        center_crop=False,
        random_flip=True,
        download=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=64,
        extra={"split": "train"},
    )

    train_cfg = TrainingConfig(
        batch_size=256,
        micro_batch_size=64,
        learning_rate=2e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        ema_decay=0.999,
        epochs=600,
        log_interval=100,
        checkpoint_interval=4000,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=False,
        checkpoint_dir="checkpoints/imagenet64",
    )

    infer_cfg = InferenceConfig(
        sampler_steps=28,
        guidance_scale=1.0,
        batch_size=16,
        num_samples=64,
        seed=2027,
        use_ema=True,
        output_dir="outputs/imagenet64",
    )

    kernel_cfg = gaussian_multiscale_kernel()
    sampler_cfg = hierarchical_sampler_default().with_overrides(
        sb_iterations=2,
        memory_efficient=True,
        verbose_logging=False,
    )

    return {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "training": train_cfg,
        "inference": infer_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
    }


def cifar10_experiment() -> Dict[str, Any]:
    """Preset bundle covering CIFAR-10 training and sampling."""

    model_cfg = HighResModelConfig(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_levels=(1, 2),
        num_heads=4,
        dropout=0.1,
        time_emb_dim=512,
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False, context_dim=0),
    )

    dataset_cfg = DatasetConfig(
        name="cifar10",
        root="./data/cifar10",
        resolution=32,
        channels=3,
        center_crop=False,
        random_flip=True,
        download=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        batch_size=128,
        extra={"split": "train"},
    )

    train_cfg = TrainingConfig(
        batch_size=256,
        micro_batch_size=64,
        learning_rate=2e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        ema_decay=0.999,
        epochs=400,
        log_interval=100,
        checkpoint_interval=2000,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=False,
        checkpoint_dir="checkpoints/cifar10",
    )

    infer_cfg = InferenceConfig(
        sampler_steps=18,
        guidance_scale=1.0,
        batch_size=32,
        num_samples=64,
        seed=2028,
        use_ema=True,
        output_dir="outputs/cifar10",
    )

    kernel_cfg = gaussian_multiscale_kernel()
    sampler_cfg = hierarchical_sampler_default().with_overrides(
        sb_iterations=2,
        memory_efficient=True,
        verbose_logging=False,
    )

    return {
        "model": model_cfg,
        "dataset": dataset_cfg,
        "training": train_cfg,
        "inference": infer_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
    }


# ============================================================================
# Consumer GPU Presets (Added for non-expert users)
# ============================================================================


def consumer_6gb_preset() -> Dict[str, Any]:
    """
    Optimized preset for 6GB consumer GPUs (GTX 1660, RTX 3050, RTX 4050).

    Features:
    - 512x512 resolution (memory-friendly)
    - Batch size 1 (minimal memory)
    - Mixed precision enabled (saves 50% memory)
    - FFT kernel solver (fast and memory-efficient)
    - CLIP disabled (saves ~300MB)
    - Gradient checkpointing enabled

    Typical memory usage: ~4-5 GB
    """
    model_cfg = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=128,  # Reduced from 192
        channel_mult=(1, 2, 2, 4),  # Simpler architecture
        num_res_blocks=2,
        attention_levels=(1,),  # Only one attention level
        num_heads=4,
        dropout=0.1,
        time_emb_dim=512,  # Reduced from 768
        conditional=False,
        cross_attention_levels=(),
        conditioning=ConditioningConfig(use_clip=False, context_dim=0),
    )

    kernel_cfg = KernelConfig(
        solver_type="fft",  # Fast and memory-efficient
        epsilon=0.01,
        rff_features=1024,  # Reduced from 2048
        n_landmarks=50,  # Reduced from 100
        max_kernel_cache_size=4,  # Small cache
    )

    sampler_cfg = SamplerConfig(
        sb_iterations=2,  # Reduced from 3
        error_tolerance=1e-3,  # Slightly relaxed
        use_linear_solver=True,
        hierarchical_sampling=True,
        memory_efficient=True,
        use_mixed_precision=True,
        memory_threshold_mb=5120,  # 5GB warning threshold
    )

    training_cfg = TrainingConfig(
        batch_size=1,
        micro_batch_size=1,
        learning_rate=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        ema_decay=0.9995,
        epochs=400,
        log_interval=50,
        checkpoint_interval=1000,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=False,
        checkpoint_dir="checkpoints/consumer_6gb",
    )

    inference_cfg = InferenceConfig(
        sampler_steps=25,  # Faster inference
        guidance_scale=1.0,
        batch_size=1,
        num_samples=4,
        seed=42,
        use_ema=True,
        output_dir="outputs/consumer_6gb",
    )

    return {
        "model": model_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
        "training": training_cfg,
        "inference": inference_cfg,
        "description": "Optimized for 6GB GPUs (GTX 1660, RTX 3050)",
        "resolution": 512,
        "max_batch_size": 1,
    }


def consumer_8gb_preset() -> Dict[str, Any]:
    """
    Optimized preset for 8GB consumer GPUs (RTX 3060, RTX 4060).

    Features:
    - 512x512 resolution
    - Batch size 2
    - Mixed precision enabled
    - FFT kernel solver
    - CLIP enabled for text-to-image
    - Good balance of speed and quality

    Typical memory usage: ~6-7 GB
    """
    model_cfg = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=160,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_levels=(1, 2),
        num_heads=4,
        dropout=0.1,
        time_emb_dim=640,
        conditional=True,
        cross_attention_levels=(1, 2),
        conditioning=ConditioningConfig(use_clip=True, context_dim=768),
    )

    kernel_cfg = KernelConfig(
        solver_type="fft",
        epsilon=0.01,
        rff_features=1536,
        n_landmarks=75,
        max_kernel_cache_size=8,
    )

    sampler_cfg = SamplerConfig(
        sb_iterations=3,
        error_tolerance=1e-4,
        use_linear_solver=True,
        hierarchical_sampling=True,
        memory_efficient=True,
        use_mixed_precision=True,
        memory_threshold_mb=7168,  # 7GB threshold
    )

    training_cfg = TrainingConfig(
        batch_size=2,
        micro_batch_size=2,
        learning_rate=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        ema_decay=0.9995,
        epochs=400,
        log_interval=100,
        checkpoint_interval=1500,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=False,
        checkpoint_dir="checkpoints/consumer_8gb",
    )

    inference_cfg = InferenceConfig(
        sampler_steps=50,
        guidance_scale=7.5,
        batch_size=2,
        num_samples=8,
        seed=42,
        use_ema=True,
        output_dir="outputs/consumer_8gb",
    )

    return {
        "model": model_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
        "training": training_cfg,
        "inference": inference_cfg,
        "description": "Optimized for 8GB GPUs (RTX 3060, RTX 4060)",
        "resolution": 512,
        "max_batch_size": 2,
    }


def consumer_12gb_preset() -> Dict[str, Any]:
    """
    Optimized preset for 12GB consumer GPUs (RTX 3080, RTX 4070 Ti).

    Features:
    - 1024x1024 resolution (high quality)
    - Batch size 4
    - Mixed precision enabled
    - Auto kernel selection
    - CLIP enabled
    - Full feature set

    Typical memory usage: ~10-11 GB
    """
    model_cfg = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=192,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_levels=(1, 2),
        num_heads=8,
        dropout=0.1,
        time_emb_dim=768,
        conditional=True,
        cross_attention_levels=(1, 2),
        conditioning=ConditioningConfig(use_clip=True, context_dim=768),
    )

    kernel_cfg = KernelConfig(
        solver_type="auto",  # Auto-select best kernel
        epsilon=0.01,
        rff_features=2048,
        n_landmarks=100,
        max_kernel_cache_size=12,
    )

    sampler_cfg = SamplerConfig(
        sb_iterations=3,
        error_tolerance=1e-4,
        use_linear_solver=True,
        hierarchical_sampling=True,
        memory_efficient=True,
        use_mixed_precision=True,
        memory_threshold_mb=11264,  # 11GB threshold
    )

    training_cfg = TrainingConfig(
        batch_size=4,
        micro_batch_size=4,
        learning_rate=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        ema_decay=0.9995,
        epochs=400,
        log_interval=100,
        checkpoint_interval=2000,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=True,  # Compile for speed
        checkpoint_dir="checkpoints/consumer_12gb",
    )

    inference_cfg = InferenceConfig(
        sampler_steps=50,
        guidance_scale=7.5,
        batch_size=4,
        num_samples=16,
        seed=42,
        use_ema=True,
        output_dir="outputs/consumer_12gb",
    )

    return {
        "model": model_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
        "training": training_cfg,
        "inference": inference_cfg,
        "description": "Optimized for 12GB GPUs (RTX 3080, RTX 4070 Ti)",
        "resolution": 1024,
        "max_batch_size": 4,
    }


def prosumer_16gb_preset() -> Dict[str, Any]:
    """
    Optimized preset for 16GB prosumer GPUs (RTX 4080, RTX 4090).

    Features:
    - 1024x1024 resolution
    - Batch size 8
    - Mixed precision enabled
    - Auto kernel selection
    - Full feature set
    - Higher quality settings

    Typical memory usage: ~14-15 GB
    """
    model_cfg = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=192,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=3,  # More blocks for quality
        attention_levels=(1, 2),
        num_heads=8,
        dropout=0.05,
        time_emb_dim=768,
        conditional=True,
        cross_attention_levels=(1, 2),
        conditioning=ConditioningConfig(use_clip=True, context_dim=768),
    )

    kernel_cfg = KernelConfig(
        solver_type="auto",
        epsilon=0.01,
        rff_features=2048,
        n_landmarks=100,
        max_kernel_cache_size=16,
    )

    sampler_cfg = SamplerConfig(
        sb_iterations=3,
        error_tolerance=1e-4,
        use_linear_solver=True,
        hierarchical_sampling=True,
        memory_efficient=False,  # Can afford more memory
        use_mixed_precision=True,
        memory_threshold_mb=15360,  # 15GB threshold
    )

    training_cfg = TrainingConfig(
        batch_size=8,
        micro_batch_size=8,
        learning_rate=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        ema_decay=0.9995,
        epochs=400,
        log_interval=100,
        checkpoint_interval=2000,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=True,
        checkpoint_dir="checkpoints/prosumer_16gb",
    )

    inference_cfg = InferenceConfig(
        sampler_steps=50,
        guidance_scale=7.5,
        batch_size=8,
        num_samples=32,
        seed=42,
        use_ema=True,
        output_dir="outputs/prosumer_16gb",
    )

    return {
        "model": model_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
        "training": training_cfg,
        "inference": inference_cfg,
        "description": "Optimized for 16GB GPUs (RTX 4080, RTX 4090)",
        "resolution": 1024,
        "max_batch_size": 8,
    }


def professional_24gb_preset() -> Dict[str, Any]:
    """
    Optimized preset for 24GB professional GPUs (RTX 4090, A5000, A5500).

    Features:
    - 1024x1024 resolution
    - Batch size 16 (large batches)
    - Full precision (FP32) option
    - Auto kernel selection
    - Maximum quality settings
    - All features enabled

    Typical memory usage: ~20-22 GB
    """
    model_cfg = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=192,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=3,
        attention_levels=(1, 2),
        num_heads=8,
        dropout=0.05,
        time_emb_dim=768,
        conditional=True,
        cross_attention_levels=(1, 2),
        conditioning=ConditioningConfig(use_clip=True, context_dim=768),
    )

    kernel_cfg = KernelConfig(
        solver_type="auto",
        epsilon=0.01,
        rff_features=4096,  # Larger for better quality
        n_landmarks=200,  # More landmarks
        max_kernel_cache_size=32,  # Large cache
    )

    sampler_cfg = SamplerConfig(
        sb_iterations=3,
        error_tolerance=1e-4,
        use_linear_solver=True,
        hierarchical_sampling=True,
        memory_efficient=False,
        use_mixed_precision=False,  # Can use FP32
        memory_threshold_mb=23552,  # 23GB threshold
    )

    training_cfg = TrainingConfig(
        batch_size=16,
        micro_batch_size=16,
        learning_rate=2e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        ema_decay=0.9995,
        epochs=400,
        log_interval=100,
        checkpoint_interval=2000,
        mixed_precision=False,  # FP32 for max quality
        gradient_clip_norm=1.0,
        compile=True,
        checkpoint_dir="checkpoints/professional_24gb",
    )

    inference_cfg = InferenceConfig(
        sampler_steps=100,  # High quality
        guidance_scale=7.5,
        batch_size=16,
        num_samples=64,
        seed=42,
        use_ema=True,
        output_dir="outputs/professional_24gb",
    )

    return {
        "model": model_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
        "training": training_cfg,
        "inference": inference_cfg,
        "description": "Optimized for 24GB GPUs (RTX 4090, A5000, A5500)",
        "resolution": 1024,
        "max_batch_size": 16,
    }


def flagship_32gb_preset() -> Dict[str, Any]:
    """Preset tuned for 32GB flagship consumer GPUs (e.g. RTX 5090)."""

    model_cfg = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        base_channels=224,
        channel_mult=(1, 2, 4, 4, 4),
        num_res_blocks=3,
        attention_levels=(1, 2, 3),
        num_heads=8,
        dropout=0.05,
        time_emb_dim=896,
        conditional=True,
        cross_attention_levels=(1, 2, 3),
        conditioning=ConditioningConfig(use_clip=True, context_dim=768),
    )

    kernel_cfg = KernelConfig(
        solver_type="auto",
        epsilon=0.008,
        rff_features=5120,
        n_landmarks=256,
        max_kernel_cache_size=48,
        multi_scale=True,
        scale_factors=[0.5, 1.0, 2.0, 3.0],
    )

    sampler_cfg = SamplerConfig(
        sb_iterations=3,
        error_tolerance=5e-5,
        use_linear_solver=True,
        hierarchical_sampling=True,
        memory_efficient=False,
        use_mixed_precision=True,
        memory_threshold_mb=30720,
        auto_tuning=True,
    )

    training_cfg = TrainingConfig(
        batch_size=12,
        micro_batch_size=6,
        learning_rate=1.8e-4,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
        ema_decay=0.9995,
        epochs=400,
        log_interval=100,
        checkpoint_interval=2000,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        compile=True,
        checkpoint_dir="checkpoints/flagship_32gb",
    )

    inference_cfg = InferenceConfig(
        sampler_steps=80,
        guidance_scale=7.5,
        batch_size=12,
        num_samples=48,
        seed=42,
        use_ema=True,
        output_dir="outputs/flagship_32gb",
    )

    return {
        "model": model_cfg,
        "kernel": kernel_cfg,
        "sampler": sampler_cfg,
        "training": training_cfg,
        "inference": inference_cfg,
        "description": "Optimized for 32GB flagship GPUs (RTX 5090 / 5090 Ti)",
        "resolution": 1024,
        "max_batch_size": 12,
    }


PRESETS = {
    "model:highres_default": highres_latent_score_default,
    "kernel:gaussian_multiscale": gaussian_multiscale_kernel,
    "sampler:hierarchical_default": hierarchical_sampler_default,
    "experiment:lsun256": lsun256_experiment,
    "experiment:celeba1024": celeba1024_experiment,
    "experiment:ffhq128": ffhq128_experiment,
    "experiment:imagenet64": imagenet64_experiment,
    "experiment:cifar10": cifar10_experiment,
    # Consumer GPU presets
    "gpu:6gb": consumer_6gb_preset,
    "gpu:8gb": consumer_8gb_preset,
    "gpu:12gb": consumer_12gb_preset,
    "gpu:16gb": prosumer_16gb_preset,
    "gpu:24gb": professional_24gb_preset,
    "gpu:32gb": flagship_32gb_preset,
}


def load_preset(name: str) -> Any:
    """Materialize a preset by name, returning a detached copy."""

    try:
        factory = PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Unknown preset '{name}'. Available keys: {sorted(PRESETS)}") from exc
    value = factory()
    return copy.deepcopy(value)
