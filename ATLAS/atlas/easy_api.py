"""
Simplified ATLAS API for Non-Experts
=====================================

This module provides a high-level, easy-to-use interface for ATLAS that abstracts
away configuration complexity and automatically handles GPU memory constraints.

Quick Start:
-----------
    import atlas.easy_api as atlas

    # For text-to-image generation
    sampler = atlas.create_sampler(
        checkpoint="model.pt",
        gpu_memory="8GB",  # Auto-configures for your GPU
    )

    images = sampler.generate(
        prompts=["a red car", "a blue house"],
        num_samples=4
    )

GPU Memory Presets:

- "6GB": Consumer GPUs (GTX 1660, RTX 3050)
- "8GB": Mid-range GPUs (RTX 3060, RTX 4060)
- "12GB": High-end consumer (RTX 3080, RTX 4070)
- "16GB": Prosumer (RTX 4080, RTX 4090)
- "24GB": Professional (RTX 4090, A5000)
- "32GB": Flagship GPUs (RTX 5090 / 5090 Ti, 4090 Ti)
"""
import torch
import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple, Mapping
from dataclasses import dataclass, replace

from atlas.models.score_network import HighResLatentScoreModel
from atlas.solvers.hierarchical_sampler import AdvancedHierarchicalDiffusionSampler
from atlas.schedules.noise import karras_noise_schedule
from atlas.config.model_config import HighResModelConfig
from atlas.config.kernel_config import KernelConfig
from atlas.config.sampler_config import SamplerConfig
from atlas.config.conditioning_config import ConditioningConfig
from atlas.utils.hardware import safe_cuda_mem_get_info
from atlas.utils.memory import get_peak_memory_mb


def _safe_torch_load(path: Union[str, Path], map_location=None):
    """
    Load a checkpoint with weights_only=True when supported to avoid unsafe pickle execution.
    Falls back to standard torch.load if the installed PyTorch version lacks the flag.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_kwargs = {"map_location": map_location}
    try:
        return torch.load(checkpoint_path, weights_only=True, **load_kwargs)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(checkpoint_path, **load_kwargs)


_CLIP_MODEL_CONTEXT_DIMS = {
    "vit-b-32": 512,
    "vit-b-16": 512,
    "vit-l-14": 768,
    "vit-h-14": 1024,
    "vit-g-14": 1536,
}


# ============================================================================
# GPU Memory Profiles
# ============================================================================

@dataclass
class GPUProfile:
    """Configuration profile optimized for specific GPU memory constraints."""

    name: str
    memory_mb: int
    batch_size: int
    resolution: int
    use_mixed_precision: bool
    kernel_solver: str
    kernel_cache_size: int
    enable_clip: bool
    gradient_checkpointing: bool
    description: str
    supports_bf16: bool = False
    auto_tune: bool = False


GPU_PROFILES = {
    "6GB": GPUProfile(
        name="6GB",
        memory_mb=6144,
        batch_size=1,
        resolution=512,
        use_mixed_precision=True,
        kernel_solver="fft",
        kernel_cache_size=4,
        enable_clip=False,  # Save memory
        gradient_checkpointing=True,
        description="Consumer GPUs: GTX 1660, RTX 3050, RTX 4050"
    ),
    "8GB": GPUProfile(
        name="8GB",
        memory_mb=8192,
        batch_size=2,
        resolution=512,
        use_mixed_precision=True,
        kernel_solver="fft",
        kernel_cache_size=8,
        enable_clip=True,
        gradient_checkpointing=True,
        description="Mid-range GPUs: RTX 3060, RTX 4060"
    ),
    "12GB": GPUProfile(
        name="12GB",
        memory_mb=12288,
        batch_size=4,
        resolution=1024,
        use_mixed_precision=True,
        kernel_solver="auto",
        kernel_cache_size=12,
        enable_clip=True,
        gradient_checkpointing=False,
        description="High-end consumer: RTX 3080, RTX 4070 Ti",
        supports_bf16=False,
        auto_tune=False,
    ),
    "16GB": GPUProfile(
        name="16GB",
        memory_mb=16384,
        batch_size=8,
        resolution=1024,
        use_mixed_precision=True,
        kernel_solver="auto",
        kernel_cache_size=16,
        enable_clip=True,
        gradient_checkpointing=False,
        description="Prosumer GPUs: RTX 4080, RTX 4090, A4000",
        supports_bf16=True,
        auto_tune=True,
    ),
    "24GB": GPUProfile(
        name="24GB",
        memory_mb=24576,
        batch_size=16,
        resolution=1024,
        use_mixed_precision=True,
        kernel_solver="auto",
        kernel_cache_size=32,
        enable_clip=True,
        gradient_checkpointing=False,
        description="Professional GPUs: RTX 4090, RTX 4090 D, A5000",
        supports_bf16=True,
        auto_tune=True,
    ),
    "32GB": GPUProfile(
        name="32GB",
        memory_mb=32768,
        batch_size=20,
        resolution=1536,
        use_mixed_precision=True,
        kernel_solver="auto",
        kernel_cache_size=48,
        enable_clip=True,
        gradient_checkpointing=False,
        description="Flagship GPUs: RTX 5090 / 5090 Ti, A6000 Ada",
        supports_bf16=True,
        auto_tune=True,
    ),
}


def _estimate_memory_usage(
    gpu_profile: GPUProfile,
    sampler_config: SamplerConfig,
    conditioning_config: Optional[ConditioningConfig],
    model_config: HighResModelConfig,
) -> float:
    """Rough but resolution-aware memory estimate in MB."""

    precision_factor = 0.5 if sampler_config.use_mixed_precision else 1.0

    # Parameter memory scales roughly with the square of base channels and network depth
    base_param_mb = 300
    channel_scale = max(model_config.base_channels / 192, 0.5)
    depth_scale = max(len(model_config.channel_mult) / 4, 0.5)
    param_memory = base_param_mb * channel_scale * depth_scale * precision_factor

    # Activation memory grows with batch size, resolution^2, and channels
    base_activation_mb = 500
    resolution_scale = (gpu_profile.resolution / 1024) ** 2
    activation_channel_scale = max(channel_scale, 0.5)
    activations = (
        gpu_profile.batch_size
        * base_activation_mb
        * resolution_scale
        * activation_channel_scale
        * precision_factor
    )

    # Kernel cache size grows with configured cache size
    kernel_cache = 100 + gpu_profile.kernel_cache_size * 5

    clip_memory = 0
    if conditioning_config and conditioning_config.use_clip:
        clip_memory = (180 if sampler_config.use_mixed_precision else 360)

    total = param_memory + activations + kernel_cache + clip_memory
    return total


def detect_gpu_profile() -> GPUProfile:
    """
    Automatically detect available GPU and return appropriate profile.

    Falls back to a conservative CPU profile if no CUDA GPU is detected.

    Returns:
        GPUProfile: Configuration optimized for detected GPU or CPU
    """
    if not torch.cuda.is_available():
        print(
            "[ATLAS] No CUDA GPU detected. Running on CPU with conservative settings.\n"
            "  Note: ATLAS runs best on CUDA; CPU is supported but slow.\n"
            "  If you have a GPU, ensure CUDA drivers are installed:\n"
            "    - NVIDIA drivers: https://www.nvidia.com/Download/index.aspx\n"
            "    - PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )
        # Return conservative 6GB profile for CPU usage
        profile = GPU_PROFILES["6GB"]
        print(f"[ATLAS] Using CPU with profile: {profile.name} - {profile.description}")
        return profile

    # Get GPU memory in MB (use FREE memory to account for other processes)
    free_memory, total_memory = safe_cuda_mem_get_info(0)
    gpu_memory_mb = free_memory / (1024 ** 2)
    gpu_name = torch.cuda.get_device_name(0)

    # Select profile based on available FREE memory
    if gpu_memory_mb < 7000:
        profile_name = "6GB"
    elif gpu_memory_mb < 10000:
        profile_name = "8GB"
    elif gpu_memory_mb < 14000:
        profile_name = "12GB"
    elif gpu_memory_mb < 20000:
        profile_name = "16GB"
    elif gpu_memory_mb < 28000:
        profile_name = "24GB"
    else:
        profile_name = "32GB"

    template = GPU_PROFILES[profile_name]

    print(f"[ATLAS] Detected GPU: {gpu_name}")
    print(f"[ATLAS] Available memory: {gpu_memory_mb:.0f} MB")
    print(f"[ATLAS] Selected profile: {template.name} - {template.description}")

    return replace(template)


# ============================================================================
# Configuration Validation
# ============================================================================

class ConfigurationError(Exception):
    """Raised when configuration parameters are incompatible or invalid."""
    pass


def validate_configs(
    model_config: HighResModelConfig,
    kernel_config: KernelConfig,
    sampler_config: SamplerConfig,
    conditioning_config: Optional[ConditioningConfig] = None,
    gpu_profile: Optional[GPUProfile] = None
) -> List[str]:
    """
    Validate configuration compatibility and return list of issues/warnings.

    Args:
        model_config: Model architecture configuration
        kernel_config: Kernel operator configuration
        sampler_config: Sampling configuration
        conditioning_config: Optional conditioning configuration
        gpu_profile: Optional GPU memory profile

    Returns:
        List of warning/error messages (empty if all valid)
    """
    errors: List[str] = []
    warnings_list: List[str] = []

    # Check model-conditioning compatibility
    if conditioning_config and conditioning_config.use_clip:
        clip_name = conditioning_config.clip_model or ""
        normalized_name = clip_name.lower()
        expected_context_dim = None
        for key, dim in _CLIP_MODEL_CONTEXT_DIMS.items():
            if key in normalized_name:
                expected_context_dim = dim
                break
        if expected_context_dim is None:
            # Default to CLIP-L like dimensions if we cannot infer
            expected_context_dim = 768
            warnings_list.append(
                "Could not infer CLIP context dimension from"
                f" clip_model='{conditioning_config.clip_model}'."
                " Assuming 768."
            )
        if model_config.context_dim != expected_context_dim:
            errors.append(
                f"Model context_dim={model_config.context_dim} doesn't match "
                f"CLIP output={expected_context_dim}. Update the model or conditioning config."
            )

    # Check kernel epsilon is reasonable
    if kernel_config.epsilon > 1.0:
        warnings_list.append(
            f"kernel_config.epsilon={kernel_config.epsilon} is very large. "
            "This may cause over-smoothing. Typical range: 0.001 - 0.1"
        )
    if kernel_config.epsilon < 1e-6:
        warnings_list.append(
            f"kernel_config.epsilon={kernel_config.epsilon} is very small. "
            "This may cause numerical instability. Typical range: 0.001 - 0.1"
        )

    # Check RFF features are reasonable
    if kernel_config.rff_features < 512:
        warnings_list.append(
            f"kernel_config.rff_features={kernel_config.rff_features} is low. "
            "Recommendation: >= 1024 for good approximation quality."
        )

    # Check memory compatibility
    if gpu_profile:
        estimated_memory = _estimate_memory_usage(
            gpu_profile=gpu_profile,
            sampler_config=sampler_config,
            conditioning_config=conditioning_config,
            model_config=model_config,
        )

        if estimated_memory > gpu_profile.memory_mb * 0.9:  # Use 90% threshold
            errors.append(
                f"Estimated memory usage ({estimated_memory:.0f} MB) may exceed "
                f"GPU capacity ({gpu_profile.memory_mb} MB). Consider reducing batch_size."
            )

    # Check hierarchical sampling compatibility
    if sampler_config.hierarchical_sampling and sampler_config.memory_efficient:
        warnings_list.append(
            "hierarchical_sampling=True with memory_efficient=True may be redundant. "
            "Hierarchical sampling already optimizes memory usage."
        )

    if errors:
        raise ValueError("\n".join(errors))

    return warnings_list


# ============================================================================
# Simplified Sampler Factory
# ============================================================================

class EasySampler:
    """
    Simplified high-level interface for ATLAS sampling.

    This class wraps the complex ATLAS sampling pipeline and provides
    a user-friendly API for non-experts.
    """

    def __init__(
        self,
        sampler: AdvancedHierarchicalDiffusionSampler,
        profile: GPUProfile,
        device: torch.device,
        model_config: "HighResModelConfig",
    ):
        self.sampler = sampler
        self.profile = profile
        self.device = device
        self.model_config = model_config
        self._clip_enabled = profile.enable_clip

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        negative_prompts: Optional[Union[str, List[str]]] = None,
        num_samples: int = 1,
        timesteps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples using ATLAS.

        Args:
            prompts: Text prompts for conditional generation (requires CLIP)
                    If None, generates unconditional samples
            negative_prompts: Negative text prompts for classifier-free guidance (requires CLIP)
                            Used to specify what to avoid in generation
            num_samples: Number of samples to generate per prompt
            timesteps: Number of diffusion steps (higher = better quality, slower)
                      Recommendation: 25 (fast), 50 (balanced), 100 (high quality)
            guidance_scale: Classifier-free guidance strength (1.0 = no guidance)
                           Recommendation: 7.5 for text-to-image
            seed: Random seed for reproducibility
            return_intermediates: If True, returns (samples, intermediate_steps)

        Returns:
            Generated samples as tensor of shape (num_samples, C, H, W)
            or tuple (samples, intermediates) if return_intermediates=True

        Raises:
            ValueError: If prompts provided but CLIP is disabled
            RuntimeError: If generation fails due to OOM or other errors
        """
        # Validate inputs
        if prompts is not None and not self._clip_enabled:
            raise ValueError(
                f"Text conditioning requires CLIP, but it's disabled in {self.profile.name} profile. "
                f"Use a higher memory profile (>=8GB) or set prompts=None for unconditional generation."
            )

        if negative_prompts is not None and prompts is None:
            raise ValueError(
                "negative_prompts require prompts to be specified. "
                "Provide positive prompts or set negative_prompts=None."
            )

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Prepare prompts
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]

            # Handle negative prompts
            if negative_prompts is not None:
                if isinstance(negative_prompts, str):
                    negative_prompts = [negative_prompts]

            # Setup conditioning
            if not hasattr(self.sampler, "prepare_conditioning_from_prompts"):
                raise RuntimeError(
                    "Sampler does not support prompt-based conditioning. "
                    "Ensure you are using AdvancedHierarchicalDiffusionSampler."
                )
            try:
                self.sampler.prepare_conditioning_from_prompts(
                    prompts=prompts,
                    negative_prompts=negative_prompts,
                    guidance_scale=guidance_scale
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to prepare CLIP conditioning: {e}\n"
                    "Ensure CLIP model is installed: pip install open-clip-torch"
                )

        # Determine total samples needed
        total_samples = len(prompts) * num_samples if prompts else num_samples
        batch_size = min(self.profile.batch_size, total_samples)

        # Calculate latent shape based on model's downsampling factor
        downsample = self.model_config.latent_downsampling_factor
        if self.profile.resolution % downsample != 0:
            raise ValueError(
                f"Resolution {self.profile.resolution} must be divisible by "
                f"latent_downsampling_factor {downsample}"
            )
        latent_size = self.profile.resolution // downsample

        # Generate samples in batches
        print(f"[ATLAS] Generating {total_samples} samples with {timesteps} steps...")
        print(f"[ATLAS] Using batch_size={batch_size}, resolution={self.profile.resolution}")

        all_samples: List[torch.Tensor] = []
        all_intermediates = [] if return_intermediates else None
        samples_generated = 0

        # Loop to generate all requested samples
        while samples_generated < total_samples:
            remaining = total_samples - samples_generated
            current_batch_size = min(batch_size, remaining)
            attempt_batch = current_batch_size

            while True:
                shape = (attempt_batch, 4, latent_size, latent_size)
                try:
                    result = self.sampler.sample(
                        shape=shape,
                        timesteps=timesteps,
                        return_intermediates=return_intermediates,
                    )
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    oom_error = "out of memory" in message or "cuda out of memory" in message
                    if oom_error and attempt_batch > 1:
                        attempt_batch = max(1, attempt_batch // 2)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                                torch.cuda.reset_peak_memory_stats()
                        print(
                            f"[ATLAS] CUDA OOM detected. Retrying with batch_size={attempt_batch} "
                            f"(profile={self.profile.name})"
                        )
                        continue
                    raise

            if attempt_batch < current_batch_size:
                batch_size = attempt_batch  # use the reduced batch size for future iterations

            # Handle return format
            if return_intermediates:
                batch_samples, batch_intermediates = result  # type: ignore[assignment]
                all_samples.append(batch_samples)
                all_intermediates.append(batch_intermediates)  # type: ignore[arg-type]
            else:
                all_samples.append(result)  # type: ignore[arg-type]

            samples_generated += attempt_batch

            if total_samples > batch_size:
                print(f"[ATLAS] Generated {samples_generated}/{total_samples} samples...")

        # Concatenate all batches
        samples = torch.cat(all_samples, dim=0)

        # Report memory usage
        peak_memory = get_peak_memory_mb()
        print(f"[ATLAS] Generation complete! Peak memory: {peak_memory:.1f} MB")

        if return_intermediates:
            if len(all_intermediates) == 1:
                intermediates = all_intermediates[0]
            else:
                num_steps = len(all_intermediates[0])
                intermediates = []
                for step_idx in range(num_steps):
                    step_tensors = [
                        batch_intermediates[step_idx] for batch_intermediates in all_intermediates
                    ]
                    intermediates.append(torch.cat(step_tensors, dim=0))
            return samples, intermediates
        return samples

    def clear_cache(self):
        """Clear kernel operator cache to free memory."""
        self.sampler.clear_kernel_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[ATLAS] Cleared kernel cache and GPU memory")
        else:
            print("[ATLAS] Cleared kernel cache")

    def set_training_mode(self, mode: bool) -> None:
        """Expose score model training/eval control."""
        self.sampler.set_model_training_mode(mode)


def _disable_clip_conditioning(
    profile: GPUProfile,
    conditioning_config: Optional[ConditioningConfig],
    sampler: AdvancedHierarchicalDiffusionSampler,
) -> Optional[ConditioningConfig]:
    """Disable CLIP across profile, configs, and sampler in a single place."""

    profile.enable_clip = False
    if conditioning_config is not None:
        conditioning_config.use_clip = False
        conditioning_config.context_dim = 0

    if hasattr(sampler, "score_model"):
        setattr(sampler.score_model, "use_context", False)
        if hasattr(sampler.score_model, "conditioning_config"):
            sampler.score_model.conditioning_config.use_clip = False

    return None


def _validate_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint_weights: Mapping[str, torch.Tensor],
) -> Tuple[List[str], List[str], List[str]]:
    """Compare model/state_dict structure before loading for clearer errors."""

    model_state = model.state_dict()
    missing: List[str] = []
    mismatched: List[str] = []
    unexpected = sorted(set(checkpoint_weights.keys()) - set(model_state.keys()))

    for key, tensor in model_state.items():
        ckpt_tensor = checkpoint_weights.get(key)
        if ckpt_tensor is None:
            missing.append(key)
            continue
        if ckpt_tensor.shape != tensor.shape:
            mismatched.append(
                f"{key}: expected {tuple(tensor.shape)}, got {tuple(ckpt_tensor.shape)}"
            )

    return missing, mismatched, unexpected


def create_sampler(
    checkpoint: Optional[Union[str, Path]] = None,
    gpu_memory: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    resolution: Optional[int] = None,
    batch_size: Optional[int] = None,
    enable_clip: Optional[bool] = None,
    **kwargs
) -> EasySampler:
    """
    Create a simple-to-use ATLAS sampler with automatic configuration.

    This is the main entry point for non-experts. It automatically detects
    your GPU, configures memory settings, and provides sensible defaults.

    Args:
        checkpoint: Path to pretrained model checkpoint (optional)
                   If None, creates a new model (requires training)
        gpu_memory: GPU memory profile ("6GB", "8GB", "12GB", "16GB", "24GB", "32GB")
                   If None, automatically detects your GPU
        device: Device to use ("cuda", "cpu", or torch.device)
               If None, uses CUDA if available
        resolution: Override resolution from profile (512, 1024)
        batch_size: Override batch size from profile
        enable_clip: Override CLIP setting from profile
        **kwargs: Additional arguments for advanced users

    Returns:
        EasySampler: Ready-to-use sampler instance

    Example:
        >>> import atlas.easy_api as atlas
        >>> sampler = atlas.create_sampler(gpu_memory="8GB")
        >>> images = sampler.generate(prompts=["a red car"], num_samples=4)
    """
    # Device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type == "cpu":
        warnings.warn(
            "Using CPU for ATLAS is very slow. For good performance, use a CUDA GPU."
        )

    # Select GPU profile
    if gpu_memory is None:
        # For CPU, use conservative profile without trying to detect GPU
        if device.type == "cpu":
            profile = GPU_PROFILES["6GB"]
            print(f"[ATLAS] Using CPU with profile: {profile.name} - {profile.description}")
        else:
            profile = detect_gpu_profile()
    else:
        if gpu_memory not in GPU_PROFILES:
            raise ValueError(
                f"Unknown gpu_memory='{gpu_memory}'. "
                f"Valid options: {list(GPU_PROFILES.keys())}"
            )
        profile = replace(GPU_PROFILES[gpu_memory])
        print(f"[ATLAS] Using manual profile: {profile.name} - {profile.description}")

    # Work with a copy so mutations do not leak back into the global presets
    profile = replace(profile)

    # Detect hardware capabilities and apply optimal settings
    from atlas.utils.hardware import (
        detect_hardware_capabilities,
        enable_optimal_precision,
        gate_expensive_feature,
    )

    hw_caps = detect_hardware_capabilities()

    # Print hardware info if verbose
    if kwargs.get("verbose", False):
        from atlas.utils.hardware import print_hardware_info
        print_hardware_info()

    # Override profile settings if specified
    if resolution is not None:
        profile.resolution = resolution
    if batch_size is not None:
        profile.batch_size = batch_size
    if enable_clip is not None:
        profile.enable_clip = enable_clip

    # Estimate free memory for the selected device
    available_memory_mb = profile.memory_mb
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(device_index)
        free_bytes, total_bytes = safe_cuda_mem_get_info(device_index)
        available_memory_mb = min(available_memory_mb, free_bytes / (1024 ** 2))

        # Enable optimal precision for hardware
        precision_mode = enable_optimal_precision()
        if kwargs.get("verbose", False):
            print(f"[ATLAS] Enabled precision mode: {precision_mode}")

        # Gate CUDA graphs based on hardware
        if "enable_cuda_graphs" in kwargs:
            allowed, reason = gate_expensive_feature(
                "CUDA Graphs",
                memory_threshold_gb=8.0,
                require_cuda_graphs=True,
            )
            if not allowed and kwargs["enable_cuda_graphs"]:
                warnings.warn(f"CUDA graphs requested but {reason}. Disabling.", UserWarning)
                kwargs["enable_cuda_graphs"] = False

    # Create configurations
    conditioning_config = ConditioningConfig(
        use_clip=profile.enable_clip,
        clip_model="ViT-L-14",
        context_dim=768 if profile.enable_clip else 0,
        guidance_scale=7.5,
    )

    model_config = HighResModelConfig(
        in_channels=4,
        out_channels=4,
        latent_downsampling_factor=8,  # Standard VAE downsampling
        base_channels=192,
        channel_mult=(1, 2, 4, 4) if profile.resolution >= 1024 else (1, 2, 2, 4),
        attention_levels=(1, 2),
        time_emb_dim=768,
        conditional=profile.enable_clip,
        context_dim=conditioning_config.context_dim,
        num_res_blocks=2,
        conditioning=conditioning_config,
    )

    # Adjust kernel config for hardware
    rff_features = 2048
    if hw_caps.device_type == "cpu":
        # CPU: reduce features for reasonable performance
        rff_features = 512
        if kwargs.get("verbose", False):
            print("[ATLAS] CPU detected: reducing RFF features to 512")
    elif hw_caps.free_memory_gb < 12.0:
        # Limited GPU memory: reduce features
        rff_features = 1024
        if kwargs.get("verbose", False):
            print(f"[ATLAS] Limited memory ({hw_caps.free_memory_gb:.1f}GB): reducing RFF features to 1024")

    # Select optimal kernel solver for hardware
    from atlas.utils.hardware import select_optimal_kernel_solver
    optimal_solver = select_optimal_kernel_solver(profile.resolution, profile.batch_size)

    # Use profile's solver unless auto
    solver_type = profile.kernel_solver if profile.kernel_solver != "auto" else optimal_solver

    kernel_config = KernelConfig(
        solver_type=solver_type,
        epsilon=0.01,
        rff_features=rff_features,
        n_landmarks=100,
        max_kernel_cache_size=profile.kernel_cache_size,
    )

    bf16_capable = (
        profile.supports_bf16
        and device.type == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    if bf16_capable:
        torch.set_float32_matmul_precision("high")

    sampler_overrides = {
        key: kwargs.pop(key)
        for key in (
            "enable_cuda_graphs",
            "cuda_graph_warmup_iters",
            "tile_size",
            "tile_stride",
            "tile_overlap",
            "tile_blending",
        )
        if key in kwargs
    }

    # Use hardware-detected precision capabilities
    use_mixed_precision = hw_caps.use_mixed_precision or bf16_capable

    sampler_config = SamplerConfig(
        sb_iterations=3,
        error_tolerance=1e-4,
        use_linear_solver=True,
        hierarchical_sampling=True,
        memory_efficient=True,
        use_mixed_precision=use_mixed_precision,
        memory_threshold_mb=available_memory_mb,
        auto_tuning=profile.auto_tune,
    )

    # Disable CUDA graphs on CPU or incompatible hardware
    if "enable_cuda_graphs" not in sampler_overrides and not hw_caps.cuda_graphs_supported:
        if kwargs.get("verbose", False):
            print("[ATLAS] CUDA graphs not supported on this hardware, keeping disabled")

    if sampler_overrides:
        sampler_config = sampler_config.with_overrides(**sampler_overrides)

    sampler_config.max_cached_batch_size = min(
        sampler_config.max_cached_batch_size, profile.batch_size
    )

    # Disable conditioning config if CLIP is not requested
    clip_conditioning_config = conditioning_config if profile.enable_clip else None

    # Validate configuration
    warnings_list = validate_configs(
        model_config=model_config,
        kernel_config=kernel_config,
        sampler_config=sampler_config,
        conditioning_config=clip_conditioning_config,
        gpu_profile=profile,
    )

    if warnings_list:
        print("[ATLAS] Configuration warnings:")
        for warning in warnings_list:
            print(f"  - {warning}")

    # Create model
    print(f"[ATLAS] Creating model with resolution={profile.resolution}, channels={model_config.base_channels}")
    model = HighResLatentScoreModel(config=model_config).to(device)
    model.eval()

    # Load checkpoint if provided
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        print(f"[ATLAS] Loading checkpoint: {checkpoint}")
        try:
            state_dict = _safe_torch_load(checkpoint, map_location=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint}: {e}\n"
                f"Ensure the checkpoint file is valid and compatible with PyTorch {torch.__version__}"
            ) from e

        # Handle different checkpoint formats
        if "model" in state_dict:
            checkpoint_weights = state_dict["model"]
        elif "ema_model" in state_dict:
            checkpoint_weights = state_dict["ema_model"]
        else:
            checkpoint_weights = state_dict

        missing, mismatched, unexpected = _validate_checkpoint_weights(
            model, checkpoint_weights
        )
        if unexpected:
            warnings.warn(
                "Checkpoint contains unexpected parameters (showing up to 5): "
                + ", ".join(unexpected[:5]),
                UserWarning,
            )
        if missing or mismatched:
            summary = []
            if missing:
                summary.append(
                    "Missing keys: " + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
                )
            if mismatched:
                summary.append(
                    "Shape mismatches: "
                    + "; ".join(mismatched[:3])
                    + ("..." if len(mismatched) > 3 else "")
                )
            raise RuntimeError(
                "Checkpoint architecture mismatch detected.\n" + "\n".join(summary)
            )

        try:
            model.load_state_dict(checkpoint_weights)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from checkpoint {checkpoint}: {e}\n"
                f"The checkpoint may be incompatible with the current model configuration."
            ) from e

    # Create sampler
    print("[ATLAS] Initializing sampler...")
    sampler = AdvancedHierarchicalDiffusionSampler(
        score_model=model,
        noise_schedule=karras_noise_schedule,
        device=device,
        kernel_config=kernel_config,
        sampler_config=sampler_config,
    )

    # Setup CLIP conditioning if enabled
    if profile.enable_clip and clip_conditioning_config:
        try:
            from atlas.conditioning.clip_interface import CLIPConditioningInterface

            clip_conditioner = CLIPConditioningInterface(
                config=conditioning_config,
                device=device,
            )
            sampler.set_conditioner(clip_conditioner)
            print(f"[ATLAS] CLIP conditioning enabled: {clip_conditioning_config.clip_model}")
        except ImportError:
            warnings.warn(
                "Could not import CLIP. Text conditioning disabled.\n"
                "Install with: pip install open-clip-torch"
            )
            clip_conditioning_config = _disable_clip_conditioning(
                profile=profile,
                conditioning_config=clip_conditioning_config,
                sampler=sampler,
            )
        except (RuntimeError, OSError) as exc:
            warnings.warn(
                f"Failed to initialize CLIP conditioning ({exc}). "
                "Continuing without text guidance."
            )
            clip_conditioning_config = _disable_clip_conditioning(
                profile=profile,
                conditioning_config=clip_conditioning_config,
                sampler=sampler,
            )

    print("[ATLAS] Sampler ready! Configuration summary:")
    print(f"  - Resolution: {profile.resolution}x{profile.resolution}")
    print(f"  - Max batch size: {profile.batch_size}")
    print(f"  - Mixed precision: {profile.use_mixed_precision}")
    print(f"  - CLIP enabled: {profile.enable_clip}")
    print(f"  - Kernel solver: {profile.kernel_solver}")

    return EasySampler(sampler=sampler, profile=profile, device=device, model_config=model_config)


# ============================================================================
# Quick Start Functions
# ============================================================================

def quick_sample(
    checkpoint: str,
    num_samples: int = 4,
    prompts: Optional[List[str]] = None,
    gpu_memory: str = "auto",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    One-liner for quick sampling without any configuration.

    Args:
        checkpoint: Path to model checkpoint
        num_samples: Number of samples to generate
        prompts: Optional text prompts (requires CLIP)
        gpu_memory: GPU memory profile or "auto" for auto-detection
        seed: Random seed for reproducibility

    Returns:
        Generated samples as tensor

    Example:
        >>> images = atlas.quick_sample("model.pt", num_samples=4, prompts=["a cat"])
    """
    sampler = create_sampler(
        checkpoint=checkpoint,
        gpu_memory=None if gpu_memory == "auto" else gpu_memory,
    )

    return sampler.generate(
        prompts=prompts,
        num_samples=num_samples,
        seed=seed,
    )


def list_profiles():
    """Print all available GPU profiles with descriptions."""
    print("Available GPU Memory Profiles:")
    print("=" * 80)
    for name, profile in GPU_PROFILES.items():
        print(f"\n{name}:")
        print(f"  Description: {profile.description}")
        print(f"  Resolution: {profile.resolution}x{profile.resolution}")
        print(f"  Batch size: {profile.batch_size}")
        print(f"  Mixed precision: {profile.use_mixed_precision}")
        print(f"  CLIP enabled: {profile.enable_clip}")
        print(f"  Kernel solver: {profile.kernel_solver}")

