"""Hardware capability detection and resource awareness utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


_WARNED_MEM_INFO_FALLBACK = False


@dataclass
class HardwareCapabilities:
    """Hardware capabilities and feature support."""

    # Device information
    device_type: str  # "cuda", "cpu", "mps"
    device_name: str
    compute_capability: Optional[Tuple[int, int]] = None

    # Memory information
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0

    # Precision support
    bf16_supported: bool = False
    fp16_supported: bool = False
    tf32_available: bool = False
    tf32_enabled: bool = False

    # Advanced features
    cuda_graphs_supported: bool = False
    cuda_version: Optional[str] = None

    # Recommended settings
    use_mixed_precision: bool = False
    recommended_precision: str = "fp32"
    max_recommended_batch_size: int = 1


def _bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024 ** 3)


def safe_cuda_mem_get_info(device_index: Optional[int] = None) -> Tuple[int, int]:
    """
    Return (free_bytes, total_bytes) even on CUDA builds that lack mem_get_info.
    Falls back to allocator statistics when necessary.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")

    target_index = device_index
    if target_index is None:
        target_index = torch.cuda.current_device()

    if hasattr(torch.cuda, "mem_get_info"):
        try:
            if device_index is None:
                return torch.cuda.mem_get_info()
            return torch.cuda.mem_get_info(target_index)
        except RuntimeError:
            pass

    total = int(torch.cuda.get_device_properties(target_index).total_memory)
    try:
        allocated = int(torch.cuda.memory_allocated(target_index))
        free = max(total - allocated, 0)
    except Exception:
        free = total

    global _WARNED_MEM_INFO_FALLBACK
    if not _WARNED_MEM_INFO_FALLBACK:
        warnings.warn(
            "torch.cuda.mem_get_info is unavailable; estimating free memory from allocator stats",
            RuntimeWarning,
        )
        _WARNED_MEM_INFO_FALLBACK = True

    return free, total


def detect_hardware_capabilities() -> HardwareCapabilities:
    """
    Detect hardware capabilities and return structured information.

    Returns:
        HardwareCapabilities with detected features
    """
    if torch.cuda.is_available():
        return _detect_cuda_capabilities()
    else:
        return _detect_cpu_capabilities()


def _detect_cuda_capabilities() -> HardwareCapabilities:
    """Detect CUDA GPU capabilities."""
    device_props = torch.cuda.get_device_properties(0)
    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)

    # Memory information
    free_mem, total_mem = safe_cuda_mem_get_info(0)
    total_memory_gb = _bytes_to_gb(total_mem)
    free_memory_gb = _bytes_to_gb(free_mem)

    # Precision support
    bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    fp16_supported = compute_cap[0] >= 5  # Maxwell and later

    # TF32 support (Ampere and later)
    tf32_available = compute_cap[0] >= 8
    tf32_enabled = tf32_available and torch.backends.cuda.matmul.allow_tf32

    # CUDA graphs (CUDA 11.0+)
    cuda_version = torch.version.cuda
    cuda_graphs_supported = False
    if cuda_version is not None:
        try:
            major = int(str(cuda_version).split(".")[0])
            cuda_graphs_supported = major >= 11
        except (ValueError, IndexError, AttributeError):
            cuda_graphs_supported = False

    # Recommended precision
    if bf16_supported and tf32_available:
        recommended_precision = "bf16"
        use_mixed_precision = True
    elif fp16_supported:
        recommended_precision = "fp16"
        use_mixed_precision = True
    else:
        recommended_precision = "fp32"
        use_mixed_precision = False

    # Batch size recommendation based on memory
    # Conservative batch size recommendation (documented as baseline)
    if free_memory_gb >= 24:
        max_batch_size = 16
    elif free_memory_gb >= 16:
        max_batch_size = 8
    elif free_memory_gb >= 12:
        max_batch_size = 4
    elif free_memory_gb >= 8:
        max_batch_size = 2
    else:
        max_batch_size = 1

    return HardwareCapabilities(
        device_type="cuda",
        device_name=device_name,
        compute_capability=compute_cap,
        total_memory_gb=total_memory_gb,
        free_memory_gb=free_memory_gb,
        bf16_supported=bf16_supported,
        fp16_supported=fp16_supported,
        tf32_available=tf32_available,
        tf32_enabled=tf32_enabled,
        cuda_graphs_supported=cuda_graphs_supported,
        cuda_version=cuda_version,
        use_mixed_precision=use_mixed_precision,
        recommended_precision=recommended_precision,
        max_recommended_batch_size=max_batch_size,
    )


def _detect_cpu_capabilities() -> HardwareCapabilities:
    """Detect CPU capabilities."""
    import platform

    device_name = f"{platform.processor()} ({platform.machine()})"

    return HardwareCapabilities(
        device_type="cpu",
        device_name=device_name,
        compute_capability=None,
        total_memory_gb=0.0,
        free_memory_gb=0.0,
        bf16_supported=False,
        fp16_supported=False,
        tf32_available=False,
        tf32_enabled=False,
        cuda_graphs_supported=False,
        cuda_version=None,
        use_mixed_precision=False,
        recommended_precision="fp32",
        max_recommended_batch_size=1,
    )


def get_hardware_info() -> Dict[str, Any]:
    """
    Get hardware information as a dictionary.

    Returns:
        Dictionary with hardware capabilities

    Example:
        >>> info = get_hardware_info()
        >>> print(f"Device: {info['device']}")
        >>> print(f"BF16: {info['bf16_supported']}")
    """
    caps = detect_hardware_capabilities()

    return {
        "device": caps.device_type,
        "device_name": caps.device_name,
        "compute_capability": caps.compute_capability,
        "total_memory_gb": caps.total_memory_gb,
        "free_memory_gb": caps.free_memory_gb,
        "bf16_supported": caps.bf16_supported,
        "fp16_supported": caps.fp16_supported,
        "tf32_available": caps.tf32_available,
        "tf32_enabled": caps.tf32_enabled,
        "cuda_graphs_supported": caps.cuda_graphs_supported,
        "cuda_version": caps.cuda_version,
        "mixed_precision": caps.use_mixed_precision,
        "recommended_precision": caps.recommended_precision,
        "max_batch_size": caps.max_recommended_batch_size,
    }


def print_hardware_info() -> None:
    """Print formatted hardware information."""
    info = get_hardware_info()

    print("\n" + "=" * 60)
    print("ATLAS Hardware Information")
    print("=" * 60)

    print(f"\nDevice Type: {info['device']}")
    print(f"Device Name: {info['device_name']}")

    if info['compute_capability'] is not None:
        major, minor = info['compute_capability']
        print(f"Compute Capability: {major}.{minor}")

    if info['total_memory_gb'] > 0:
        print(f"\nMemory:")
        print(f"  Total: {info['total_memory_gb']:.1f} GB")
        print(f"  Free:  {info['free_memory_gb']:.1f} GB")

    bool_to_text = lambda flag: "Yes" if flag else "No"

    print(f"\nPrecision Support:")
    print(f"  FP16:  {bool_to_text(info['fp16_supported'])}")
    print(f"  BF16:  {bool_to_text(info['bf16_supported'])}")
    print(f"  TF32:  {bool_to_text(info['tf32_available'])} (enabled: {info['tf32_enabled']})")

    print(f"\nAdvanced Features:")
    print(f"  CUDA Graphs: {bool_to_text(info['cuda_graphs_supported'])}")
    if info['cuda_version']:
        print(f"  CUDA Version: {info['cuda_version']}")

    print(f"\nRecommendations:")
    print(f"  Precision: {info['recommended_precision'].upper()}")
    print(f"  Mixed Precision: {'Enabled' if info['mixed_precision'] else 'Disabled'}")
    print(f"  Max Batch Size: {info['max_batch_size']}")

    print("=" * 60 + "\n")


def validate_feature_support(
    feature: str,
    required: bool = True,
    fallback_message: Optional[str] = None,
) -> bool:
    """
    Validate if a hardware feature is supported with appropriate warnings.

    Args:
        feature: Feature name ("bf16", "cuda_graphs", "tf32", etc.)
        required: If True, raise error; if False, warn and return False
        fallback_message: Custom message for fallback behavior

    Returns:
        True if supported, False otherwise

    Raises:
        RuntimeError: If feature required but not supported
    """
    caps = detect_hardware_capabilities()

    supported = {
        "bf16": caps.bf16_supported,
        "fp16": caps.fp16_supported,
        "tf32": caps.tf32_available,
        "cuda_graphs": caps.cuda_graphs_supported,
        "cuda": caps.device_type == "cuda",
        "gpu": caps.device_type in ("cuda", "mps"),
    }

    is_supported = supported.get(feature.lower(), False)

    if not is_supported:
        msg = fallback_message or f"{feature.upper()} not supported on this hardware"

        if required:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)

    return is_supported


def enable_optimal_precision() -> str:
    """
    Enable optimal precision settings for current hardware.

    Returns:
        String describing enabled precision mode

    Example:
        >>> precision = enable_optimal_precision()
        >>> print(f"Using: {precision}")
    """
    caps = detect_hardware_capabilities()

    if caps.device_type != "cuda":
        return "fp32 (CPU)"

    # Enable TF32 if available
    if caps.tf32_available:
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True

    # Set matmul precision for BF16
    if caps.bf16_supported and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        return "bf16 + tf32"

    # FP16 for older GPUs
    if caps.fp16_supported:
        return "fp16"

    return "fp32"


def gate_expensive_feature(
    feature_name: str,
    memory_threshold_gb: Optional[float] = None,
    min_compute_capability: Optional[Tuple[int, int]] = None,
    require_cuda_graphs: bool = False,
) -> Tuple[bool, str]:
    """
    Gate expensive features based on hardware capabilities.

    Args:
        feature_name: Name of feature for error messages
        memory_threshold_gb: Minimum free memory required
        min_compute_capability: Minimum CUDA compute capability (major, minor)
        require_cuda_graphs: Whether CUDA graphs are required

    Returns:
        (allowed, reason) tuple. If not allowed, reason explains why.

    Example:
        >>> allowed, reason = gate_expensive_feature(
        ...     "Large FFT kernels",
        ...     memory_threshold_gb=10.0,
        ...     min_compute_capability=(7, 0),
        ... )
        >>> if not allowed:
        ...     print(f"Disabled: {reason}")
    """
    caps = detect_hardware_capabilities()

    # Check memory
    if memory_threshold_gb is not None:
        if caps.free_memory_gb < memory_threshold_gb:
            return False, (
                f"{feature_name} requires {memory_threshold_gb:.1f}GB free memory, "
                f"but only {caps.free_memory_gb:.1f}GB available"
            )

    # Check compute capability
    if min_compute_capability is not None:
        if caps.compute_capability is None:
            return False, f"{feature_name} requires CUDA GPU"

        required_major, required_minor = min_compute_capability
        actual_major, actual_minor = caps.compute_capability

        if (actual_major, actual_minor) < (required_major, required_minor):
            return False, (
                f"{feature_name} requires compute capability "
                f"{required_major}.{required_minor}, "
                f"but GPU is {actual_major}.{actual_minor}"
            )

    # Check CUDA graphs
    if require_cuda_graphs and not caps.cuda_graphs_supported:
        return False, f"{feature_name} requires CUDA graphs (CUDA 11.0+)"

    return True, ""


def select_optimal_kernel_solver(
    resolution: int,
    batch_size: int,
) -> str:
    """
    Select optimal kernel solver based on hardware and problem size.

    Args:
        resolution: Image resolution (pixels per side)
        batch_size: Number of samples in batch

    Returns:
        Recommended solver type: "direct", "fft", "rff", or "nystrom"
    """
    caps = detect_hardware_capabilities()

    # CPU: prefer RFF
    if caps.device_type == "cpu":
        return "rff"

    # Grid-structured, GPU: use FFT
    if resolution >= 256:
        # Check if we have enough memory for FFT
        # Rough estimate: ~200MB per scale for 1024px
        required_mem_gb = (resolution / 1024) ** 2 * 0.6

        if caps.free_memory_gb >= required_mem_gb:
            return "fft"

    # Small batch: direct method
    if batch_size <= 500 and caps.free_memory_gb >= 4.0:
        return "direct"

    # Default: RFF (balanced)
    return "rff"


def adjust_config_for_hardware(
    config: dict,
    target_resolution: int,
) -> dict:
    """
    Adjust configuration parameters based on hardware capabilities.

    Args:
        config: Configuration dictionary to adjust
        target_resolution: Target resolution for generation

    Returns:
        Adjusted configuration dictionary
    """
    caps = detect_hardware_capabilities()

    # Adjust batch size
    if "batch_size" in config:
        config["batch_size"] = min(
            config["batch_size"],
            caps.max_recommended_batch_size,
        )

    # Adjust mixed precision
    if "use_mixed_precision" in config:
        config["use_mixed_precision"] = caps.use_mixed_precision

    # Disable CUDA graphs on incompatible hardware
    if "enable_cuda_graphs" in config:
        if not caps.cuda_graphs_supported:
            config["enable_cuda_graphs"] = False
            warnings.warn(
                "CUDA graphs not supported, disabling feature",
                UserWarning,
            )

    # Adjust RFF features for memory
    if "rff_features" in config and caps.free_memory_gb < 12.0:
        original = config["rff_features"]
        config["rff_features"] = min(original, 1024)

        if config["rff_features"] != original:
            warnings.warn(
                f"Reduced RFF features from {original} to {config['rff_features']} "
                f"due to limited memory ({caps.free_memory_gb:.1f}GB)",
                UserWarning,
            )

    # Enable tiling for ultra-high-res on limited memory
    if target_resolution > 1536 and caps.free_memory_gb < 20.0:
        if "tile_size" not in config or config["tile_size"] is None:
            config["tile_size"] = 512
            config["tile_overlap"] = 0.125
            config["tile_blending"] = "hann"

            warnings.warn(
                f"Enabled tiling for {target_resolution}px resolution "
                f"on {caps.free_memory_gb:.1f}GB GPU",
                UserWarning,
            )

    return config


# Convenience function for CLI tools
def check_requirements(
    min_memory_gb: Optional[float] = None,
    require_cuda: bool = False,
    require_bf16: bool = False,
) -> bool:
    """
    Check if system meets minimum requirements.

    Args:
        min_memory_gb: Minimum free GPU memory required
        require_cuda: Whether CUDA is required
        require_bf16: Whether BF16 support is required

    Returns:
        True if requirements met, False otherwise

    Raises:
        RuntimeError: If requirements not met
    """
    caps = detect_hardware_capabilities()

    if require_cuda and caps.device_type != "cuda":
        raise RuntimeError("CUDA GPU required but not available")

    if min_memory_gb and caps.free_memory_gb < min_memory_gb:
        raise RuntimeError(
            f"Requires {min_memory_gb}GB free GPU memory, "
            f"but only {caps.free_memory_gb:.1f}GB available"
        )

    if require_bf16 and not caps.bf16_supported:
        raise RuntimeError("BF16 precision required but not supported")

    return True

