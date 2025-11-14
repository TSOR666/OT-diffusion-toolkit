# GPU/CPU Behavior Guide

## Overview

ATLAS automatically detects hardware capabilities and adjusts its behavior accordingly. This document explains expected behavior on different hardware configurations.

## GPU Detection and Fallbacks

### Automatic Detection

```python
from atlas.easy_api import create_sampler

# Auto-detects GPU memory and selects appropriate profile
sampler = create_sampler(checkpoint="model.pt", gpu_memory="auto")
```

**Detection Priority:**
1. Check `torch.cuda.is_available()`
2. Query free GPU memory via `torch.cuda.mem_get_info()`
3. Detect GPU name and compute capability
4. Check for BF16 support (`torch.cuda.is_bf16_supported()`)
5. Enable TF32 if available (Ampere+)

### CPU Fallback Behavior

When no GPU is detected:
- **Kernel Solver**: Defaults to RFF (2048 features → 512 for CPU)
- **Batch Size**: Reduced to 1-2
- **Mixed Precision**: Disabled (FP32 only)
- **CUDA Graphs**: Disabled
- **FFT Kernels**: Work but slower than RFF on CPU

**Performance**: Expect 5-10x slower than GPU for typical workloads.

## Precision Modes

### Float32 (FP32) - Universal
- **Availability**: Always available
- **Accuracy**: Highest precision
- **Speed**: Baseline
- **Memory**: Highest usage
- **Use Case**: CPU, legacy GPUs, debugging

### Float16 (FP16) - Mixed Precision
- **Availability**: CUDA GPUs (compute 5.3+)
- **Accuracy**: Good (watch for underflow)
- **Speed**: 2-3x faster on Tensor Cores
- **Memory**: 50% reduction
- **Use Case**: Volta, Turing GPUs (V100, RTX 20xx)

**Enabled by default on compatible GPUs:**
```python
SamplerConfig(use_mixed_precision=True)
```

### BFloat16 (BF16) - Modern Mixed Precision
- **Availability**: Ampere+ (A100, RTX 30xx/40xx/50xx)
- **Accuracy**: Better than FP16 (wider exponent range)
- **Speed**: 2-3x faster with TF32 acceleration
- **Memory**: 50% reduction
- **Use Case**: Ampere, Ada, Hopper GPUs

**Auto-enabled on supported hardware:**
```python
# Automatically checks torch.cuda.is_bf16_supported()
sampler_config = SamplerConfig(use_mixed_precision=True)
```

### TensorFloat32 (TF32) - Automatic Acceleration
- **Availability**: Ampere+ (RTX 30xx/A100+)
- **Accuracy**: FP32 range, reduced mantissa
- **Speed**: 8x faster matmul throughput
- **Memory**: Same as FP32
- **Automatic**: Enabled by default, transparent to user

**Manually control:**
```python
torch.backends.cuda.matmul.allow_tf32 = True   # Default on Ampere+
torch.backends.cudnn.allow_tf32 = True
```

## Kernel Operator Behavior

### Direct Kernel (`solver_type="direct"`)
- **Complexity**: O(n²) memory, O(n²) compute
- **Best For**: Batches < 500 samples
- **GPU**: Fast with cuBLAS optimization
- **CPU**: Slow, not recommended for n > 100

### FFT Kernel (`solver_type="fft"`)
- **Complexity**: O(r^d) memory, O(r^d log r) compute
- **Best For**: Grid-structured data (images)
- **GPU**: Very fast with cuFFT
- **CPU**: Slower, but still efficient for small images
- **Memory**: Precomputes FFT kernels (~64MB per scale @ 1024px)

### RFF Kernel (`solver_type="rff"`)
- **Complexity**: O(d × features) memory, O(n × features) compute
- **Best For**: High-dimensional point clouds, general purpose
- **GPU**: Highly optimized, scales to large batches
- **CPU**: Best CPU option (2048 → 512 features automatically)
- **Adaptive**: Feature count scales with resolution

### Nyström Kernel (`solver_type="nystrom"`)
- **Complexity**: O(m²) memory where m=landmarks
- **Best For**: Very tight memory budgets
- **GPU**: Good for m < 200
- **CPU**: Reasonable for small problems
- **Quality**: Lower than RFF for same memory

## Memory Management by GPU Tier

### 6-8 GB (RTX 3050, 3060, 4070)
```python
GPUProfile(
    resolution=512,
    batch_size=2,
    use_mixed_precision=True,  # FP16
    kernel_solver="rff",
    rff_features=1024,         # Reduced
    kernel_cache_size=4,
)
```
**Behavior:**
- Automatic batch size reduction on OOM
- Gradient checkpointing enabled
- Small kernel cache
- Tiling disabled (too slow)

### 12-16 GB (RTX 3080, 4080, 4090)
```python
GPUProfile(
    resolution=1024,
    batch_size=4,
    use_mixed_precision=True,  # BF16 if supported
    kernel_solver="auto",
    rff_features=2048,
    kernel_cache_size=16,
    enable_cuda_graphs=True,   # 10-30% speedup
)
```
**Behavior:**
- Full feature set enabled
- CUDA graphs for repeated shapes
- TF32 acceleration
- Adaptive kernel selection

### 24-32 GB (4090 Workstation, A5000)
```python
GPUProfile(
    resolution=1536,
    batch_size=8,
    use_mixed_precision=True,
    kernel_solver="auto",
    rff_features=2048,
    kernel_cache_size=32,
    enable_cuda_graphs=True,
    tile_size=512,             # Ultra-high-res
    tile_overlap=0.125,
)
```
**Behavior:**
- Native 2K support (1536-2048px)
- Tiling for >2K images
- Large kernel cache
- Maximum performance optimizations

## CUDA Graphs

### Requirements
- **CUDA**: 11.0+
- **PyTorch**: 2.0+
- **Memory**: 16GB+ recommended
- **Shapes**: Must be static (fixed batch/resolution)

### Behavior
**Enabled:**
```python
SamplerConfig(
    enable_cuda_graphs=True,
    cuda_graph_warmup_iters=2,  # Warmup runs before capture
)
```

**When Active:**
- First call: Captures computation graph (slow)
- Subsequent calls: Replays graph (10-30% faster)
- Cache: Stores up to 16 graphs (LRU eviction)

**Disabled Automatically When:**
- CPU mode
- CUDA < 11.0
- Variable input shapes
- Out of memory

### Verification
```python
import torch

print(f"CUDA Graphs Supported: {torch.cuda.is_available() and int(torch.version.cuda.split('.')[0]) >= 11}")
```

## Tiling for Ultra-High Resolution

### When Tiling Activates
- **Resolution**: > 1536px (configurable)
- **Memory**: Insufficient for full-res batch
- **Automatic**: Tiles split when OOM detected

### Configuration
```python
SamplerConfig(
    tile_size=512,              # Tile dimensions
    tile_stride=448,            # 512 * (1 - 0.125) overlap
    tile_overlap=0.125,         # 12.5% overlap
    tile_blending="hann",       # Smooth blending (hann/linear/none)
)
```

### Behavior
- **Splitting**: Image divided into overlapping tiles
- **Processing**: Each tile processed independently
- **Blending**: Hann window applied to tile edges
- **Slowdown**: 2-4x slower than full-batch (depends on overlap)

**CPU Warning**: Tiling on CPU is very slow (10-30x). Use reduced resolution instead.

## Performance Expectations

### Typical Inference Times (RTX 4090)

| Resolution | Batch | Steps | FP32 | BF16 + TF32 | CUDA Graphs |
|------------|-------|-------|------|-------------|-------------|
| 512×512    | 4     | 50    | 45s  | 18s         | 13s         |
| 1024×1024  | 2     | 50    | 90s  | 35s         | 25s         |
| 1536×1536  | 1     | 50    | 180s | 70s         | 50s         |
| 2048×2048  | 1     | 50    | 300s | 120s        | 85s (tiled) |

### Training Throughput (A100 80GB)

| Config | Batch | Resolution | Samples/sec | Mixed Precision |
|--------|-------|------------|-------------|-----------------|
| Base   | 32    | 256×256    | 8.5         | FP32            |
| Large  | 16    | 512×512    | 12.2        | BF16            |
| XL     | 8     | 1024×1024  | 4.8         | BF16 + Grad Checkpoint |

## Debugging Performance Issues

### Check Current Hardware State
```python
from atlas.utils.hardware import get_hardware_info

info = get_hardware_info()
print(f"Device: {info['device']}")
print(f"Mixed Precision: {info['mixed_precision']}")
print(f"TF32 Enabled: {info['tf32_enabled']}")
print(f"BF16 Supported: {info['bf16_supported']}")
print(f"Free Memory: {info['free_memory_gb']:.1f} GB")
```

### Common Issues

**Slow CPU Performance**
- ✅ Verify GPU is detected: `torch.cuda.is_available()`
- ✅ Check CUDA drivers installed
- ✅ Use RFF kernels (faster than Direct on CPU)

**OOM Errors**
- ✅ Reduce batch size
- ✅ Lower resolution
- ✅ Enable tiling for ultra-high-res
- ✅ Clear kernel cache: `sampler.clear_cache()`

**Slow GPU Performance**
- ✅ Enable mixed precision
- ✅ Enable CUDA graphs for fixed shapes
- ✅ Use RFF/FFT instead of Direct kernels
- ✅ Check GPU not thermal throttling

**BF16 Not Working**
- ✅ Verify Ampere+ GPU
- ✅ Check CUDA 11.8+ installed
- ✅ Ensure PyTorch 2.0+

## Best Practices

### For Training
1. **Use mixed precision**: BF16 on Ampere+, FP16 on older GPUs
2. **Enable TF32**: Automatic on Ampere+
3. **Gradient checkpointing**: For deep models on 12-16GB GPUs
4. **Profile first batch**: Find optimal batch size before full run

### For Inference
1. **CUDA graphs**: Enable for fixed-shape sampling (10-30% speedup)
2. **Kernel caching**: Reuse kernel operators across batches
3. **Batch processing**: Process multiple prompts together
4. **Memory monitoring**: Clear cache between different resolutions

### For CPU
1. **Use RFF kernels**: Faster than Direct/FFT on CPU
2. **Reduce features**: 512-1024 instead of 2048
3. **Lower resolution**: 256-512px practical range
4. **Multi-core**: Ensure OpenMP/MKL threading enabled
