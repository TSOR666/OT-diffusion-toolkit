# CUDA Graphs and Tiling Guide

## CUDA Graphs

### What Are CUDA Graphs?

CUDA graphs capture a sequence of GPU operations into a replayable graph structure. Instead of launching kernels individually (high CPU overhead), the entire sequence is replayed as a unit, providing 10-30% speedup.

### Requirements

- **CUDA**: 11.0 or later
- **PyTorch**: 2.0 or later
- **GPU Memory**: 16GB+ recommended
- **Input Shapes**: Must be static (same batch size and resolution)

### When to Use

✅ **Good Use Cases:**
- Repeated inference with same shape (batch production)
- Fixed-resolution generation (e.g., always 1024×1024)
- High-throughput scenarios

❌ **Bad Use Cases:**
- Variable batch sizes per call
- Different resolutions in same session
- Debugging (harder to trace)
- Limited GPU memory (graphs cache multiple shapes)

### Basic Setup

```python
from atlas.config import SamplerConfig
from atlas.easy_api import create_sampler

# Enable CUDA graphs in sampler config
sampler_config = SamplerConfig(
    enable_cuda_graphs=True,
    cuda_graph_warmup_iters=2,  # Warmup runs before graph capture
)

sampler = create_sampler(
    checkpoint="model.pt",
    gpu_memory="16GB",
    # Pass sampler config via kwargs
    enable_cuda_graphs=True,
    cuda_graph_warmup_iters=2,
)

# First call: slow (captures graph)
images1 = sampler.generate(prompts=["a cat"], num_samples=4, timesteps=50)

# Second call with SAME shape: fast (replays graph)
images2 = sampler.generate(prompts=["a dog"], num_samples=4, timesteps=50)

# Different shape: captures new graph
images3 = sampler.generate(prompts=["a bird"], num_samples=2, timesteps=50)
```

### Advanced Configuration

```python
from atlas.solvers import AdvancedHierarchicalDiffusionSampler
from atlas.utils.cuda_graphs import CUDAGraphModelWrapper

# Wrap score model with CUDA graph acceleration
model_with_graphs = CUDAGraphModelWrapper(
    model=score_model,
    warmup_iters=3,        # More warmup for stability
    max_cache_size=16,     # LRU cache for multiple shapes
)

sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model_with_graphs,  # Use wrapped model
    noise_schedule=karras_noise_schedule,
    device=device,
    kernel_config=kernel_config,
    sampler_config=sampler_config,
)
```

### Graph Cache Management

ATLAS uses an LRU (Least Recently Used) cache for CUDA graphs:

```python
# Default: caches up to 16 different shapes
CUDAGraphModelWrapper(model, max_cache_size=16)

# Reduce for memory-constrained scenarios
CUDAGraphModelWrapper(model, max_cache_size=4)

# Increase for many shape variations
CUDAGraphModelWrapper(model, max_cache_size=32)
```

**Memory Impact:**
- Each cached graph: ~100-500MB depending on model size
- 16 graphs × 200MB = ~3.2GB graph overhead
- Graphs evicted automatically when cache full

### Performance Expectations

| Configuration | Without Graphs | With Graphs | Speedup |
|---------------|----------------|-------------|---------|
| RTX 4090, 1024px, batch=4 | 35s | 25s | 1.4x |
| A100, 512px, batch=16 | 18s | 13s | 1.38x |
| V100, 1024px, batch=2 | 60s | 45s | 1.33x |

**Note**: Speedup varies by:
- Model size (larger = more benefit)
- Batch size (smaller batches benefit more from reduced overhead)
- Number of sampling steps

### Troubleshooting

**RuntimeError: CUDA out of memory**
- Reduce `max_cache_size`
- Use smaller batch sizes
- Disable graphs for variable shapes

**No speedup observed**
- Ensure using same shape repeatedly
- Check CUDA version ≥ 11.0
- Verify PyTorch 2.0+
- Monitor with: `torch.cuda.synchronize()` before timing

**Graphs not being used**
- Check: `sampler_config.enable_cuda_graphs == True`
- Verify no CPU fallback: `device.type == 'cuda'`
- Look for warning messages in logs

---

## Tiling for Ultra-High Resolution

### What Is Tiling?

Tiling splits large images into overlapping patches (tiles), processes each independently, then blends them back together. This trades speed for memory efficiency.

### When to Use

✅ **Enable Tiling When:**
- Resolution > 1536px
- GPU memory < 24GB
- Batch size = 1 required
- Quality matters more than speed

❌ **Avoid Tiling When:**
- Sufficient GPU memory available
- Speed is critical
- Running on CPU (extremely slow)

### Basic Configuration

```python
from atlas.config import SamplerConfig

sampler_config = SamplerConfig(
    tile_size=512,              # Size of each tile (pixels)
    tile_stride=448,            # Distance between tiles (512 - 64 overlap)
    tile_overlap=0.125,         # 12.5% overlap (alternative to stride)
    tile_blending="hann",       # Blending method
)
```

**Tile Overlap Calculation:**
```python
# If using tile_overlap:
stride = tile_size * (1.0 - tile_overlap)

# If using tile_stride directly:
overlap = 1.0 - (tile_stride / tile_size)
```

### Blending Methods

**Hann Window (Recommended)**
```python
tile_blending="hann"
```
- Smooth cosine-based blending
- Eliminates visible seams
- Best visual quality

**Linear Blending**
```python
tile_blending="linear"
```
- Simple linear interpolation in overlap regions
- Faster than Hann
- May show subtle seams

**No Blending**
```python
tile_blending="none"
```
- Hard boundaries between tiles
- Fastest processing
- Visible seams (debugging only)

### Advanced Tiling Setup

```python
from atlas.easy_api import create_sampler

# For 2048×2048 generation on 16GB GPU
sampler = create_sampler(
    checkpoint="model.pt",
    gpu_memory="16GB",
    # Tiling parameters
    tile_size=512,
    tile_stride=448,
    tile_blending="hann",
)

# Generate ultra-high-res image
images = sampler.generate(
    prompts=["a detailed landscape"],
    num_samples=1,              # Batch=1 required
    timesteps=60,
    resolution=2048,            # 2K generation
)
```

### Overlap Recommendations

| Resolution | Tile Size | Overlap | Stride | Tiles | Speed Impact |
|------------|-----------|---------|--------|-------|--------------|
| 1536×1536  | 512       | 10%     | 461    | 3×3   | 1.5x slower  |
| 2048×2048  | 512       | 12.5%   | 448    | 4×4   | 2.5x slower  |
| 2560×2560  | 640       | 15%     | 544    | 4×4   | 2.8x slower  |
| 3072×3072  | 512       | 20%     | 410    | 6×6   | 4x slower    |

**Overlap Guidelines:**
- **5-10%**: Minimal, may show seams
- **12.5%**: Good balance (default)
- **20%**: Smooth blending, slower
- **>25%**: Diminishing returns

### Memory vs Speed Tradeoff

```python
# Memory-constrained (8GB GPU, 2K image)
SamplerConfig(
    tile_size=384,              # Smaller tiles
    tile_overlap=0.20,          # More overlap for quality
    tile_blending="hann",
)
# Result: 6×6 tiles = ~5x slower

# Balanced (16GB GPU, 2K image)
SamplerConfig(
    tile_size=512,
    tile_overlap=0.125,
    tile_blending="hann",
)
# Result: 4×4 tiles = ~2.5x slower

# Speed-optimized (24GB GPU, 2K image)
SamplerConfig(
    tile_size=1024,             # Larger tiles
    tile_overlap=0.10,          # Less overlap
    tile_blending="linear",     # Faster blending
)
# Result: 2×2 tiles = ~1.5x slower
```

### Sequential vs Parallel Tiling

```python
# Parallel (default): Process multiple tiles simultaneously
SamplerConfig(
    ultrahighres_sequential_tiles=False,  # Default
)
# Pro: Faster (if memory allows)
# Con: Higher memory usage

# Sequential: One tile at a time
SamplerConfig(
    ultrahighres_sequential_tiles=True,
)
# Pro: Lower memory usage
# Con: Slower (but necessary on <12GB GPUs)
```

### Complete Example: 4K Generation

```python
from atlas.easy_api import create_sampler
from atlas.config import SamplerConfig

# Configure for 4K (4096×4096) on RTX 4090 (24GB)
sampler = create_sampler(
    checkpoint="atlas_large.pt",
    gpu_memory="24GB",
    tile_size=1024,              # Large tiles for speed
    tile_stride=922,             # ~10% overlap
    tile_blending="hann",
    enable_cuda_graphs=False,    # Disable for variable tile shapes
)

# Generate 4K image
result = sampler.generate(
    prompts=["an epic fantasy landscape with mountains and castles"],
    num_samples=1,
    timesteps=80,                # More steps for quality
    guidance_scale=7.5,
)

# Save result
import torch
from torchvision.utils import save_image

save_image(result, "4k_output.png")
```

### Troubleshooting

**Visible Seams**
- Increase overlap: `tile_overlap=0.20`
- Use Hann blending: `tile_blending="hann"`
- Increase tile size if memory allows

**Out of Memory**
- Reduce tile size: `tile_size=384`
- Enable sequential: `ultrahighres_sequential_tiles=True`
- Set batch size to 1

**Too Slow**
- Increase tile size
- Reduce overlap
- Use linear blending instead of Hann
- Consider lower resolution

**Tiles Don't Align**
- Ensure image size divisible by stride
- Use auto-padding: enabled by default
- Check logs for shape warnings

---

## Combining CUDA Graphs + Tiling

**Generally NOT recommended** because:
- Tiling creates variable shapes per tile → defeats graph caching
- Each tile would need separate graph → high memory overhead

**Exception**: Fixed tile configuration
```python
SamplerConfig(
    tile_size=512,
    tile_overlap=0.125,
    enable_cuda_graphs=True,     # Graph each 512×512 tile
    cuda_graph_warmup_iters=1,   # Reduce warmup overhead
)
```

This works if:
- All tiles are same size (512×512)
- Processing many 2K+ images with same tiling
- Graph cache size = 1 (only cache tile shape)

**Performance**: ~15% speedup over tiling alone, but still slower than non-tiled.

---

## Best Practices Summary

### For Maximum Speed
```python
SamplerConfig(
    enable_cuda_graphs=True,
    tile_size=None,              # Disable tiling
    use_mixed_precision=True,
)
```

### For Maximum Resolution
```python
SamplerConfig(
    enable_cuda_graphs=False,
    tile_size=512,
    tile_overlap=0.15,
    tile_blending="hann",
    ultrahighres_sequential_tiles=True,
)
```

### Balanced (Recommended)
```python
SamplerConfig(
    enable_cuda_graphs=True,     # For ≤1536px
    tile_size=512,               # Activates for >1536px
    tile_overlap=0.125,
    tile_blending="hann",
    ultrahighres_sequential_tiles=False,
)
```

### Debugging
```python
SamplerConfig(
    enable_cuda_graphs=False,    # Easier to trace errors
    tile_size=None,              # Simpler execution path
    verbose_logging=True,        # Enable detailed logs
)
```

---

## Performance Monitoring

```python
import time
import torch

def benchmark_generation(sampler, config_name, **kwargs):
    """Benchmark generation with detailed timing."""
    torch.cuda.synchronize()
    start = time.time()

    images = sampler.generate(**kwargs)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Get memory stats
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    print(f"\n=== {config_name} ===")
    print(f"Time: {elapsed:.2f}s")
    print(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    print(f"Images: {images.shape}")

    return images, elapsed

# Compare configurations
baseline, t1 = benchmark_generation(
    sampler, "Baseline",
    prompts=["test"], num_samples=4, timesteps=50
)

# Enable CUDA graphs
sampler.sampler.enable_cuda_graphs = True
with_graphs, t2 = benchmark_generation(
    sampler, "With CUDA Graphs",
    prompts=["test"], num_samples=4, timesteps=50
)

print(f"\nSpeedup: {t1/t2:.2f}x")
```
