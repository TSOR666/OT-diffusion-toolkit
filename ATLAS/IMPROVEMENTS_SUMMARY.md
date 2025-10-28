# ATLAS Usability Improvements Summary

## Overview

ATLAS has been significantly hardened and simplified to make it easy to use for non-experts and compatible with consumer GPUs. These improvements focus on:

1. **Simplified API** - Easy-to-use interface requiring minimal configuration
2. **Consumer GPU Support** - Optimized presets for 6GB, 8GB, 12GB, 16GB, and 24GB GPUs
3. **Better Error Handling** - Clear, actionable error messages with suggestions
4. **Memory Management** - Automatic memory estimation and budget control
5. **Configuration Validation** - Fail-fast validation with helpful warnings
6. **Comprehensive Documentation** - Quick start guide and examples

---

## What's New

### 1. Easy API (`atlas/easy_api.py`)

**Problem:** Previous API required extensive knowledge of diffusion models, kernel operators, and configuration management.

**Solution:** New simplified API that handles everything automatically.

**Before (Complex):**
```python
# Required 20+ lines of configuration
model_cfg = HighResModelConfig(in_channels=4, out_channels=4, ...)
kernel_cfg = KernelConfig(solver_type="auto", epsilon=0.01, ...)
sampler_cfg = SamplerConfig(sb_iterations=3, ...)
model = HighResLatentScoreModel(config=model_cfg).to(device)
sampler = AdvancedHierarchicalDiffusionSampler(
    score_model=model,
    noise_schedule=karras_noise_schedule,
    device=device,
    kernel_config=kernel_cfg,
    sampler_config=sampler_cfg,
)
# ... more setup for CLIP, etc.
```

**After (Simple):**
```python
import atlas

# Automatically detects GPU and configures everything
sampler = atlas.create_sampler(checkpoint="model.pt")
images = sampler.generate(num_samples=4)
```

**Features:**
- ✅ Automatic GPU detection and memory profiling
- ✅ Optimal configuration selection
- ✅ Built-in error handling with helpful messages
- ✅ Memory estimation and warnings
- ✅ One-line text-to-image generation
- ✅ Configuration validation

---

### 2. Consumer GPU Presets (`atlas/config/presets.py`)

**Problem:** Default configurations assumed 16GB+ professional GPUs, causing OOM errors on consumer hardware.

**Solution:** Five carefully tuned presets for different GPU memory tiers.

#### Available Presets

| Preset | GPU Examples | Resolution | Batch | CLIP | Memory |
|--------|--------------|------------|-------|------|--------|
| `gpu:6gb` | GTX 1660, RTX 3050 | 512×512 | 1 | ❌ | ~4-5 GB |
| `gpu:8gb` | RTX 3060, RTX 4060 | 512×512 | 2 | ✅ | ~6-7 GB |
| `gpu:12gb` | RTX 3080, RTX 4070 Ti | 1024×1024 | 4 | ✅ | ~10-11 GB |
| `gpu:16gb` | RTX 4080, RTX 4090 | 1024×1024 | 8 | ✅ | ~14-15 GB |
| `gpu:24gb` | RTX 4090, A5000 | 1024×1024 | 16 | ✅ | ~20-22 GB |

**Usage:**
```python
from atlas.config.presets import load_preset

# Load preset optimized for your GPU
config = load_preset("gpu:8gb")

# Customize if needed
config["training"].epochs = 200
config["dataset"].root = "./my_data"
```

**Each preset configures:**
- Model architecture (channels, layers, attention)
- Kernel solver (FFT for small GPUs, auto for large)
- Memory thresholds and cache sizes
- Training batch sizes and micro-batching
- Mixed precision settings
- CLIP conditioning (enabled only when memory allows)

---

### 3. Enhanced Error Handling (`atlas/solvers/hierarchical_sampler.py`)

**Problem:** Cryptic error messages like "RuntimeError: CUDA out of memory" with no guidance.

**Solution:** Detailed, actionable error messages with specific suggestions.

**Before:**
```
RuntimeError: CUDA out of memory
```

**After:**
```
RuntimeError: Out of GPU memory during sampling.
Memory usage: 8234.5 MB allocated, 9012.3 MB peak
Current settings:
  - batch_size: 8
  - shape: (8, 4, 128, 128)
  - mixed_precision: True

Suggestions:
  1. Reduce batch_size (currently 8)
  2. Enable mixed_precision if not already enabled
  3. Use smaller resolution
  4. Clear GPU cache: torch.cuda.empty_cache()
  5. Reduce kernel cache size in KernelConfig
```

**Enhanced validation:**
- ✅ Batch size validation (must be > 0)
- ✅ Shape validation (must include batch and channels)
- ✅ Timestep validation with clear error messages
- ✅ Pre-generation memory checks with warnings
- ✅ Conditioning setup error handling
- ✅ Context tracking (which step failed)

---

### 4. Configuration Validation (`atlas/easy_api.py`)

**Problem:** Configuration errors discovered late during training/sampling, wasting time and GPU hours.

**Solution:** Fail-fast validation with cross-configuration checks.

**Validation checks:**
```python
issues = validate_configs(
    model_config=model_cfg,
    kernel_config=kernel_cfg,
    sampler_config=sampler_cfg,
    conditioning_config=cond_cfg,
    gpu_profile=profile,
)

if issues:
    print("Configuration warnings:")
    for issue in issues:
        print(f"  - {issue}")
```

**Detects:**
- ❌ Model `context_dim` mismatch with CLIP output
- ❌ Kernel epsilon out of reasonable range (too large/small)
- ❌ RFF features too low for good approximation
- ❌ Estimated memory exceeds GPU capacity
- ❌ Redundant configuration combinations
- ❌ Invalid parameter types or ranges

---

### 5. Memory Management Tools

#### Automatic Memory Estimation
```python
mem = sampler.estimate_memory_usage(batch_size=4, resolution=1024)
print(f"Model params: {mem['model_params_mb']} MB")
print(f"Activations: {mem['activations_mb']} MB")
print(f"Total: {mem['total_mb']} MB")
```

#### Memory Budget Control
- Pre-generation memory checks
- Configurable memory thresholds per GPU
- Automatic warning system
- Cache clearing utilities

#### Cache Management
```python
# Clear kernel cache to free memory
sampler.clear_cache()

# Or from SB solver
sampler.clear_kernel_cache()
```

---

### 6. Quick Start Guide (`QUICK_START.md`)

**New comprehensive documentation including:**

1. **Installation instructions**
2. **3-line quick start example**
3. **GPU memory requirements table**
4. **Basic usage examples**
5. **Text-to-image generation guide**
6. **Training tutorials**
7. **Troubleshooting section**
8. **Performance tips**
9. **Common workflows**

**Key sections:**
- GPU requirements for different use cases
- Error message interpretation
- Memory optimization strategies
- Quality vs. speed trade-offs
- Step-by-step training guide

---

### 7. Easy Start Example (`atlas/examples/easy_start.py`)

**Command-line tool for quick experimentation:**

```bash
# Auto-detect GPU and generate samples
python easy_start.py --checkpoint model.pt --num_samples 4

# Specify GPU profile manually
python easy_start.py --checkpoint model.pt --gpu_memory 8GB

# Text-to-image generation
python easy_start.py --checkpoint model.pt \
    --prompts "a red car" "a blue house" \
    --guidance_scale 7.5

# List available GPU profiles
python easy_start.py --list_profiles
```

**Features:**
- Automatic GPU detection
- Memory estimation before generation
- Clear progress indicators
- Helpful error messages
- Image saving utilities
- Seed control for reproducibility

---

## Usage Examples

### Example 1: Absolute Beginner (3 lines)

```python
import atlas

sampler = atlas.create_sampler(checkpoint="model.pt")
images = sampler.generate(num_samples=4)
```

### Example 2: Text-to-Image

```python
import atlas

sampler = atlas.create_sampler(checkpoint="model.pt", gpu_memory="8GB")
images = sampler.generate(
    prompts=["a red sports car", "a mountain landscape"],
    num_samples=4,
    guidance_scale=7.5,
)
```

### Example 3: Memory-Constrained GPU

```python
import atlas

# Explicitly use 6GB profile
sampler = atlas.create_sampler(
    checkpoint="model.pt",
    gpu_memory="6GB",
    batch_size=1,
    resolution=512,
)

# Check memory before generating
mem = sampler.sampler.estimate_memory_usage(batch_size=1, resolution=512)
print(f"Estimated: {mem['total_mb']} MB")

images = sampler.generate(num_samples=1, timesteps=25)
```

### Example 4: Training with Preset

```python
from atlas.config.presets import load_preset
from atlas.examples.training_pipeline import run_training

# Load 8GB GPU preset
config = load_preset("gpu:8gb")

# Point to your data
config["dataset"].root = "./my_images"
config["training"].epochs = 200

# Train!
run_training(config)
```

### Example 5: List GPU Profiles

```python
import atlas

# See all available profiles
atlas.list_profiles()

# Output:
# Available GPU Memory Profiles:
# ===============================================================================
#
# 6GB:
#   Description: Consumer GPUs: GTX 1660, RTX 3050, RTX 4050
#   Resolution: 512x512
#   Batch size: 1
#   Mixed precision: True
#   CLIP enabled: False
#   Kernel solver: fft
# ...
```

---

## Technical Improvements

### Code Quality

1. **Type Safety**
   - Added comprehensive type hints
   - Better error messages for type mismatches

2. **Documentation**
   - Extensive docstrings for all new functions
   - Clear parameter descriptions
   - Usage examples in docstrings

3. **Error Handling**
   - Try-except blocks with context
   - Specific error types (ValueError, RuntimeError)
   - Helpful suggestions in error messages

4. **Memory Safety**
   - Pre-allocation checks
   - Memory estimation tools
   - Automatic cache clearing

### Performance

1. **Memory Optimization**
   - Reduced cache sizes for small GPUs
   - Mixed precision by default
   - Gradient checkpointing options

2. **Configuration Tuning**
   - Optimized kernel selection (FFT for small GPUs)
   - Reduced model complexity for 6GB tier
   - Balanced quality vs. memory trade-offs

3. **Monitoring**
   - Peak memory tracking
   - Performance statistics
   - Automatic warnings

---

## Migration Guide

### For Existing Users

**Old code still works!** The improvements are additive.

```python
# Old code (still supported)
from atlas import HighResLatentScoreModel, AdvancedHierarchicalDiffusionSampler
model = HighResLatentScoreModel(config=cfg)
sampler = AdvancedHierarchicalDiffusionSampler(...)

# New code (easier)
import atlas
sampler = atlas.create_sampler(checkpoint="model.pt")
```

### For New Users

**Start with the easy API:**
1. Read `QUICK_START.md`
2. Try `atlas/examples/easy_start.py`
3. Use `atlas.create_sampler()` for projects
4. Graduate to advanced API when needed

---

## File Changes Summary

### New Files
- ✅ `atlas/easy_api.py` - Simplified API (600+ lines)
- ✅ `QUICK_START.md` - Comprehensive guide (400+ lines)
- ✅ `atlas/examples/easy_start.py` - CLI example (200+ lines)
- ✅ `IMPROVEMENTS_SUMMARY.md` - This document

### Modified Files
- ✅ `atlas/__init__.py` - Added easy API exports
- ✅ `atlas/config/presets.py` - Added 5 consumer GPU presets (400+ lines)
- ✅ `atlas/solvers/hierarchical_sampler.py` - Enhanced error handling, memory tools

### Lines of Code Added
- ~1,800 lines of new functionality
- ~600 lines of documentation
- ~200 lines of examples

---

## Testing Recommendations

### Manual Testing

1. **Test GPU detection:**
   ```python
   import atlas
   profile = atlas.detect_gpu_profile()
   print(profile.name)
   ```

2. **Test memory estimation:**
   ```python
   sampler = atlas.create_sampler()
   mem = sampler.sampler.estimate_memory_usage(batch_size=4, resolution=1024)
   assert mem['total_mb'] > 0
   ```

3. **Test error messages:**
   ```python
   # Try invalid batch size
   sampler = atlas.create_sampler()
   try:
       sampler.generate(num_samples=0)  # Should fail with helpful error
   except ValueError as e:
       print(e)  # Should see actionable suggestion
   ```

4. **Test each GPU preset:**
   ```python
   for preset_name in ["gpu:6gb", "gpu:8gb", "gpu:12gb", "gpu:16gb", "gpu:24gb"]:
       config = atlas.presets.load_preset(preset_name)
       assert config["resolution"] > 0
       assert config["max_batch_size"] > 0
   ```

### Integration Testing

1. Run `easy_start.py` with different profiles
2. Test training with each preset
3. Verify memory stays within limits
4. Check error messages are helpful

---

## Future Enhancements

### Potential Additions

1. **Automatic batch size tuning**
   - Measure actual memory usage
   - Binary search for optimal batch size
   - Save results for future runs

2. **Model zoo**
   - Pre-trained checkpoints for each GPU tier
   - Download and use immediately
   - Fine-tuning examples

3. **Web UI**
   - Gradio interface for non-programmers
   - Visual configuration builder
   - Real-time memory monitoring

4. **Multi-GPU support**
   - Automatic sharding across GPUs
   - Larger batch sizes
   - Faster training

5. **CPU fallback**
   - Slow but functional CPU mode
   - For users without GPUs
   - Testing and development

---

## Conclusion

ATLAS is now **significantly more accessible** for:
- ✅ Non-experts learning diffusion models
- ✅ Researchers with consumer GPUs
- ✅ Developers building applications
- ✅ Students on limited hardware

**Key achievements:**
- 🎯 3-line quick start (vs. 20+ before)
- 🎯 5 GPU tiers supported (vs. 1 before)
- 🎯 Automatic configuration (vs. manual before)
- 🎯 Helpful errors (vs. cryptic before)
- 🎯 Comprehensive docs (vs. sparse before)

**The improvements maintain:**
- ✅ Full backward compatibility
- ✅ Advanced features for experts
- ✅ Code quality and performance
- ✅ Flexibility and extensibility

---

## Credits

These improvements were designed to make ATLAS accessible to everyone, regardless of their machine learning expertise or hardware budget.

For questions or issues, see `QUICK_START.md` or the documentation in `reports/atlas/`.
