# ATLAS Documentation

Comprehensive guides for using and extending ATLAS.

## Getting Started

- [Main README](../README.md) - Overview, installation, quick start
- [How to Train & Run Anywhere](HOW_TO_TRAIN_AND_INFER.md) - OS-specific setup, CPU/GPU guidance, torch.compile tips
- [Dependencies Matrix](DEPENDENCIES.md) - Requirements, compatibility, installation

## Core Guides

### Hardware & Performance
- **[GPU/CPU Behavior](GPU_CPU_BEHAVIOR.md)** - Hardware detection, precision modes, performance expectations
  - Automatic capability detection
  - FP16/BF16/TF32 support
  - CPU vs GPU behavior
  - Performance benchmarks
  - Troubleshooting

- **[CUDA Graphs & Tiling](CUDA_GRAPHS_TILING.md)** - Advanced optimization techniques
  - CUDA graph acceleration (10-30% speedup)
  - Ultra-high-resolution tiling
  - Memory vs speed tradeoffs
  - Complete configuration examples

### Customization
- **[Extending ATLAS](EXTENDING.md)** - Implement custom components
  - Custom score networks
  - Custom kernel operators
  - Custom noise schedules
  - Custom conditioning
  - Custom samplers

## Quick Links

### For Users
- **New to ATLAS?** Start with the [Main README](../README.md)
- **Installation issues?** See [Dependencies](DEPENDENCIES.md)
- **Performance problems?** Check [GPU/CPU Behavior](GPU_CPU_BEHAVIOR.md)
- **Out of memory?** Read [CUDA Graphs & Tiling](CUDA_GRAPHS_TILING.md)

### For Developers
- **Want to extend ATLAS?** Follow [Extending ATLAS](EXTENDING.md)
- **Contributing?** See [CONTRIBUTING.md](../CONTRIBUTING.md) (if available)

## Common Tasks

### Check Hardware Capabilities
```python
from atlas.utils.hardware import print_hardware_info
print_hardware_info()
```

### Enable CUDA Graphs
```python
from atlas.easy_api import create_sampler

sampler = create_sampler(
    checkpoint="model.pt",
    gpu_memory="16GB",
    enable_cuda_graphs=True,
)
```

### Generate Ultra-High-Res (2K+)
```python
sampler = create_sampler(
    checkpoint="model.pt",
    gpu_memory="24GB",
    tile_size=512,
    tile_overlap=0.125,
)

images = sampler.generate(
    prompts=["a landscape"],
    num_samples=1,
    resolution=2048,
)
```

### Implement Custom Kernel
```python
from atlas.kernels.base import KernelOperator

class MyKernel(KernelOperator):
    def apply(self, x, v):
        # Your implementation
        ...
```
See [Extending ATLAS](EXTENDING.md) for full examples.

### Validate Noise Prediction Models

Use `atlas.utils.NoisePredictionAdapter` to sanity-check custom score models before they are
passed to the Schr\"odinger bridge solver. The adapter enforces the expected shape/dtype, ensures
finite outputs, and applies classifier-free guidance hooks identically to the solver's runtime.

```python
from atlas.utils import NoisePredictionAdapter

adapter = NoisePredictionAdapter(model)
noise = adapter.predict_noise(x, t=0.5, conditioning=conditioning_payload)
```

## FAQ

**Q: What GPU do I need?**
A: ATLAS runs on any CUDA GPU (compute 7.0+). Recommended: RTX 3060+ for 512px, RTX 4080+ for 1024px, RTX 4090 for 2K.

**Q: Can I run on CPU?**
A: Yes, but 5-10x slower. Use reduced RFF features (512) and lower resolution (256-512px).

**Q: How do I enable BF16?**
A: Automatic on Ampere+ GPUs (RTX 30xx/40xx/A100). Check with `torch.cuda.is_bf16_supported()`.

**Q: What's the difference between CUDA graphs and tiling?**
A: CUDA graphs speed up execution (10-30%), tiling enables higher resolutions by processing in chunks (slower but less memory).

**Q: My generations are slow, how can I speed them up?**
A:
1. Enable CUDA graphs for fixed shapes
2. Use mixed precision (BF16/FP16)
3. Reduce number of sampling steps
4. Use RFF kernels instead of Direct
5. Ensure TF32 is enabled (Ampere+)

**Q: I get OOM errors, what should I do?**
A:
1. Reduce batch size
2. Lower resolution
3. Enable tiling for ultra-high-res
4. Use Nyström kernels for tight memory
5. Clear cache: `sampler.clear_cache()`

## Support

- **Issues**: Report bugs at [GitHub Issues](https://github.com/tsoreze/OT-diffusion-toolkit/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/tsoreze/OT-diffusion-toolkit/discussions)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)

## Document Index

| Document | Topics | Audience |
|----------|--------|----------|
| [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md) | Cross-platform install, training & inference walkthrough | All users |
| [DEPENDENCIES.md](DEPENDENCIES.md) | Requirements, compatibility, installation | All users |
| [GPU_CPU_BEHAVIOR.md](GPU_CPU_BEHAVIOR.md) | Hardware, precision, performance | Users, advanced |
| [CUDA_GRAPHS_TILING.md](CUDA_GRAPHS_TILING.md) | Optimization, ultra-high-res | Advanced users |
| [EXTENDING.md](EXTENDING.md) | Custom components, API | Developers |

---

*Last updated: 2025-11-06*
