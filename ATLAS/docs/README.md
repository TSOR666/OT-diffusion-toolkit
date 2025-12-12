# ATLAS Documentation

Comprehensive guides for using and extending ATLAS.

## Choose Your Path

### 🚀 I'm New to ATLAS or Diffusion Models
**Start here:** [Complete Beginner's Guide](GETTING_STARTED.md)

This guide covers:
- What ATLAS is and what you can do with it
- Step-by-step installation for any platform
- Understanding checkpoints and the training/inference workflow
- Your first image generation
- Common beginner questions and troubleshooting

**Time:** 15-30 minutes to get set up and generating images

### ⚡ I Want to Get Started Quickly
**Start here:** [Quick Start Guide](QUICKSTART.md)

For users who:
- Have basic Python/ML familiarity
- Want a quick reference guide
- Need working code examples fast

**Time:** 5-10 minutes

### 🎓 I Want a Complete Training & Inference Tutorial
**Start here:** [How to Train & Run Anywhere](HOW_TO_TRAIN_AND_INFER.md)

For users who want:
- Cross-platform installation details (Windows/Mac/Linux)
- Complete training walkthrough with presets
- Production deployment guidance
- LoRA fine-tuning instructions

**Time:** 30-60 minutes to understand, hours/days for actual training

---

## Core Documentation

### Getting Started
- **[Complete Beginner's Guide](GETTING_STARTED.md)** ✨ - Start here if you're new
- **[Quick Start Guide](QUICKSTART.md)** ⚡ - Quick reference with examples
- **[How to Train & Run Anywhere](HOW_TO_TRAIN_AND_INFER.md)** 🎓 - Complete training/inference guide
- **[Dependencies Matrix](DEPENDENCIES.md)** 📦 - Requirements, compatibility, installation

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

## By Use Case

### I want to...
| Task | Guide |
|------|-------|
| **Understand what ATLAS does** | [Beginner's Guide § What is ATLAS](GETTING_STARTED.md#what-is-atlas) |
| **Install ATLAS on Windows/Mac/Linux** | [Beginner's Guide § Installation](GETTING_STARTED.md#step-1-installation) |
| **Get my first images quickly** | [Quick Start Guide](QUICKSTART.md) |
| **Train a custom model** | [How to Train & Run Anywhere § Training](HOW_TO_TRAIN_AND_INFER.md#5-training-atlas-presets) |
| **Fix "out of memory" errors** | [Quick Start § Troubleshooting](QUICKSTART.md#8-troubleshooting) |
| **Generate 2K+ resolution images** | [CUDA Graphs & Tiling](CUDA_GRAPHS_TILING.md) |
| **Speed up generation** | [GPU/CPU Behavior § Performance](GPU_CPU_BEHAVIOR.md#performance-expectations) |
| **Fine-tune an existing model** | [How to Train & Run Anywhere § LoRA](HOW_TO_TRAIN_AND_INFER.md#7-fine-tuning-with-lora-adapters) |
| **Create custom kernels/samplers** | [Extending ATLAS](EXTENDING.md) |
| **Debug hardware issues** | [GPU/CPU Behavior Guide](GPU_CPU_BEHAVIOR.md) |

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

### Hardware & Requirements

**Q: What GPU do I need?**
A: ATLAS runs on any CUDA GPU (compute 7.0+) or CPU. Recommended:
- 512px generation: RTX 3060+ (8GB VRAM)
- 1024px generation: RTX 4080+ (16GB VRAM)
- 2K generation: RTX 4090 (24GB VRAM)
- CPU-only works but is 5-10x slower

**Q: Can I run ATLAS without a GPU?**
A: Yes! ATLAS works on CPU but is significantly slower (5-10x). Use:
- Lower resolution (256-512px)
- Reduced RFF features (512 instead of 2048)
- Smaller batch sizes (1-2)
- See [Beginner's Guide § Prerequisites](GETTING_STARTED.md#prerequisites)

**Q: Does ATLAS support AMD GPUs or Apple Silicon?**
A: Yes, but with caveats:
- **AMD (ROCm)**: Supported on Linux. See [HOW_TO_TRAIN_AND_INFER.md § Install PyTorch](HOW_TO_TRAIN_AND_INFER.md#3-install-pytorch-for-your-platform)
- **Apple Silicon (M1/M2/M3)**: Supported via MPS backend, 2-3x slower than NVIDIA
- **Intel Arc**: Not officially tested

### Getting Started

**Q: Where do I get model checkpoints?**
A: You need to either:
1. **Train your own** (see [HOW_TO_TRAIN_AND_INFER.md](HOW_TO_TRAIN_AND_INFER.md))
2. **Download community checkpoints** (check GitHub Discussions/Issues)
3. **Train a quick test model** (1-2 hours for 64x64 ImageNet)

ATLAS does not ship with pre-trained models due to size and licensing.

**Q: How long does training take?**
A: Depends on your hardware and target resolution:
- 64x64 (test): 2-4 hours on RTX 4090
- 256x256: 2-4 days on RTX 4090
- 1024x1024: 1-2 weeks on RTX 4090
- See [HOW_TO_TRAIN_AND_INFER.md § Training Times](HOW_TO_TRAIN_AND_INFER.md#55-estimated-training-times-default-presets)

**Q: What datasets can I use?**
A: ATLAS supports:
- **Auto-download**: CIFAR-10, MNIST (via torchvision)
- **Manual download**: CelebA, CelebA-HQ, FFHQ, ImageNet, LSUN
- **Custom**: Any folder of images (ImageFolder format)
- Minimum 1000+ diverse images recommended

For download links, see [Getting Started § Dataset Downloads](GETTING_STARTED.md#dataset-downloads)

### Performance & Optimization

**Q: My generations are slow, how can I speed them up?**
A: Try these in order:
1. Enable CUDA graphs: `enable_cuda_graphs=True` (10-30% speedup)
2. Use mixed precision (automatic on compatible GPUs)
3. Reduce timesteps: `timesteps=30` instead of 50
4. Use RFF kernels instead of Direct
5. Check GPU usage: `python -m atlas.check_hardware`

See [GPU/CPU Behavior § Performance](GPU_CPU_BEHAVIOR.md#performance-expectations)

**Q: I get "CUDA out of memory" errors, what should I do?**
A: Solutions in order:
1. Reduce batch size: `batch_size=1`
2. Lower resolution: `resolution=512`
3. Enable tiling for ultra-high-res: `tile_size=512`
4. Use Nyström kernels: `solver_type="nystrom"`
5. Clear cache: `sampler.clear_cache()`

See [Quick Start § Troubleshooting OOM](QUICKSTART.md#cuda-out-of-memory)

**Q: How do I enable BF16/TF32?**
A: Both are automatic on compatible hardware:
- **BF16**: Auto-enabled on Ampere+ GPUs (RTX 30xx/40xx/50xx, A100+)
- **TF32**: Auto-enabled on Ampere+ for matmul operations
- Check: `python -m atlas.check_hardware`

**Q: What's the difference between CUDA graphs and tiling?**
A: Different purposes:
- **CUDA graphs**: Speed up execution (10-30% faster) by caching computation graph
- **Tiling**: Enable higher resolutions by processing in chunks (slower but less memory)
- You can use both together!

See [CUDA Graphs & Tiling Guide](CUDA_GRAPHS_TILING.md)

### Troubleshooting

**Q: Images are all noise/garbage**
A: Common causes:
1. Model not trained enough (need more epochs)
2. Learning rate too high/low
3. Dataset too small (need 1000+ images)
4. Wrong checkpoint loaded

**Q: "No module named 'atlas'" error**
A: Installation issue:
```bash
cd ATLAS/
pip install -e .[vision,clip]
```

**Q: Can I use pre-trained Stable Diffusion checkpoints?**
A: Not directly. ATLAS uses a different architecture. You would need to:
1. Convert the checkpoint (advanced, not officially supported)
2. Train from scratch (recommended)

### Advanced Topics

**Q: Can I fine-tune with LoRA?**
A: Yes! See [HOW_TO_TRAIN_AND_INFER.md § LoRA](HOW_TO_TRAIN_AND_INFER.md#7-fine-tuning-with-lora-adapters)

**Q: How do I create custom kernels?**
A: See [Extending ATLAS](EXTENDING.md) for complete guide

**Q: Can I use ATLAS for other modalities (audio, video)?**
A: ATLAS is designed for images. For other modalities, you would need to adapt the architecture (advanced).

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
