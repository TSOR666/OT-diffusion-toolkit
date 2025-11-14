# ATLAS Dependencies Matrix

## Core Requirements

| Package | Minimum Version | Recommended | Purpose |
|---------|----------------|-------------|---------|
| Python | 3.10 | 3.11+ | Core language |
| PyTorch | 2.0.0 | 2.2.0+ | Deep learning framework |
| NumPy | 1.24.0 | 1.26.0+ | Numerical operations |
| tqdm | 4.65.0 | latest | Progress bars |

## Optional Dependencies

### Vision Extras (`pip install .[vision]`)
| Package | Version | Purpose |
|---------|---------|---------|
| torchvision | 0.15.0+ | Dataset utilities |
| Pillow | 9.0.0+ | Image I/O |

### CLIP Conditioning (`pip install .[clip]`)
| Package | Version | Purpose |
|---------|---------|---------|
| open-clip-torch | 2.20.0+ | Text-to-image conditioning |
| transformers | 4.30.0+ | Tokenization (optional) |

### Development Tools (`pip install .[dev]`)
| Package | Version | Purpose |
|---------|---------|---------|
| pytest | 7.0.0+ | Testing framework |
| pytest-cov | 4.0.0+ | Coverage reporting |
| ruff | 0.1.0+ | Fast linter |
| mypy | 1.5.0+ | Type checking |

## CUDA/GPU Requirements

### For GPU Acceleration
- **NVIDIA GPU**: Compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- **CUDA Toolkit**: 11.8+ (12.1+ recommended)
- **CUDA Driver**: 520+ (535+ for CUDA 12.x)

### For CUDA Graphs
- **CUDA Version**: 11.0+ required
- **PyTorch**: 2.0+ with CUDA compilation
- **GPU Memory**: 16GB+ recommended for high-res generation

### For BF16 Training
- **GPU**: Ampere (RTX 30xx/A100) or newer
- **PyTorch**: 2.0+ with CUDA 11.8+
- **Check**: `torch.cuda.is_bf16_supported()`

### For TF32 Acceleration
- **GPU**: Ampere (RTX 30xx/A100) or newer
- **CUDA**: 11.0+
- **Automatic**: Enabled by default on compatible hardware

## CPU-Only Mode

ATLAS runs on CPU with reduced performance:
- All kernels work (RFF recommended over Direct)
- CUDA graphs disabled automatically
- Mixed precision falls back to FP32
- 2-10x slower than GPU depending on operation

**Recommended**: 16+ CPU cores, 32GB+ RAM for 512px generation

## Dependency Installation Examples

### Minimal (CPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .
```

### Full GPU (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install .[vision,clip]
```

### Development Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .[vision,clip,dev]
```

## Compatibility Matrix

| PyTorch | Python | CUDA | BF16 | CUDA Graphs |
|---------|--------|------|------|-------------|
| 2.0.x   | 3.10-3.11 | 11.8 | ✅ | ✅ |
| 2.1.x   | 3.10-3.11 | 11.8, 12.1 | ✅ | ✅ |
| 2.2.x   | 3.10-3.12 | 11.8, 12.1 | ✅ | ✅ |
| 2.3.x   | 3.10-3.12 | 11.8, 12.1, 12.4 | ✅ | ✅ |

## Verification Commands

```python
import torch

# Check PyTorch version and CUDA availability
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")

    # Check memory
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total GPU Memory: {total_mem:.1f} GB")
```

## Troubleshooting

### ImportError: libcuda.so.1
- **Cause**: NVIDIA drivers not installed
- **Fix**: Install drivers from https://www.nvidia.com/Download/index.aspx

### RuntimeError: CUDA out of memory
- **Cause**: Insufficient GPU memory
- **Fix**: Reduce batch size, resolution, or use CPU offloading

### UserWarning: bf16 not supported
- **Cause**: Pre-Ampere GPU (e.g., GTX 1080, RTX 20xx)
- **Fix**: Disable BF16 in config, use FP16 instead

### ImportError: open_clip
- **Cause**: CLIP extras not installed
- **Fix**: `pip install .[clip]` or disable CLIP conditioning
