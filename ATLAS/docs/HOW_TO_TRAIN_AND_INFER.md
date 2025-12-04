# How to Train and Run ATLAS Anywhere

> **Audience:** Practitioners who are new to diffusion tooling and need an end-to-end guide that works on Windows, Linux, or macOS, even without a discrete GPU.
>
> **Goal:** Install PyTorch and ATLAS, configure the environment, launch training, and run inference with clear fallbacks when optional accelerators such as `torch.compile` are unavailable.

---

## 1. Understand the Hardware Story

| Scenario | Works? | Practical tips |
| --- | --- | --- |
| Dedicated NVIDIA GPU (CUDA 11.8+/12.x) | ✅ | Fastest experience. Install CUDA-enabled PyTorch wheels and keep drivers ≥ 525.60.13. |
| AMD GPU on Linux (ROCm 5.6+) | ✅ | Use the ROCm PyTorch builds. Limit resolution to ≤ 768px for 16 GB cards. |
| Apple Silicon (M1/M2/M3) | ✅ | Uses Metal (`mps`). Expect 2–3× slower than NVIDIA RTX 4080. Reduce batch size to 1–2. |
| CPU only (any OS) | ✅ (slower) | Training/inference works but is 5–15× slower. Use 256 px presets, micro-batches of 1, and disable `torch.compile`. |
| Windows Subsystem for Linux 2 (WSL2) | ✅ | Treat as Linux. Install CUDA toolkit and drivers on Windows, then CUDA-enabled PyTorch inside WSL. |

**Key takeaway:** ATLAS automatically detects available accelerators. If no GPU is present it falls back to CPU without code changes, but you must dial down batch sizes/resolution for reasonable runtimes.

---

## 2. Prepare Your Python Environment

1. Install Python 3.10 or 3.11.
   - **Windows:** Use [Python.org installers](https://www.python.org/downloads/windows/) or the Microsoft Store. Tick “Add Python to PATH”.
   - **Linux:** Use your package manager (`sudo apt install python3.11 python3.11-venv` on Ubuntu 22.04+).
   - **macOS:** Install via [Homebrew](https://brew.sh/) (`brew install python@3.11`).
2. Create an isolated virtual environment (recommended for every OS):
   ```bash
   python -m venv .venv
   # On Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux
   source .venv/bin/activate
   ```
3. Upgrade pip, setuptools, and wheel in the environment:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```
4. Install build tools if you are on Windows and plan to compile optional dependencies:
   - Visual Studio Build Tools (Desktop Development with C++)
   - Enable “Long Paths” in Windows if cloning into deep folder structures.

> **Tip:** If you prefer Conda/Mamba, create an environment with `conda create -n atlas python=3.11` and activate it before continuing.

---

## 3. Install PyTorch for Your Platform

Choose exactly **one** of the commands below inside the activated environment. They install PyTorch, torchvision, and matching dependencies.

| Platform | Command |
| --- | --- |
| CUDA 12.1 (NVIDIA) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 (legacy NVIDIA) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| ROCm 5.6 (AMD on Linux) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6` |
| Apple Silicon (MPS) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` (Metal backend included) |
| CPU-only (any OS) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |

Verify the install:
```bash
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
PY
```

If `torch.cuda.is_available()` prints `False` on a GPU machine, check driver installation (Windows: GeForce Experience, Linux: `nvidia-smi`, macOS: use CPU/MPS).

---

## 4. Install ATLAS and Optional Extras

1. Clone the repository (or use an existing checkout):
   ```bash
   git clone https://github.com/tsoreze/OT-diffusion-toolkit.git
   cd OT-diffusion-toolkit/ATLAS
   ```
2. Install ATLAS in editable mode so scripts pick up local changes:
   ```bash
   pip install -e .[vision,clip]
   ```
   - The `vision` extra provides torchvision image utilities needed by the example inference pipeline.
   - The `clip` extra installs CLIP conditioning support. Omit extras on constrained systems if you do not need them.
3. (Optional) Install developer utilities for linting/tests:
   ```bash
   pip install -e .[dev]
   ```

Run the built-in hardware report to confirm detection:
```bash
python -m atlas.check_hardware
```
This prints detected GPUs/CPUs, available precision (FP16/BF16), and recommended batch sizes.

---

## 5. Training ATLAS Presets

ATLAS ships ready-to-train presets. The `atlas.examples.training_pipeline.run_training` helper orchestrates loading the preset, dataset, and training loop.

### 5.1 Minimal Training Command (All Platforms)

```bash
# CIFAR-10: Auto-downloads, fastest for testing
python -m atlas.examples.cifar10_training \
  --data-root ./data/cifar10 \
  --checkpoints ./checkpoints \
  --device cpu

# LSUN: Requires manual download
python -m atlas.examples.lsun256_training \
  --data-root /path/to/lsun/bedroom \
  --checkpoints ./checkpoints \
  --device cpu
```

Other datasets follow the exact same CLI surface—only the preset name changes:

```bash
# FFHQ 128×128
python -m atlas.examples.ffhq128_training \
  --data-root /datasets/ffhq \
  --checkpoints ./checkpoints/ffhq128

# ImageNet 64×64
python -m atlas.examples.imagenet64_training \
  --data-root /datasets/imagenet64 \
  --checkpoints ./checkpoints/imagenet64

# CelebA-HQ 1024×1024
python -m atlas.examples.celeba1024_training \
  --data-root /datasets/celeba_hq \
  --checkpoints ./checkpoints/celeba
```

**Notes:**
- Replace `--device cpu` with `--device cuda:0` (NVIDIA), `--device mps` (Apple Silicon), or omit it to let ATLAS auto-detect
- **CIFAR-10** downloads automatically via torchvision
- **Other datasets** require manual download - see [Dataset Downloads](GETTING_STARTED.md#dataset-downloads) for links
- Datasets should be in ImageFolder format (folder per class)

### 5.2 Adjusting for CPU-Only or Low-Memory Systems

1. **Reduce batch sizes in the preset:** Edit [atlas/config/presets.py](../atlas/config/presets.py) and adjust the `batch_size`, `micro_batch_size`, and `dataset.batch_size` fields for the preset you are using (for example, `lsun256_experiment`). Set them to values such as `batch_size=4`, `micro_batch_size=1`, and `dataset.batch_size=1` to keep memory demands low.
2. **Disable data loader workers:** In the same preset block, change `num_workers` to `0` to avoid multiprocessing overhead on CPU-only machines.
3. **Device Override:** Pass `--device cpu` when launching the training script to force CPU execution.
4. **Precision:** Mixed precision (`train_cfg.mixed_precision`) only helps on CUDA GPUs. Leave it `False` on CPU/MPS.
5. **Gradient Accumulation:** `train_cfg.batch_size` is implemented via accumulation inside the training loop, so even small micro-batches produce stable updates.

### 5.3 Disabling or Handling `torch.compile`

Some presets enable `train_cfg.compile` for extra speed on PyTorch 2.0+. During startup ATLAS attempts to wrap the model with `torch.compile` and logs failures before falling back to eager mode. Beginners often worry about warnings such as:

```
torch.compile failed (<error message>); continuing without compilation
```

This warning is benign—training continues normally. To silence it or skip compilation entirely, set `compile=False` in the preset you are using (for example, inside `lsun256_experiment` or `celeba1024_experiment` in [atlas/config/presets.py](../atlas/config/presets.py)). On Windows without CUDA or on CPU builds of PyTorch, `torch.compile` is unavailable, so ATLAS already logs and continues safely.

### 5.4 Checkpointing and Resume

- Checkpoints are written to `train_cfg.checkpoint_dir` (default `./checkpoints`). Each contains model weights, EMA weights, optimizer state, and the configuration bundle.
- To resume, point `--checkpoints` to the existing directory and training will continue from the latest file.

### 5.5 Estimated Training Times (Default Presets)

| Dataset (Preset) | Resolution | Default Epochs | RTX 3090 (24GB) | RTX 4090 (24GB) | RTX 5090 (24GB) | A100 80GB |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 (`experiment:cifar10`) | 32×32 | 400 | ~1.2 days | ~0.8 days | ~0.6 days | ~0.5 days |
| ImageNet 64 (`experiment:imagenet64`) | 64×64 | 600 | ~2.1 days | ~1.5 days | ~1.2 days | ~0.9 days |
| FFHQ 128 (`experiment:ffhq128`) | 128×128 | 800 | ~3.0 days | ~2.1 days | ~1.7 days | ~1.3 days |
| LSUN Bedroom 256 (`experiment:lsun256`) | 256×256 | 400 | ~4.5 days | ~3.1 days | ~2.6 days | ~2.0 days |
| CelebA-HQ 1024 (`experiment:celeba1024`) | 1024×1024 | 600 | ~8.2 days | ~5.6 days | ~4.3 days | ~3.1 days |

**Quick training for testing:**
- CIFAR-10 with `--max-steps 10000`: ~1-2 hours on RTX 4090
- ImageNet 64 with `--max-steps 10000`: ~2-3 hours on RTX 4090

_Assumes mixed-precision training, default gradient accumulation, and a single GPU running PyTorch 2.1+ with cuDNN 9.1+. Expect ±15% variance from storage throughput, dataset augmentation cost, and driver versions._

---

## 6. Running Inference on Any Platform

Use `atlas.examples.training_pipeline.run_inference` to load a checkpoint and generate images. The helper accepts a `device` override and saves PNG grids plus a manifest.

### 6.1 Quick Inference Command

```bash
python -m atlas.examples.lsun256_inference \
  --checkpoint ./checkpoints/latest.pt \
  --output ./samples \
  --device cpu
```

If you need a custom script (for example, for a different preset), construct one using the helper:

```python
from atlas.examples.training_pipeline import run_inference

run_inference(
    "experiment:lsun256",
    checkpoint_path="./checkpoints/latest.pt",
    output_dir="./samples",
    device="cpu",  # use "cuda:0" or "mps" when available
)
```

### 6.2 CPU / Low-Memory Tips

- Set `bundle["inference"] = bundle["inference"].with_overrides(batch_size=1)` for limited memory.
- Reduce the number of samples (`num_samples`) and resolution to keep runtimes reasonable.
- Disable tiling unless generating resolutions above 1024px.

---

## 7. Fine-Tuning with LoRA Adapters

LoRA (Low-Rank Adaptation) lets you adapt the ATLAS score network by training a
small set of rank-decomposed matrices while keeping the original backbone
frozen. The integration is built into the model configuration so you can toggle
it without rewriting the training loop.

### 7.1 Enable LoRA in a Preset Bundle

1. Clone a preset bundle and flip on the LoRA configuration using
   `dataclasses.replace`.
2. Register a new preset name (or call the training helper directly with the
   modified bundle).

```python
import copy
from dataclasses import replace

from atlas.config import presets
from atlas.examples.training_pipeline import run_training

bundle = presets.load_preset("experiment:lsun256")
model_cfg = bundle["model"]
model_cfg = replace(
    model_cfg,
    lora=replace(
        model_cfg.lora,
        enabled=True,
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=("to_q", "to_k", "to_v", "to_out"),
    ),
)
bundle["model"] = model_cfg

presets.PRESETS["experiment:lsun256_lora"] = lambda: copy.deepcopy(bundle)
run_training(
    "experiment:lsun256_lora",
    dataset_root="/path/to/lsun",
    checkpoint_dir="./checkpoints/lora",
    device="cuda:0",
)
```

The `LoRAConfig` dataclass (in [atlas/config/conditioning_config.py](../atlas/config/conditioning_config.py)) exposes `enabled`, `rank`, `alpha`, `dropout`, and a
tuple of `target_modules` that determine which linear layers receive LoRA
wrappers. Leave `target_modules` empty to wrap every `nn.Linear` in the score model, or provide
substrings that match the attention projections you want to adapt (the defaults
cover self- and cross-attention blocks).

### 7.2 Start from a Pretrained Checkpoint

Because LoRA keeps the underlying weights frozen, load the baseline checkpoint
before starting the fine-tuning loop:

```python
import torch

checkpoint = torch.load("./checkpoints/base/latest.pt", map_location="cpu")
score_model.load_state_dict(checkpoint["model"], strict=False)
```

Insert this snippet right after instantiating `HighResLatentScoreModel` inside a
copy of `run_training` if you need to warm-start from a full-weights model.
LoRA branches are initialized to zero so omitting the checkpoint simply keeps
the original behavior.

### 7.3 Training and Saving LoRA Parameters

- Only LoRA matrices receive gradients—the base projection weights have
  `requires_grad=False` automatically set by the wrapper.
- Lower the learning rate (e.g., `1e-4 → 5e-5`) and shorten training schedules;
  LoRA converges quickly because the parameter count is small.
- Checkpoints written by `run_training` store both the frozen backbone and the
  LoRA weights, so you can run inference with `run_inference` without extra
  merging steps.

### 7.4 Inference with LoRA

The saved checkpoint already includes the LoRA adapters. To apply them to a
base model that shipped without LoRA weights, construct the model, call
`apply_lora_to_model` with a matching config, and load the LoRA state dict:

```python
from atlas.models import HighResLatentScoreModel
from atlas.models.lora import apply_lora_to_model
import torch

model = HighResLatentScoreModel(model_cfg)
apply_lora_to_model(model, model_cfg.lora)
model.load_state_dict(torch.load("./checkpoints/lora/latest.pt")['model'])
```

As long as `model_cfg.lora` matches the configuration used during training, the
LoRA adapters will be recreated and weights loaded into the correct branches.

---

## 8. Troubleshooting Checklist

| Symptom | Likely Cause | Remedy |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'torchvision.utils'` | `vision` extras not installed | `pip install -e .[vision]` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` during training | Batch size changed but micro-batch not divisible | Ensure `batch_size % micro_batch_size == 0`. |
| Training stalls at CPU 100% with tiny steps | Running on CPU with large batch/resolution | Lower `dataset_cfg.resolution` and micro-batch size. |
| `torch.compile` warning appears | PyTorch build lacks compile backend | Set `compile=False` or ignore; ATLAS falls back automatically. |
| `torch.cuda.is_available()` false on Windows | Missing driver or running on CPU PyTorch wheel | Install NVIDIA drivers and reinstall CUDA PyTorch wheel. |
| Images save fails on macOS sandbox | Output folder lacks permissions | Use a user-writable directory (`~/Pictures/atlas_samples`). |

---

## 9. Next Steps

- Explore other presets in [atlas/config/presets.py](../atlas/config/presets.py) for different datasets and resolutions.
- Customize kernels and samplers using the advanced guides in `docs/GPU_CPU_BEHAVIOR.md` and `docs/CUDA_GRAPHS_TILING.md`.
- Share feedback or ask questions in [GitHub Discussions](https://github.com/tsoreze/OT-diffusion-toolkit/discussions).

With these steps you can install, train, and run ATLAS models on virtually any machine—even laptops without discrete GPUs—while understanding the optional performance knobs such as `torch.compile`.
