# ATLAS Library – Bug Fix & Optimization Changelog

**Status**: Production Ready 🟢  
**Date**: November 20, 2025
**Reviewed by** : Thierry Silvio Claude

This document summarizes the critical mathematical corrections, algorithmic improvements, and safety mechanisms implemented to bring the **ATLAS** diffusion library to a production-ready state.

---

## 1. 🧮 Core Mathematical Corrections  
*Fixes that ensure the model trains on correct targets and respects diffusion theory.*

| Component | File | Issue | Resolution |
| :--- | :--- | :--- | :--- |
| **Noise Variance** | `training_pipeline.py` | Used incorrect SNR-based formula $\sigma = \sqrt{(1-\alpha)/\alpha}$. | **Fixed**: Updated to standard DDPM variance $\sigma = \sqrt{1-\alpha}$, ensuring stable and theoretically correct training. |
| **Sigma Computation** | `schrodinger_bridge.py` | Inherited the same incorrect SNR formula for $\sigma$. | **Fixed**: Aligned $\sigma$ with the training variance definition for consistent time steps between training and sampling. |
| **Time Embeddings** | `embeddings.py` | Frequency calculation divided by `half_dim - 1`, causing an off-by-one frequency scaling error. | **Fixed**: Changed divisor to `half_dim` to match standard sinusoidal positional encodings (Vaswani et al.). |
| **Cosine Schedule** | `timesteps.py` | Implementation deviated from the reference cosine schedule. | **Fixed**: Replaced with the exact formula from *Nichol & Dhariwal (2021)* to preserve signal at $t = 0$ and match theory. |
| **Gaussian Kernel** | `fft.py` | Gaussian bandwidth formula missed the variance square term. | **Fixed**: Corrected from $\exp(-d^2 / 2\epsilon)$ to $\exp(-d^2 / 2\epsilon^2)$ for a properly parameterized Gaussian kernel. |
| **Nyström Error** | `nystrom.py` | Error bound formula incorrectly suggested that more samples increase error. | **Fixed**: Corrected to the standard $O(1/\sqrt{m})$ dependence, where $m$ is the number of landmarks. |

---

## 2. ⚙️ Algorithmic & Logic Repairs  
*Fixes preventing runtime failures, crashes, or silent data corruption.*

| Component | File | Issue | Resolution |
| :--- | :--- | :--- | :--- |
| **Gradient Accumulation Data Loss** | `training_pipeline.py` | Final partial batch was dropped if it didn’t align with `accum_steps`. | **Fixed**: Added logic to apply gradients for the remainder batch at the end of each epoch. |
| **RFF Normalization** | `rff.py` | Multi-scale features normalized by the *total* feature count, distorting per-scale variance. | **Fixed**: Each scale is now normalized by its *own* feature count, preserving unit variance per scale. |
| **Cache Collisions** | `rff.py` | Used `tensor.sum()` as a cache key, allowing collisions between different tensors. | **Fixed**: Switched to `tensor.data_ptr()` as a unique identifier for tensors in memory, eliminating collisions. |
| **Timestep Safety** | `hierarchical_sampler.py` | Schedule ended at exactly `0.0`, risking division-by-zero and degenerate noise levels. | **Fixed**: Adjusted endpoint to `0.01`, keeping noise close to zero but numerically safe. |
| **LoRA Dropout Placement** | `lora.py` | Dropout applied between low-rank matrices, disrupting the learned adapter mapping. | **Fixed**: Dropout now applied to the *final* adapter output, preserving the low-rank factorization. |
| **Tiling Seams** | `tiling.py` | Window function clamped prematurely, causing visible seams at tile boundaries. | **Fixed**: Implemented a smooth linear ramp window that blends cleanly to zero at tile edges. |
| **Attention Mask Logic** | `attention.py` | Inverted mask logic resulted in attention to padding instead of ignoring it. | **Fixed**: Mask semantics inverted so that padding tokens are correctly excluded from attention. |

---

## 3. 🛡️ Safety, Robustness & Usability  
*Improvements that prevent crashes and enhance developer experience.*

### API & Configuration

- **Memory Estimation (`easy_api.py`)**  
  Replaced naive rule `150 + batch*250` with a model-aware estimation that accounts for network depth, width, and resolution scaling to give more reliable batch size suggestions.

- **Checkpoint Safety (`easy_api.py`)**  
  Added strict key and shape validation before loading checkpoints to fail fast on incompatible weights instead of silently misloading parameters.

- **Hardware Auto-Tuning (`hardware.py`)**  
  Implemented automatic detection of BF16/TF32 support and conservative batch size recommendations to reduce out-of-memory errors across heterogeneous hardware.

### Runtime Safety

- **CUDA Graph Capture (`cuda_graphs.py`)**  
  Added signature inspection to automatically disable CUDA graphs for models with dynamic or incompatible arguments (e.g., variable conditioning inputs).

- **Determinism Controls (`random.py`)**  
  Made cuDNN deterministic mode optional and documented the tradeoff between determinism and performance so users can choose behavior explicitly.

---

## 4. 🧹 Syntax & Code Hygiene  
*Cleanup of errors preventing interpretation or efficient execution.*

- **`memory.py`**: Fixed missing closing parenthesis in `warnings.warn(...)`.  
- **`nystrom.py`**: Removed superfluous closing parenthesis in a return statement.  
- **`image_ops.py`**: Replaced a slow Python loop over channels with an optimized grouped convolution implementation.  
- **`rff.py`**: Removed unsupported `"Cauchy"` kernel type that lacked a valid Fourier representation in the current implementation.

---

## ✅ Summary

The **ATLAS** diffusion library is now **production ready**.  
All identified mathematical inconsistencies have been aligned with the theoretical literature, algorithmic edge cases have been fixed to avoid silent failures, and additional safety checks and hardware-aware heuristics have been added to support reliable deployment in real training and inference pipelines.
