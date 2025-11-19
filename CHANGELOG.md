# Changelog

## [2025-11-19] ATLAS Maintenance Release

### Fixed
- Corrected syntax issues in conditioning and memory configs and ensured noise/power schedules align between training and sampling.
- Repaired CLIP conditioning (attention masks, context handling) and improved latent sinusoidal embeddings.
- Overhauled kernel operators (FFT, RFF, Nyström, Direct) plus gaussian blur/tiling utilities for numerical correctness.
- Updated Schrodinger Bridge solver, hierarchical sampler, and CUDA graph wrappers for stable sigma/SDE math, timestep validation, and graph caching.
- Hardened training pipeline gradient accumulation, sampler configs, and hardware detection/precision setup.
- Improved easy_api checkpoint validation, CLIP bootstrapping, prompt handling, and OOM recovery.
- Added channel-aware dataset normalization and clarified reproducibility controls.

### Packaging
- Added release archive 
elease_ATLAS_update.zip containing all touched modules for contributor reference.

