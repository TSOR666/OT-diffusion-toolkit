# Changelog

## [2025-11-25] ATLAS Maintenance Release

### Fixed
- Corrected several blockers and minor errors across the codebase 
- Updated versioning to 0.1.1
- Added missing inference scripts for ffhq128 and imagenet64


## [2025-11-19] ATLAS Maintenance Release

### Fixed
- Corrected syntax issues in conditioning and memory configs and ensured noise/power schedules align between training and sampling.
- Repaired CLIP conditioning (attention masks, context handling) and improved latent sinusoidal embeddings.
- Overhauled kernel operators (FFT, RFF, Nyström, Direct) plus gaussian blur/tiling utilities for numerical correctness.
- Updated Schrodinger Bridge solver, hierarchical sampler, and CUDA graph wrappers for stable sigma/SDE math, timestep validation, and graph caching.
- Hardened training pipeline gradient accumulation, sampler configs, and hardware detection/precision setup.
- Improved easy_api checkpoint validation, CLIP bootstrapping, prompt handling, and OOM recovery.
- Added channel-aware dataset normalization and clarified reproducibility controls.

## [2025-11-20] Solver correctness sweep

### FastSB-OT
- ensured the full OT map uses row-mass normalisation so barycentric projections honour uniform marginals and removed tensor-buffer misuse in the momentum transport module, fixing runtime errors when applying lookahead transport.

### SBDS
- resurface a `model_outputs_noise` hint and always convert scores only when the model predicts epsilon, clamp the probability-flow drift (`dx/dt = -0.5βx - βσ²score`), and bound `β(t)` near the schedule endpoints to avoid explosive values.
- stabilized Sinkhorn/RFF/Nystrom transport paths by flooring kernels, normalising plans before barycentric mapping, tying ε to data scale, and preventing degeneracies in the blockwise transport math.

### SPOT
- rewrote the Heun/Adaptive/Exponential integrators to follow the variance-preserving ODE drift, added a guarded finite-difference β estimator, and unified the exponential integrator to explicit PF-ODE stepping for consistent continuous-time behaviour.

### Packaging
- Added release archive 
elease_ATLAS_update.zip containing all touched modules for contributor reference.

