# Changelog

All notable changes to ATLAS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- FFHQ 128x128 inference script (`atlas/examples/ffhq128_inference.py`)
- ImageNet 64x64 inference script (`atlas/examples/imagenet64_inference.py`)
- Examples README summarizing scripts and usage (`atlas/examples/README.md`)
- Comprehensive documentation suite with 6 guides (QUICKSTART, DEPENDENCIES, GPU_CPU_BEHAVIOR, CUDA_GRAPHS_TILING, EXTENDING, docs index)
- Hardware capability detection module (`atlas/utils/hardware.py`)
- CLI hardware check tool (`python -m atlas.check_hardware`)
- Automatic resource awareness and optimization
- CUDA graph support with LRU cache management
- Examples README summarizing training/inference scripts and usage (`atlas/examples/README.md`)
- Native 2K generation support with tiling
- Configuration validation via `__post_init__`
- Configurable latent downsampling factor
- Memory usage warnings for large context expansion
- CI/CD workflows (GitHub Actions)
- CONTRIBUTING.md with development guidelines
- CODE_OF_CONDUCT.md (Contributor Covenant 2.1)
- Issue templates (bug report, feature request, documentation)
- This CHANGELOG.md

### Changed
- GPU profile detection now uses FREE memory instead of TOTAL memory
- 32GB profile now correctly detected (was always capping at 24GB)
- RFF cache key now includes shape, strides, and device (prevents collisions)
- Drift calculation uses conservative minimum (1e-8) instead of `finfo().tiny` for numerical stability
- CUDA graph wrapper with configurable cache size and LRU eviction
- README updated with quick start, hardware check, and documentation links

### Fixed
- **CRITICAL**: Missing closing triple-quote in `easy_api.py` module docstring (prevented compilation)
- Invalid type annotation (`any` → `Any`) in `hardware.py`
- FFT kernel `clear_cache()` now actually frees GPU memory
- Kernel cache key collision risk in Schrödinger bridge solver
- Numerical instability in conjugate gradient solver (dtype-aware epsilon)
- max_kernel_cache_size validation (now rejects 0, minimum 1)

### Security
- Added dependency security scanning in CI
- Added code quality checks (ruff, mypy)

## [0.1.0] - YYYY-MM-DD (Template)

### Added
- FFHQ 128x128 inference script (`atlas/examples/ffhq128_inference.py`)
- ImageNet 64x64 inference script (`atlas/examples/imagenet64_inference.py`)
- Examples README summarizing scripts and usage (`atlas/examples/README.md`)
- Initial release
- Score-based diffusion models
- Schrödinger bridge transport
- Multiple kernel operators (Direct, FFT, RFF, Nyström)
- Hierarchical sampler
- CLIP conditioning support
- GPU memory profiles (6GB-32GB)
- Easy API for non-experts

---

## Release Notes Guidelines

### Version Format
- **Major.Minor.Patch** (e.g., 1.2.3)
- Major: Breaking API changes
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible

### Categories
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes

### Example Entry
```markdown
## [1.2.0] - 2025-01-15

### Added
- FFHQ 128x128 inference script (`atlas/examples/ffhq128_inference.py`)
- ImageNet 64x64 inference script (`atlas/examples/imagenet64_inference.py`)
- Examples README summarizing scripts and usage (`atlas/examples/README.md`)
- Support for ControlNet conditioning (#123) @contributor
- New `adaptive_timesteps` sampling mode (#145)

### Changed
- Improved CUDA graph caching efficiency (#156)
- Updated dependencies: PyTorch 2.2.0+ required

### Fixed
- Fixed OOM error with large batch sizes (#134)
- Corrected score normalization in RFF kernels (#142)

### Contributors
Special thanks to @user1, @user2, @user3 for their contributions!
```

---

## Upgrading

### From 0.x to 1.0
- Check [MIGRATION.md](MIGRATION.md) for breaking changes
- Update import statements if APIs changed
- Review new configuration options

---

## Links
- [Documentation](docs/)
- [Contributing Guidelines](CONTRIBUTING.md)
- [GitHub Releases](https://github.com/tsoreze/OT-diffusion-toolkit/releases)
