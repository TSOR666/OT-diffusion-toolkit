# ATLAS Change Summary

This document captures the end-to-end work that brought ATLAS to a production-ready baseline. It reflects the latest
additions across documentation, user experience, training workflows, infrastructure, and quality safeguards.

---

## 1. Documentation & Onboarding

ATLAS now ships with a structured documentation suite that meets users where they are:

- **Entry points** – The docs index links straight to installation, quick usage, and troubleshooting material so new users can get moving immediately.
- **Platform-aware install & usage guide** – *How to Train and Run ATLAS Anywhere* walks through Windows, Linux, macOS, CPU-only, and mixed-accelerator setups while explaining preset tweaks, torch.compile fallbacks, and end-to-end workflows.
- **Deep dives** – Dedicated references cover dependency management, GPU/CPU behavior, CUDA graphs & tiling, and extension best practices so advanced users have authoritative guidance without digging into source first.

## 2. Training & Inference Workflows

Training and evaluation are no longer bespoke scripts; they are reusable workflows backed by presets:

- **Preset bundles** encapsulate dataset configuration, training hyperparameters, inference defaults, and kernel/sampler choices for LSUN Bedroom 256×256 and CelebA-HQ 1024×1024, including gradient accumulation, EMA, and compile toggles for mixed hardware environments.
- **Shared training pipeline** handles seed control, gradient scaling, EMA updates, checkpointing, and optional torch.compile usage with automatic fallbacks when compilation fails.
- **CLI entry points** (`atlas.examples.*`) expose those pipelines behind simple commands so teams can run dataset-specific experiments without writing glue code.
- **Dataset utilities** build torchvision-based dataloaders with configurable transforms and fallbacks, making it easy to override dataset roots or swap in synthetic data for smoke tests.

## 3. Easy API & Resource Awareness

The simplified API now goes beyond presets and actively adapts to the machine it is running on:

- **GPU-aware profiles** encode batch sizes, precision modes, kernel solvers, and CLIP availability for 6–32 GB devices and are automatically selected (or overridden) inside `create_sampler` to keep sampling ergonomic.
- **Hardware detection helpers** report compute capability, precision support, free vs. total memory, and CUDA graph eligibility, with safe fallbacks when `torch.cuda.mem_get_info` is missing.
- **Configuration validation** blocks incompatible CLIP/context combinations, warns about risky kernel parameters, and estimates memory usage against the active GPU profile; the test suite exercises these failures to prevent regressions.
- **Memory visibility** exposes peak-tracking helpers and high-memory warnings so interactive sessions surface problems before they crash.

## 4. Kernel, Solver, and Sampler Hardening

Runtime reliability was improved through targeted fixes:

- The CUDA graph wrapper now warms models, caches captures per-shape, and evicts least-recently-used graphs to prevent unbounded GPU memory growth.
- Schrödinger bridge solver caches, epsilon handling, and kernel keys were tightened to eliminate shape collisions and numerical instability.
- Attention modules and configuration dataclasses gained validation hooks and resource warnings so extreme settings surface actionable guidance earlier in the workflow.

## 5. Quality Gates & Tooling

Confidence in releases is anchored by layered checks:

- **CPU-only validation script** (`validate_atlas.py`) exercises imports, miniature models, sampler runs, and kernel tests to guarantee the package works on developer laptops before hitting CI.
- **Cross-platform tests** cover configs, solvers, conditioning, kernels, and integration sampling to ensure the simplified API stays correct across upgrades.
- **CI workflows** run lint, type checks, pytest (on Linux, macOS, Windows, PyTorch CPU wheels), import smoke tests, and documentation link checks for every push/PR.

## 6. Community & Governance

To support collaboration and long-term maintenance:

- **Contributing guide** documents environment setup, branch strategy, coding standards, lint/test commands, and release expectations.
- **Code of Conduct** adopts the Contributor Covenant to set participation expectations and enforcement paths.
- **Issue & PR templates plus release automation** live under `.github`, standardizing bug reports, feature requests, and changelog hygiene.

---

## Next Steps

1. Continue expanding real-world training presets (e.g., diffusion transformers, ControlNet adapters).
2. Ship GPU-enabled CI lanes for smoke-testing CUDA graph capture on supported runners.
3. Capture performance baselines in docs to quantify improvements after each release.
