# OT Diffusion Toolkit

The OT Diffusion Toolkit brings together four production-grade diffusion solvers
developed by Thierry Silvio Claude Soreze. Each subproject focuses on a different
aspect of optimal transport driven generative modelling and ships as an
independent Python package with its own API surface, documentation, and license.

| Component   | Focus | Highlights |
|-------------|-------|------------|
| **ATLAS** (`ATLAS/`) | Full‑stack diffusion toolkit (standalone) | Hierarchical samplers, kernel registry, CLIP guidance, CUDA graphs + tiling (1K–2K on 4090/5090) |
| **FastSB-OT** (`FastSB-OT/`) | Solver library (requires score network) | Triton kernels, Fisher-aware momentum transport, deployment-friendly presets |
| **SBDS** (`SBDS/`) | Schrödinger bridge solver (requires a pretrained score network) | Multi-tier transport (full/RFF/Nystrom/FFT), adaptive timesteps, Hilbert Sinkhorn |
| **SPOT** (`SPOT/`) | Solver library (requires score network) | Patch OT, DPM-Solver++ integrators, deterministic modes, CLI validation |

All projects follow a shared design philosophy:
- mathematically grounded derivations linked directly to the code paths,
- reproducible sampling and diagnostic tooling,
- explicit GPU memory management for consumer and datacenter hardware.

## Quick Start
1. Create a Python 3.10+ environment and install PyTorch for your platform.
2. Clone this repository and move into the desired subpackage directory.
3. Install the package in editable mode (extras are documented in each README):
   ```bash
   cd ATLAS        # or FastSB-OT, SBDS, SPOT
   pip install -e .[dev]
   ```
4. Run the local self-tests or demos:
   ```bash
   pytest                    # where available
   python -m SPOT            # SPOT validation flow
   python -m atlas.examples.basic_sampling
   ```
5. Follow the training or inference cookbooks provided in each component README.

## Project Categories

- `SPOT/`, `FastSB-OT/`, and `SBDS/` are SOLVERS. They couple a trained score network with
  Schrödinger-bridge/OT updates. You bring the score model; the solver handles
  transport, integrators, and the research-ready tiers in SBDS.
- `ATLAS/` is a full-stack DIFFUSION TOOLKIT. It ships end-to-end sampling stacks
  (models, schedules, samplers), training cookbooks, and consumer-GPU presets.

## Mathematical Background
All four solvers rely on the same core ideas:
- **Score-based diffusion**: approximate the score `nabla_x log p_t(x)` (Song et al. 2021).
- **Schrodinger bridge**: solve for the entropic OT interpolation between prior and data
  marginals using Sinkhorn-style updates (De Bortoli et al. 2021).
- **Probability flow ODE**: integrate the deterministic counterpart of the diffusion SDE.
- **Approximate kernels**: random Fourier features, FFT-based convolution, and Nystrom
  sketching provide scalable approximations to the Gaussian/Laplacian kernels appearing
  in transport maps.

Each subproject README expands on the specific assumptions, e.g. ATLAS covers hierarchical
samplers, SBDS describes its multi-tier transport, and SPOT details the partial OT setting.

## Repository Layout
```
ATLAS/       # High-resolution diffusion toolkit (Apache-2.0)
FastSB-OT/   # Fast Schrodinger bridge with optimal transport (Apache-2.0)
SBDS/        # Score-based Schrodinger diffusion solver (MIT)
SPOT/        # Schrodinger Partial Optimal Transport sampler (Apache-2.0)
```

## Licensing
- **ATLAS** – Apache License 2.0 (`ATLAS/LICENSE`)
- **FastSB-OT** – Apache License 2.0 (`FastSB-OT/LICENSE`)
- **SBDS** – MIT License (`SBDS/LICENSE`)
- **SPOT** – Apache License 2.0 (`SPOT/LICENSE`)

Please retain attribution to **Thierry Silvio Claude Soreze** in any redistribution or derivative
work across all subprojects. See the individual READMEs for additional citation suggestions.

## Contributing and Support
Bug reports, mathematical clarifications, and feature proposals are welcome. When filing issues,
please mention the relevant subpackage and include minimal repro scripts or model summaries.
Contributions should include unit tests or notebook snippets that demonstrate the improvement.

For more detailed guides, see:
- [ATLAS/README.md](ATLAS/README.md) – training/inference cookbooks and kernel registry.
- [FastSB-OT/README.md](FastSB-OT/README.md) – production pipelines and Triton deployment notes.
- [SBDS/README.md](SBDS/README.md) – mathematical derivations and tier selection algorithms.
- [SPOT/README.md](SPOT/README.md) – partial OT background and validation workflow.

All modelling, numerical analysis, and implementation work is credited to Thierry Silvio Claude
Soreze. Contributions and citations are encouraged to help advance the diffusion and optimal
transport community.


## Troubleshooting

### November 2025 placeholder files

The merge titled *Add comprehensive tests and new module files* accidentally
introduced placeholder Python files at the repository root that only contained
a UTF-8 byte-order mark. Because these files shadowed standard-library modules
(e.g. `random`), imports such as `import torch` failed. The placeholders have
been removed; please work inside the component packages (`ATLAS/`, `FastSB-OT/`,
`SBDS/`, `SPOT/`) where the actual implementations live.

