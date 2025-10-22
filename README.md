# OT Diffusion Toolkit

The OT Diffusion Toolkit collects three research-ready diffusion components from
Thierry Silvio Claude Soreze:

- **ATLAS** (`ATLAS/`) – a modular high-resolution diffusion framework with
  Schrodinger bridge samplers and kernel operators.
- **FastSB-OT** (`FastSB-OT/`) – a production-focused Schrodinger bridge solver
  with adaptive optimal transport refinements and Triton acceleration.
- **SBDS** (`SBDS/`) – an enhanced score-based Schrodinger diffusion solver with
  Hilbert Sinkhorn divergences and random Fourier feature kernels.

Each component now ships as a dedicated Python package with its own README,
usage examples, and license.

## Getting Started
1. Create and activate a Python 3.10+ virtual environment.
2. Install PyTorch for your platform (CUDA, ROCm, or CPU).
3. Install any additional dependencies listed in the subproject you want to
   use (`pip install -r requirements.txt` if provided, or `pip install -e .`).
4. Follow the sampling or training guides in the corresponding README:
   - [ATLAS/README.md](ATLAS/README.md)
   - [FastSB-OT/README.md](FastSB-OT/README.md)
   - [SBDS/README.md](SBDS/README.md)

## Repository Structure
```
ATLAS/       # High-resolution diffusion toolkit
FastSB-OT/   # Fast Schrodinger bridge with optimal transport
SBDS/        # Score-based Schrodinger diffusion solver
```

## Licensing
- **ATLAS** – Apache License 2.0 (`ATLAS/LICENSE`)
- **FastSB-OT** – Apache License 2.0 (`FastSB-OT/LICENSE`)
- **SBDS** – MIT License (`SBDS/LICENSE`)

Please retain attribution to **Thierry Silvio Claude Soreze** in any
redistribution or derivative work across all subprojects.

## Acknowledgements
All modelling, numerical analysis, and implementation work is credited to
Thierry Silvio Claude Soreze. Contributions and citations are welcome.
