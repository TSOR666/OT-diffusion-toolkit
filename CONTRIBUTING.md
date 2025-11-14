# Contributing to OT Diffusion Toolkit

Thank you for your interest in contributing to the OT Diffusion Toolkit! This document provides guidelines for contributing to the project.

## Getting Started

The OT Diffusion Toolkit is a monorepo containing four independent packages:
- **ATLAS** - Full-stack diffusion toolkit
- **FastSB-OT** - Fast Schrödinger bridge solver
- **SBDS** - Research toolkit for Schrödinger bridges
- **SPOT** - Schrödinger Partial Optimal Transport solver

Each package can be developed and tested independently.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/tsoreze/OT-diffusion-toolkit.git
   cd OT-diffusion-toolkit
   ```

2. **Set up a development environment**
   ```bash
   # Create a Python 3.10+ environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install PyTorch for your platform
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install a package in development mode**
   ```bash
   cd ATLAS  # or FastSB-OT, SBDS, SPOT
   pip install -e .[dev]
   ```

## Making Contributions

### Bug Reports and Feature Requests

When filing issues, please:
- Mention the relevant subpackage (ATLAS, FastSB-OT, SBDS, or SPOT)
- Include minimal reproducible scripts or model summaries
- Specify your environment (Python version, PyTorch version, CUDA version, GPU model)
- For mathematical questions, reference the relevant section in the README

### Code Contributions

1. **Before starting work on a major feature:**
   - Open an issue to discuss the proposed changes
   - Ensure the feature aligns with the package's goals
   - Check that similar functionality doesn't already exist

2. **Code quality requirements:**
   - Include unit tests or notebook snippets demonstrating the improvement
   - Follow the existing code style (we use `ruff` for linting)
   - Add docstrings for public APIs
   - Update relevant documentation in the package README

3. **Testing:**
   ```bash
   # Run tests for a specific package
   cd ATLAS
   pytest

   # Run linting
   ruff check .
   ```

4. **Commit messages:**
   - Use clear, descriptive commit messages
   - Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update")
   - Reference relevant issues when applicable

5. **Pull requests:**
   - Keep changes focused and atomic
   - Update tests and documentation as needed
   - Ensure all tests pass before submitting
   - Provide a clear description of what the PR changes and why

## Mathematical Rigor

This toolkit emphasizes mathematically grounded implementations:
- When adding new algorithms, link to papers or derivations
- Include references in docstrings for non-standard techniques
- Add mathematical correctness tests where applicable (see existing test files for examples)
- Document assumptions and limitations

## Documentation

- Update the package README for user-facing changes
- Add code comments for complex mathematical operations
- Include usage examples for new features
- Keep hardware/memory guidance up to date

## Licensing and Attribution

- ATLAS, FastSB-OT, and SPOT are licensed under Apache 2.0
- SBDS is licensed under MIT
- All contributions must retain attribution to Thierry Silvio Claude Soreze
- By contributing, you agree that your contributions will be licensed under the same license as the package you're contributing to

## Code of Conduct

- Be respectful and constructive in all interactions
- Focus on the technical merits of contributions
- Help create a welcoming environment for contributors of all skill levels

## Questions?

If you have questions about contributing, please:
- Check the package-specific README files
- Review existing issues and pull requests
- Open a new issue with the "question" label

---

All modeling, numerical analysis, and implementation work is credited to Thierry Silvio Claude Soreze.
Contributions and citations are encouraged to help advance the diffusion and optimal transport community.
