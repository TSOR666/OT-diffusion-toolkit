# Contributing to ATLAS

Thank you for your interest in contributing to ATLAS! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- Git
- (Optional) CUDA-capable GPU for testing GPU features

### Setting Up Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/OT-diffusion-toolkit.git
   cd OT-diffusion-toolkit/ATLAS
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   # Install PyTorch first
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install ATLAS in editable mode with dev dependencies
   pip install -e .[dev,vision,clip]
   ```

4. **Verify installation**:
   ```bash
   python -m atlas.check_hardware
   pytest atlas/tests/ -v
   ```

### Project Structure

```
ATLAS/
├── atlas/              # Main package
│   ├── models/         # Neural network architectures
│   ├── solvers/        # Sampling algorithms
│   ├── kernels/        # Kernel operators
│   ├── conditioning/   # CLIP and other conditioning
│   ├── config/         # Configuration dataclasses
│   ├── utils/          # Utilities
│   └── tests/          # Test suite
├── docs/               # Documentation
├── examples/           # Example scripts
└── .github/            # CI/CD workflows
```

## Development Workflow

### Branching Strategy

- `main` - Stable releases only
- `develop` - Active development branch
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write code** following our [coding standards](#coding-standards)
2. **Add tests** for new functionality
3. **Update documentation** as needed
4. **Run linters** and fix any issues:
   ```bash
   ruff check atlas/
   ruff format atlas/
   mypy atlas/ --ignore-missing-imports
   ```

5. **Run tests**:
   ```bash
   pytest atlas/tests/ -v
   ```

6. **Check compilation**:
   ```bash
   python -m compileall atlas/
   ```

## Pull Request Process

### Before Submitting

- ✅ All tests pass
- ✅ Code is formatted with ruff
- ✅ Type hints added for new functions
- ✅ Documentation updated
- ✅ CHANGELOG.md updated (see [Keep a Changelog](https://keepachangelog.com/))
- ✅ No unrelated changes included

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe tests performed:
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review performed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added that prove fix/feature works
- [ ] CHANGELOG.md updated
```

### Review Process

1. Automated CI checks must pass
2. At least one maintainer approval required
3. Address review feedback promptly
4. Squash commits before merge (if requested)

## Coding Standards

### Python Style

- **PEP 8** compliant (enforced by ruff)
- **Type hints** for all public functions
- **Docstrings** for all public classes and functions (Google style)

### Example Function

```python
def compute_score(
    x: torch.Tensor,
    t: Union[float, torch.Tensor],
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    Compute the score function for diffusion sampling.

    Args:
        x: Input tensor of shape (batch, channels, height, width)
        t: Time value(s) in [0, 1]
        epsilon: Regularization parameter

    Returns:
        Score tensor of same shape as x

    Raises:
        ValueError: If t is outside [0, 1] range

    Example:
        >>> x = torch.randn(4, 3, 32, 32)
        >>> score = compute_score(x, t=0.5)
        >>> score.shape
        torch.Size([4, 3, 32, 32])
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"t must be in [0, 1], got {t}")

    # Implementation
    ...
```

### Type Hints

```python
from typing import Optional, Union, List, Tuple, Dict, Any

# Good
def process(
    data: torch.Tensor,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ...

# Avoid
def process(data, config=None):  # No type hints
    ...
```

### Error Handling

```python
# Specific exceptions
class ATLASError(Exception):
    """Base exception for ATLAS errors."""

class ConfigurationError(ATLASError):
    """Invalid configuration."""

# Usage
if epsilon <= 0:
    raise ConfigurationError(
        f"epsilon must be positive, got {epsilon}. "
        f"Check KernelConfig.epsilon parameter."
    )
```

## Testing Guidelines

### Writing Tests

Tests go in `atlas/tests/` with structure mirroring the main package:

```
atlas/tests/
├── test_models.py
├── test_solvers.py
├── test_kernels.py
└── utils/
    └── test_hardware.py
```

### Test Structure

```python
import pytest
import torch
from atlas.kernels import RFFKernelOperator

class TestRFFKernel:
    """Test suite for RFF kernel operator."""

    @pytest.fixture
    def kernel(self):
        """Create kernel instance for testing."""
        return RFFKernelOperator(
            input_dim=10,
            feature_dim=128,
            kernel_type="gaussian",
            epsilon=0.1,
            device=torch.device("cpu"),
        )

    def test_initialization(self, kernel):
        """Test kernel initializes correctly."""
        assert kernel.input_dim == 10
        assert kernel.feature_dim == 128

    def test_apply_shape(self, kernel):
        """Test output shape is correct."""
        x = torch.randn(5, 10)
        v = torch.randn(5, 3)
        result = kernel.apply(x, v)
        assert result.shape == (5, 3)

    def test_apply_deterministic(self, kernel):
        """Test operation is deterministic."""
        x = torch.randn(5, 10)
        v = torch.randn(5, 3)
        result1 = kernel.apply(x, v)
        result2 = kernel.apply(x, v)
        torch.testing.assert_close(result1, result2)

    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_various_batch_sizes(self, kernel, batch_size):
        """Test kernel works with different batch sizes."""
        x = torch.randn(batch_size, 10)
        v = torch.randn(batch_size, 3)
        result = kernel.apply(x, v)
        assert result.shape[0] == batch_size
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest atlas/tests/test_kernels.py

# Specific test
pytest atlas/tests/test_kernels.py::TestRFFKernel::test_apply_shape

# With coverage
pytest --cov=atlas --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

### Test Markers

```python
@pytest.mark.slow
def test_large_scale_sampling():
    """Test that takes >1 second."""
    ...

@pytest.mark.gpu
def test_cuda_graphs():
    """Test requiring GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    ...
```

## Documentation

### Docstring Format (Google Style)

```python
def sample(
    self,
    shape: Tuple[int, ...],
    timesteps: Union[int, List[float]],
    condition: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Sample from the diffusion model.

    This method performs ancestral sampling from the learned score function,
    optionally conditioned on provided embeddings.

    Args:
        shape: Output tensor shape (batch, channels, height, width)
        timesteps: Number of steps or custom timestep schedule
        condition: Optional conditioning dictionary with keys:
            - "context": Text embeddings [batch, seq_len, dim]
            - "mask": Attention mask [batch, seq_len]

    Returns:
        Generated samples as tensor of specified shape

    Raises:
        ValueError: If shape is invalid or timesteps < 1
        RuntimeError: If conditioning required but not provided

    Example:
        >>> sampler = create_sampler("model.pt")
        >>> images = sampler.sample(
        ...     shape=(4, 3, 256, 256),
        ...     timesteps=50,
        ... )
        >>> images.shape
        torch.Size([4, 3, 256, 256])

    Note:
        For classifier-free guidance, provide both positive and
        negative conditioning via the condition dict.
    """
```

### Adding Examples

Place examples in `examples/` with clear docstrings:

```python
"""
Example: Text-to-Image Generation with CLIP
============================================

This example shows how to generate images from text prompts
using CLIP conditioning.

Requirements:
    pip install open-clip-torch

Usage:
    python examples/text_to_image.py --prompt "a sunset" --steps 50
"""
```

### Updating Documentation

After adding features, update:
- Relevant docs in `docs/`
- Main `README.md` if user-facing
- `CHANGELOG.md` with changes
- Docstrings for new/modified functions

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** - Check if already reported
2. **Check documentation** - See if issue is addressed
3. **Test on latest version** - Update to latest release
4. **Gather information** - Hardware, versions, error messages

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the issue.

**To Reproduce**
Steps to reproduce:
1. Run '...'
2. With config '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11]
- PyTorch: [e.g., 2.2.0]
- CUDA: [e.g., 12.1]
- GPU: [e.g., RTX 4090]

**Additional context**
- Output of `python -m atlas.check_hardware`
- Full error traceback
- Minimal reproducing code
```

## Community

### Getting Help

- **Documentation**: Check [docs/](docs/) first
- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs via GitHub Issues
- **Discord**: Join our community server (link in README)

### Recognition

Contributors are recognized in:
- `CHANGELOG.md` for each release
- GitHub contributors page
- Special mentions for significant contributions

## Release Process

(For maintainers)

1. Update the version constant in `atlas/__version__.py` (pyproject reads `atlas.__version__`)
2. Update `CHANGELOG.md`
3. Create release branch: `release/vX.Y.Z`
4. Tag release: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
5. Push tag: `git push origin vX.Y.Z`
6. GitHub Actions will build and publish

---

## Questions?

- **Issues**: https://github.com/tsoreze/OT-diffusion-toolkit/issues
- **Discussions**: https://github.com/tsoreze/OT-diffusion-toolkit/discussions

Thank you for contributing to ATLAS!
