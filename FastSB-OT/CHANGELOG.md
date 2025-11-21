# Changelog

All notable changes to FastSB-OT will be documented in this file.

## [Unreleased] - 2025-11-21

### Fixed

#### Variable Naming Clarity in DDPM/DDIM Steps
- **CRITICAL FIX**: Renamed variables in `ddpm_step_improved` and `ddim_step` for mathematical clarity
  - `alpha_bar_t` → `alpha_bar_curr` (α̅ at current timestep)
  - `alpha_bar_next` → `alpha_bar_prev` (α̅ at previous/less-noisy timestep)
  - This resolves the confusing naming where "next" in denoising direction meant "previous" in forward diffusion

- **Mathematical Correctness**: Verified and documented that posterior mean and variance computations are mathematically correct
  - DDPM posterior variance: `σ²_t = β_t · (1 - α̅_{t-1}) / (1 - α̅_t)` ✅
  - DDPM posterior mean coefficients are correct ✅
  - DDIM step implementation is correct ✅

#### Documentation Improvements
- Added comprehensive inline documentation explaining:
  - The denoising direction (t_curr → t_next where t_next < t_curr)
  - Relationship between forward and reverse process notation
  - Clear variable naming conventions

- Added detailed function docstrings with Args sections for:
  - `ddpm_step_improved()`
  - `ddim_step()`

### Added

#### Comprehensive Test Suite
- **26 new tests** covering:
  - Noise schedule monotonicity and correctness
  - DDPM step shape preservation, determinism, and variance positivity
  - DDIM step shape preservation, determinism, and stochasticity
  - Full sampling pipeline (both DDPM and DDIM modes)
  - Optimal transport operations (sliced OT, full OT, Sinkhorn)
  - Numerical stability at extreme timesteps
  - Tensor reshaping utilities

#### Code Review Documentation
- Added [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) with comprehensive analysis:
  - Mathematical correctness verification
  - Identified issues and their resolutions
  - Code quality assessment
  - Recommendations for future improvements

### Details

The main mathematical implementations were found to be **correct**, but the variable naming was confusing:

**Before:**
```python
alpha_bar_t = self._get_cached_noise_schedule(t_curr)
alpha_bar_next = self._get_cached_noise_schedule(t_next)
alpha_bar_prev = alpha_bar_next if t_next > 0 else 1.0
```

**After:**
```python
alpha_bar_curr = self._get_cached_noise_schedule(t_curr)
alpha_bar_prev = self._get_cached_noise_schedule(t_next) if t_next > 0 else 1.0
```

This makes it immediately clear that:
- `alpha_bar_curr` is α̅ at the current (noisy) state
- `alpha_bar_prev` is α̅ at the previous (less noisy) state in forward diffusion
- The denoising process moves from t_curr (noisy) → t_next (cleaner)

### Testing

All 26 tests pass successfully:
```bash
pytest tests/ -v
============================= 26 passed in 2.11s ==============================
```

Tests cover:
- ✅ Noise schedule properties
- ✅ DDPM step correctness
- ✅ DDIM step correctness
- ✅ Sampling pipeline
- ✅ Optimal transport components
- ✅ Numerical stability

### Migration Guide

**No breaking changes** - the behavior is identical, only variable names changed internally.

If you've subclassed `FastSBOTSolver` and overridden `ddpm_step_improved()` or `ddim_step()`, you'll need to update variable names:
- `alpha_bar_t` → `alpha_bar_curr`
- `alpha_bar_next` → `alpha_bar_prev` (when it refers to the less noisy timestep)


