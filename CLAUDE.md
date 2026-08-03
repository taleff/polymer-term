# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**polyterm** is a Python library for analyzing termination kinetics in fast-initiating polymerizations without chain transfer. It fits experimental SEC/GPC molecular weight distribution data to kinetic models to determine termination rate constants.

## Environment

```bash
# Virtual environment location
source ~/.virtualenvs/polymer-term/bin/activate
```

## Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests with coverage
~/.virtualenvs/polymer-term/bin/pytest

# Run a single test file
~/.virtualenvs/polymer-term/bin/pytest tests/test_fit_mwd.py

# Run a specific test
~/.virtualenvs/polymer-term/bin/pytest tests/test_fit_mwd.py::test_function_name -v

# Run tests without coverage (faster)
~/.virtualenvs/polymer-term/bin/pytest --no-cov
```

## Architecture

### Package Structure

Each top-level module (except `mwd.py`) contains a single user-facing function or set of related functions. All internal logic lives in `core/`.

```
polyterm/
├── __init__.py          # Public API exports
├── fit_mwd.py           # Fit kinetic model to MWD (user-facing)
├── calculate_mwd.py     # Forward calculation of MWD from parameters (user-facing)
├── estimate_alpha.py    # Estimate alpha from conversion vs Mn data (user-facing)
├── estimate_death.py    # Dead chain fraction estimation (user-facing)
├── calibration.py       # SEC broadening calibration, EMG/EGH (user-facing)
├── mwd.py               # MWDResult dataclass
└── core/                # Internal computation modules
    ├── kinetics.py      # Analytical kinetic equations (monomer conversion, living chain DP, etc.)
    ├── kinetics_models.py # Pre-built kinetics dicts (STANDARD, ROMP) and model utilities
    ├── broadening.py    # SEC instrumental broadening (Gaussian/EMG/EGH)
    ├── distributions.py # DP range, Poisson calculations, and mass fraction helpers
    ├── initial_guess.py # Initial alpha estimation for optimizer seeding
    └── utils.py         # Utility functions (DPn, R², edge fitting)
```

### Key Concepts

1. **Reduced time**: Kinetics use dimensionless reduced time `t' = kt * t / [I]₀^(1-n)` where `kt` is termination rate, `[I]₀` is initiator concentration, and `n` is termination order.

2. **Alpha (α)**: Ratio of termination to propagation rate constants (`kt/kp`). Primary fitted parameter.

3. **Broadening models**: SEC instrumental broadening is modeled in log(MW) space:
   - Gaussian: symmetric broadening with parameter `sigma`
   - EMG (Exponentially Modified Gaussian): adds asymmetric tailing with `tau`
   - EGH (Exponential-Gaussian Hybrid): numerically stable alternative to EMG

4. **MWD calculation flow**:
   - Convert conversion → reduced time via `conversion_to_time()`
   - Compute living chain DP distribution (Poisson at `nup(t)`)
   - Integrate dead chain distribution over time using Gauss-Legendre quadrature
   - Apply broadening matrix to convert DP fractions → MW intensities

### Data Flow

```
fit_mwd() → _build_param_spec() → _create_objective() → _optimize() → _build_fit_result()
                                         ↓
                              _compute_mwd_for_fit()
                                         ↓
                    _compute_dead_chain_fracs() + _compute_live_chain_fracs()
                                         ↓
                              _compute_mwd_from_fracs()
```

### Result Classes

- **MWDResult**: Immutable container with `molecular_weights`, `intensities`, `dead_chain_intensities`, `live_chain_intensities`, `dead_chain_fraction`, fitted parameters (`alpha`, `init`, `sigma`, etc.), and `r_squared`
- **CalibrationResult**: Contains `sigma`, `tau`, `center`, `r_squared` from calibration

## Unit Consistency

All parameters must use consistent units throughout. The library is unit-agnostic but mixing units produces incorrect results. Common choices:
- Molecular weights in g/mol
- Concentrations in mol/L
- Rate constants in L/(mol·s)

## Docstring Style

**Public functions** (exported in `__init__.py`): Use numpy-style docstrings.

```python
def fit_mwd(molecular_weights, intensities, order, monomer_mw, init_mon, *,
            sigma=None, tau=None):
    """
    Fit kinetic model to a molecular weight distribution.

    Longer description with implementation details if needed.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights from SEC/GPC measurement.
    order : float
        Termination reaction order (e.g., 1.0, 1.5, 2.0).

    Returns
    -------
    MWDResult
        Dataclass containing the distribution and kinetic parameters.

    Examples
    --------
    >>> result = fit_mwd(mws, ints, order=1.5, monomer_mw=104.15, init_mon=1.0)
    """
```

**Private helper functions** (prefixed with `_`): Brief single-line or short description only.

```python
def _compute_dead_chain_fracs(time, dps, alpha, init_mon, init, order, bn,
                              combination, n_quadrature_points):
    """Compute mole fractions of dead chains at each DP using quadrature."""
```
