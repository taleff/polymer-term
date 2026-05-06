"""
polyterm: Polymer Termination Kinetics Analysis

A Python library for analyzing termination rates in fast-initiating polymerizations.
Provides tools for generating molecular weight distributions from kinetic parameters
and fitting experimental SEC/GPC data to determine termination kinetics.

IMPORTANT: All parameters must use consistent units. For example, if monomer
molecular weight is in g/mol, all mass units must be in grams, all concentrations
must use the same volume basis, and rate constants must be compatible with those
units.

Basic Usage
-----------

Fit termination kinetics using the functional API:

>>> from polyterm import fit_mwd
>>> result = fit_mwd(
...     molecular_weights, intensities,
...     order=1.5,
...     monomer_mw=104.15,
...     init_mon=1.0
... )
>>> print(f"alpha (kt/kp) = {result.alpha:.4f}")
>>> print(f"R^2 = {result.r_squared:.4f}")

With calibrated broadening parameters:

>>> result = fit_mwd(
...     molecular_weights, intensities,
...     order=1.5,
...     monomer_mw=104.15,
...     init_mon=1.0,
...     sigma=0.05,
...     tau=0.02
... )

Batch processing with functools.partial:

>>> from functools import partial
>>> fit_my_instrument = partial(
...     fit_mwd,
...     monomer_mw=104.15,
...     init_mon=1.0,
...     sigma=0.05
... )
>>> results = [fit_my_instrument(mws, ints, order=1.5) for mws, ints in samples]

Modules
-------
core : Core kinetic and distribution calculations
models : Fitting models for kinetic parameter determination
utils : Utility functions for data analysis

Functions
---------
fit_mwd : Fit kinetic model to molecular weight distribution (recommended)
calculate_mwd : Calculate theoretical MWD from kinetic parameters
estimate_death : Estimate dead chain fraction from experimental MWD
estimate_alpha : Estimate alpha (kt/kp) from conversion vs living Mn data

Classes
-------
MWDResult : Immutable container for MWD data and parameters
"""

from ._version import __version__

# Main user-facing classes and functions
from .mwd import MWDResult
from .fit_mwd import fit_mwd

from .calculate_mwd import calculate_mwd
from .estimate_death import estimate_death
from .estimate_alpha import estimate_alpha
from .calibration import (
    calibrate_emg_broadening,
    calibrate_egh_broadening,
    CalibrationResult,
    compute_poisson_broadened_mwd,
)

# Utility functions
from .utils import (
    calculate_number_average_dp,
    fit_right_edge,
    calculate_r_squared,
)

# Core functions (for advanced users)
from .core import (
    monomer_conversion,
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
)

# Kinetics models (for custom kinetics)
from .core import (
    STANDARD_KINETICS,
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    validate_kinetics,
)

__all__ = [
    # Version
    '__version__',
    # Main API (recommended)
    'fit_mwd',
    'calculate_mwd',
    'estimate_death',
    'estimate_alpha',
    'MWDResult',
    # Calibration
    'calibrate_emg_broadening',
    'calibrate_egh_broadening',
    'CalibrationResult',
    'compute_poisson_broadened_mwd',
    # Utilities
    'calculate_number_average_dp',
    'fit_right_edge',
    'calculate_r_squared',
    # Core functions
    'monomer_conversion',
    'living_chain_concentration',
    'living_chain_dp',
    'conversion_to_time',
    # Kinetics models (for custom kinetics)
    'STANDARD_KINETICS',
    'LIVING_CHAIN_CONC',
    'LIVING_CHAIN_DP',
    'CONVERSION_TO_TIME',
    'MONOMER_CONVERSION',
    'validate_kinetics',
]

# Package metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"
