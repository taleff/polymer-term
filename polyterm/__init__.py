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

Classes
-------
MolecularWeightDistribution : Container for MWD data
FitResult : Container for fitting results
"""

from ._version import __version__

# Main user-facing classes and functions
from .mwd import MolecularWeightDistribution
from .models import (
    fit_mwd,
    FitResult,
    fit_living_peak,
    LivingPeakResult,
    estimate_living_fraction,
    LivingFractionResult,
)
from .calibration import (
    calibrate_emg_broadening,
    calibrate_egh_broadening,
    EMGCalibrationResult,
    EGHCalibrationResult,
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
    calculate_mwd,
    calculate_distribution,
)

__all__ = [
    # Version
    '__version__',
    # Main API (recommended)
    'fit_mwd',
    'FitResult',
    'fit_living_peak',
    'LivingPeakResult',
    'estimate_living_fraction',
    'LivingFractionResult',
    'MolecularWeightDistribution',
    # Calibration
    'calibrate_emg_broadening',
    'calibrate_egh_broadening',
    'EMGCalibrationResult',
    'EGHCalibrationResult',
    # Utilities
    'calculate_number_average_dp',
    'fit_right_edge',
    'calculate_r_squared',
    # Core functions
    'monomer_conversion',
    'living_chain_concentration',
    'living_chain_dp',
    'conversion_to_time',
    'calculate_mwd',
    'calculate_distribution',
]

# Package metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"
