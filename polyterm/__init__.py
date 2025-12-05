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

Create a molecular weight distribution from experimental data:

>>> from polyterm import MolecularWeightDistribution
>>> mwd = MolecularWeightDistribution.from_data(
...     molecular_weights=mw_array,
...     intensities=intensity_array,
...     monomer_mw=104.15
... )

Fit termination kinetics to the distribution:

>>> from polyterm import SingleOrderModel
>>> model = SingleOrderModel(
...     monomer_mw=104.15,
...     init_mon=1.0,
...     order=1.5
... )
>>> result = model.fit(mwd)
>>> print(f"α (kt/kp) = {result.alpha:.4f}")
>>> print(f"R² = {result.r_squared:.4f}")

Generate a theoretical distribution:

>>> mwd_theory = MolecularWeightDistribution.from_kinetics(
...     molecular_weights=np.logspace(3, 6, 500),
...     monomer_mw=104.15,
...     nu=50.0,
...     alpha=0.01,
...     init_mon=1.0,
...     init=0.02,
...     order=1.5,
...     sigma=0.05
... )

Modules
-------
core : Core kinetic and distribution calculations
models : Fitting models for kinetic parameter determination
utils : Utility functions for data analysis

Classes
-------
MolecularWeightDistribution : Container for MWD data
SingleOrderModel : Main fitting model (single termination order)
FitResult : Container for fitting results
"""

from ._version import __version__

# Main user-facing classes
from .mwd import MolecularWeightDistribution
from .models import (
    SingleOrderModel,
    FitResult,
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
    # Main classes
    'MolecularWeightDistribution',
    'SingleOrderModel',
    'FitResult',
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
