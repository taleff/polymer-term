"""
Kinetic fitting models for polymer termination analysis.

This subpackage provides fitting models for determining termination kinetics
from molecular weight distributions.
"""

from .fitting import (
    fit_mwd,
    FitResult,
    fit_living_peak,
    LivingPeakResult,
    estimate_living_fraction,
    LivingFractionResult,
)
from .estimation import estimate_alpha

__all__ = [
    # Functional API (recommended)
    'fit_mwd',
    'FitResult',
    'fit_living_peak',
    'LivingPeakResult',
    'estimate_living_fraction',
    'LivingFractionResult',
    'estimate_alpha',
]
