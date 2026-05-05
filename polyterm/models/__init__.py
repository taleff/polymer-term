"""
Kinetic fitting models for polymer termination analysis.

This subpackage provides fitting models for determining termination kinetics
from molecular weight distributions.
"""

from .estimation import estimate_alpha

__all__ = [
    'estimate_alpha',
]
