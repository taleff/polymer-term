"""
Kinetic fitting models for polymer termination analysis.

This subpackage provides fitting models for determining termination kinetics
from molecular weight distributions.
"""

from .base import BaseModel, FitResult
from .single_order import SingleOrderModel
from .multi_order import MultiOrderModel
from .global_fit import GlobalFitModel

__all__ = [
    'BaseModel',
    'FitResult',
    'SingleOrderModel',
    'MultiOrderModel',
    'GlobalFitModel',
]
