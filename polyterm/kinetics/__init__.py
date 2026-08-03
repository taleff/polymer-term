"""
Kinetics models for living polymerizations with termination.

This subpackage contains the fundamental kinetic equations and pre-built
kinetics dictionaries for various polymerization mechanisms.

Modules
-------
base : Core kinetic functions (living chain concentration, DP, etc.)
models : Standard kinetics model, constants, and validation utilities
romp : ROMP kinetics (first-order, second-order, multi-pathway)
combined : Combined first + second order termination kinetics
"""

from .base import (
    monomer_conversion,
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    time_to_chain_death,
)

from .models import (
    STANDARD_KINETICS,
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    CHAIN_DEATH_RATE,
    SECOND_ORDER_DEATH_RATE,
    DEFAULT_COMBINATION,
    REQUIRED_KEYS,
    validate_kinetics,
    find_chain_death_time,
)

from .romp import (
    ROMP_FIRST_ORDER_KINETICS,
    ROMP_SECOND_ORDER_KINETICS,
    make_romp_kinetics,
)

from .combined import (
    make_combined_kinetics,
)

__all__ = [
    # Base kinetics
    'monomer_conversion',
    'living_chain_concentration',
    'living_chain_dp',
    'conversion_to_time',
    'time_to_chain_death',
    # Constants
    'LIVING_CHAIN_CONC',
    'LIVING_CHAIN_DP',
    'CONVERSION_TO_TIME',
    'MONOMER_CONVERSION',
    'CHAIN_DEATH_RATE',
    'SECOND_ORDER_DEATH_RATE',
    'DEFAULT_COMBINATION',
    'REQUIRED_KEYS',
    # Standard kinetics
    'STANDARD_KINETICS',
    # ROMP kinetics
    'ROMP_FIRST_ORDER_KINETICS',
    'ROMP_SECOND_ORDER_KINETICS',
    'make_romp_kinetics',
    # Combined kinetics
    'make_combined_kinetics',
    # Utilities
    'validate_kinetics',
    'find_chain_death_time',
]
