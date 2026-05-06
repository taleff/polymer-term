"""
Core kinetic and distribution calculation functions.

This subpackage contains the fundamental mathematical models for living
polymerizations with termination. Most users will not need to import from
this module directly, as the high-level API provides more convenient access.
"""

from .kinetics import (
    monomer_conversion,
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    time_to_chain_death,
)

from .kinetics_models import (
    STANDARD_KINETICS,
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    CHAIN_DEATH_RATE,
    SECOND_ORDER_DEATH_RATE,
    REQUIRED_KEYS,
    validate_kinetics,
    find_chain_death_time,
    make_combined_kinetics,
)

from .distributions import (
    calculate_dp_range,
    get_poisson_dp_range,
    compute_poisson_mass_fracs,
)

from .broadening import (
    gaussian_broadening,
    emg_broadening,
    egh_broadening,
    compute_broadening_matrix,
)

__all__ = [
    # Kinetics
    'monomer_conversion',
    'living_chain_concentration',
    'living_chain_dp',
    'conversion_to_time',
    'time_to_chain_death',
    # Kinetics models
    'STANDARD_KINETICS',
    'LIVING_CHAIN_CONC',
    'LIVING_CHAIN_DP',
    'CONVERSION_TO_TIME',
    'MONOMER_CONVERSION',
    'REQUIRED_KEYS',
    'validate_kinetics',
    'find_chain_death_time',
    # Broadening
    'gaussian_broadening',
    'emg_broadening',
    'egh_broadening',
    'compute_broadening_matrix',
    # Distributions
    'calculate_dp_range',
]
