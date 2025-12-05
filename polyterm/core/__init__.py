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

from .distributions import (
    gaussian_broadening,
    calculate_distribution,
    calculate_mwd,
    living_distribution_integrand,
)

__all__ = [
    # Kinetics
    'monomer_conversion',
    'living_chain_concentration',
    'living_chain_dp',
    'conversion_to_time',
    'time_to_chain_death',
    # Distributions
    'gaussian_broadening',
    'calculate_distribution',
    'calculate_mwd',
    'living_distribution_integrand',
]
