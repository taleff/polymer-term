"""
Molecular weight distribution calculation from kinetic parameters.

This module provides functions for calculating theoretical molecular
weight distributions for fast initiating chain-growth polymerizations
with termination.
"""

import numpy as np

from .kinetics.models import (
    STANDARD_KINETICS,
    validate_kinetics,
    DEFAULT_COMBINATION,
)

from .core.broadening import compute_broadening_matrix
from .mwd import MWDResult

from .core.mwd_computation import (
    poisson_distribution,
    conversion_to_reduced_time,
    compute_dead_chain_fracs,
    compute_live_chain_fracs,
    compute_mwd_from_fracs,
    compute_dead_chain_fraction,
)


__all__ = [
    'calculate_mwd',
]


def calculate_mwd(molecular_weights, monomer_mw, init_mon, alpha,
                  init, conversion, order, sigma, tau=0,
                  combination=None, bn=1.0, n_quadrature_points=200,
                  kinetics=STANDARD_KINETICS, distribution=poisson_distribution):
    """
    Calculate the molecular weight distribution from kinetics.

    Computes the weight fraction distribution including both living and
    dead chains, with SEC/GPC instrumental broadening.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which to calculate the distribution. Units
        must be consistent with monomer_mw.
    monomer_mw : float
        Molecular weight of one monomer unit. Same units as
        molecular_weights.
    init_mon : float
        Initial monomer concentration.
    alpha : float
        Ratio kt/kp of rate constants.
    init : float
        Initial initiator concentration.
    conversion : float
        Conversion of the monomer, must be between 0 to 1 where 1 is
        100% conversion.
    order : float
        Order of termination reaction.
    sigma : float
        SEC line broadening parameter.
    tau : float, optional
        SEC line broadening parameter asymmetry factor (used for
        exponential Gaussian hybrid broadening). Default 0.
    combination : float or None, optional
        Fraction of termination events that proceed by combination,
        between 0.0 (pure disproportionation) and 1.0 (pure
        combination). If None, uses the kinetics dict's
        ``default_combination`` value, or 0.0 if not present.
    bn : float, optional
        Inverse of propagation order. Default 1.0.
    n_quadrature_points : int, optional
        Number of Gauss-Legendre quadrature points for dead chain
        integration. Default 200.
    kinetics : dict, optional
        Dictionary of kinetic functions. Default is STANDARD_KINETICS.
        Keys: 'living_chain_concentration', 'living_chain_dp',
        'conversion_to_time', 'monomer_conversion'.
    distribution : callable, optional
        Function with signature ``distribution(dps, nup)`` that returns
        the probability mass at each degree of polymerization ``dps``
        given kinetic chain length ``nup``. Default is Poisson PMF.

    Returns
    -------
    MWDResult
        Dataclass containing the distribution and kinetic parameters.

    Notes
    -----
    The calculation proceeds in several steps:
    1. Calculate mole fractions at discrete DPs (living and dead)
    2. Apply Gaussian broadening to account for SEC instrumental effects
    3. Convert from number to weight distribution (multiply by DP)
    4. Normalize peak to one

    Examples
    --------
    >>> mws = np.logspace(3, 5, 1000)  # 1 kDa to 100 kDa
    >>> distribution = calculate_mwd(
    ...     mws, monomer_mw=104.15, alpha=0.001, init_mon=1.0,
    ...     init=0.01, conversion=0.8, order=1.0, sigma=0.128,
    ...     tau=0.0456
    ... )
    """
    # Validate inputs
    validate_kinetics(kinetics)
    if combination is None:
        combination = kinetics.get(DEFAULT_COMBINATION, 0.0)
    if not (0 <= combination <= 1):
        raise ValueError("combination must be between 0 and 1")

    # Calculate DP range
    max_dp = int(np.max(molecular_weights) / monomer_mw)
    dps = np.arange(1, max_dp, dtype=int)

    # Convert conversion to reduced time
    try:
        time = conversion_to_reduced_time(
            kinetics, alpha, init_mon, init, order, conversion, bn
        )
    except ValueError as e:
        raise ValueError(
            f"Cannot calculate MWD at 100% conversion with these kinetics "
            f"parameters. For ROMP kinetics, this occurs when alpha*init_mon < init. "
            f"Use a conversion < 1 instead. Original error: {e}"
        ) from e

    # Compute mole fractions
    dead_fracs = compute_dead_chain_fracs(
        time, dps, alpha, init_mon, init, order, bn,
        combination, n_quadrature_points, kinetics, distribution
    )
    live_fracs = compute_live_chain_fracs(
        time, dps, alpha, init_mon, init, order, bn, kinetics, distribution
    )

    # Compute broadening matrix
    broadenings = compute_broadening_matrix(
        molecular_weights, dps, monomer_mw, sigma, tau
    )

    # Compute MWD
    ints, dead_ints, live_ints = compute_mwd_from_fracs(
        dead_fracs, live_fracs, dps, broadenings
    )

    # Compute dead chain fraction
    dead_fraction = compute_dead_chain_fraction(dead_fracs, live_fracs)

    return MWDResult(
        molecular_weights, ints, dead_ints, live_ints, dead_fraction,
        alpha, init, order, sigma, tau, conversion
    )
