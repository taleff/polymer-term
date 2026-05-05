"""
Molecular weight distribution calculation from kinetic parameters.

This module provides functions for calculating theoretical molecular
weight distributions for fast initiating chain-growth polymerizations
with termination.
"""

import numpy as np
from scipy.stats import poisson

from .core.kinetics_models import (
    STANDARD_KINETICS,
    validate_kinetics,
    find_chain_death_time,
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    CHAIN_DEATH_RATE,
)

from .core.distributions import get_poisson_dp_range

from .core.broadening import compute_broadening_matrix
from .mwd.mwd import MWDResult


__all__ = [
    'calculate_mwd',
    # Helper functions
    '_get_quadrature_points',
    '_compute_dead_chain_fracs',
    '_compute_live_chain_fracs',
    '_compute_mwd_from_fracs',
    '_compute_dead_chain_fraction',
]


# At conversions close to one define a maximum conversion close enough
# to one to avoid computational problems
CHAIN_DEATH_FRACTION = 0.9999


def _poisson_distribution(dps, nup):
    """Default Poisson distribution for living chain length."""
    return poisson.pmf(dps, nup)


def calculate_mwd(molecular_weights, monomer_mw, init_mon, alpha,
                  init, conversion, order, sigma, tau=0,
                  combination=0.0, bn=1.0, n_quadrature_points=40,
                  kinetics=STANDARD_KINETICS, distribution=_poisson_distribution):
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
    combination : float, optional
        Fraction of termination events that proceed by combination,
        between 0.0 (pure disproportionation) and 1.0 (pure
        combination). Default 0.0.
    bn : float, optional
        Inverse of propagation order. Default 1.0.
    n_quadrature_points : int, optional
        Number of Gauss-Legendre quadrature points for dead chain
        integration. Default 40.
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
    if not (0 <= combination <= 1):
        raise ValueError("combination must be between 0 and 1")

    # Calculate DP range
    max_dp = int(np.max(molecular_weights) / monomer_mw)
    dps = np.arange(1, max_dp, dtype=int)

    # Convert conversion to reduced time
    if np.isclose(conversion, 1):
        try:
            time = find_chain_death_time(
                kinetics, alpha, init_mon, init, order, CHAIN_DEATH_FRACTION, bn
            )
        except ValueError as e:
            raise ValueError(
                f"Cannot calculate MWD at 100% conversion with these kinetics "
                f"parameters. For ROMP kinetics, this occurs when alpha*init_mon < init. "
                f"Use a conversion < 1 instead. Original error: {e}"
            ) from e
    else:
        time = kinetics[CONVERSION_TO_TIME](alpha, init_mon, init, order, conversion, bn)

    # Compute mole fractions
    dead_fracs = _compute_dead_chain_fracs(
        time, dps, alpha, init_mon, init, order, bn,
        combination, n_quadrature_points, kinetics, distribution
    )
    live_fracs = _compute_live_chain_fracs(
        time, dps, alpha, init_mon, init, order, bn, kinetics, distribution
    )

    # Compute broadening matrix
    broadenings = compute_broadening_matrix(
        molecular_weights, dps, monomer_mw, sigma, tau
    )

    # Compute MWD
    ints, dead_ints, live_ints = _compute_mwd_from_fracs(
        dead_fracs, live_fracs, dps, broadenings
    )

    # Compute dead chain fraction
    dead_fraction = _compute_dead_chain_fraction(dead_fracs, live_fracs)

    return MWDResult(
        molecular_weights, ints, dead_ints, live_ints, dead_fraction,
        alpha, init, order, sigma, tau, conversion
    )


def _get_quadrature_points(n_points, time_end):
    """
    Get Gauss-Legendre quadrature nodes and weights scaled to [0, time_end].
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    scaled_nodes = (nodes + 1) * time_end / 2
    scaled_weights = weights * time_end / 2
    return scaled_nodes, scaled_weights


def _compute_dead_chain_fracs(time, dps, alpha, init_mon, init, order,
                              bn, combination, n_quadrature_points, kinetics,
                              distribution=_poisson_distribution):
    """Compute mole fractions of dead chains at each DP using quadrature."""
    # Get quadrature points
    quad_t, quad_w = _get_quadrature_points(n_quadrature_points, time)

    # Compute kinetic chain length of the live chain distribution at each time
    nups = np.array([
        kinetics[LIVING_CHAIN_DP](alpha, init_mon, init, order, t, bn)
        for t in quad_t
    ])

    # Compute chain death rate at each time point using the kinetics-specific
    # death rate function (accounts for different termination mechanisms)
    integrand_weights = np.array([
        kinetics[CHAIN_DEATH_RATE](alpha, init_mon, init, order, t, bn)
        for t in quad_t
    ])

    # Check for invalid kinetics parameters (NaN in nups or integrand_weights)
    if not np.all(np.isfinite(nups)) or not np.all(np.isfinite(integrand_weights)):
        raise ValueError(
            "Invalid kinetics parameters: NaN or Inf encountered in "
            "chain DP or death rate calculations. This typically occurs when "
            "the fitted alpha value is outside the valid range for the kinetics model."
        )

    # Compute dead chain fractions, blending disproportionation and
    # combination contributions based on the combination fraction.
    # For Poisson distributions, Poisson(a) * Poisson(b) = Poisson(a+b),
    # so combination chains at DP = 2*nup use the same Poisson formula.
    if combination <= 0.0:
        # Pure disproportionation
        idx_end = get_poisson_dp_range(np.max(nups), dps)
        dist_matrix = np.zeros((len(nups), len(dps)), dtype=float)
        dist_matrix[:, :idx_end] = distribution(
            dps[:idx_end][np.newaxis, :], nups[:, np.newaxis]
        )
        dead_fracs = (quad_w * integrand_weights) @ dist_matrix
    elif combination >= 1.0:
        # Pure combination: doubled DP, halved event count
        idx_end = get_poisson_dp_range(np.max(2 * nups), dps)
        dist_matrix = np.zeros((len(nups), len(dps)), dtype=float)
        dist_matrix[:, :idx_end] = distribution(
            dps[:idx_end][np.newaxis, :], (2 * nups)[:, np.newaxis]
        )
        dead_fracs = (quad_w * integrand_weights / 2) @ dist_matrix
    else:
        # Mixed: blend disproportionation and combination
        disp_idx = get_poisson_dp_range(np.max(nups), dps)
        disp_matrix = np.zeros((len(nups), len(dps)), dtype=float)
        disp_matrix[:, :disp_idx] = distribution(
            dps[:disp_idx][np.newaxis, :], nups[:, np.newaxis]
        )
        comb_idx = get_poisson_dp_range(np.max(2 * nups), dps)
        comb_matrix = np.zeros((len(nups), len(dps)), dtype=float)
        comb_matrix[:, :comb_idx] = distribution(
            dps[:comb_idx][np.newaxis, :], (2 * nups)[:, np.newaxis]
        )
        disp_fracs = (quad_w * integrand_weights) @ disp_matrix
        comb_fracs = (quad_w * integrand_weights / 2) @ comb_matrix
        dead_fracs = (1 - combination) * disp_fracs + combination * comb_fracs

    return dead_fracs


def _compute_live_chain_fracs(time, dps, alpha, init_mon, init, order, bn,
                              kinetics, distribution=_poisson_distribution):
    """Compute mole fractions of living chains at each DP."""
    # Calculating the kinetic chain length and concentration of the
    # living chains
    b = kinetics[LIVING_CHAIN_CONC](alpha, init_mon, init, order, time, bn)
    nup = kinetics[LIVING_CHAIN_DP](alpha, init_mon, init, order, time, bn)

    # Use optimized computation (only computes in relevant range)
    idx_end = get_poisson_dp_range(nup, dps)
    result = np.zeros(len(dps), dtype=float)
    result[:idx_end] = distribution(dps[:idx_end], nup)

    return b * result


def _compute_mwd_from_fracs(dead_fracs, live_fracs, dps, broadenings):
    """
    Compute broadened MWD from mole fraction distributions.

    Converts mole fractions to weight fractions, applies broadening,
    and normalizes.
    """
    # Convert to weight distribution and apply broadening
    raw_mwd = np.matmul(broadenings, (live_fracs + dead_fracs) * dps)

    # Normalize the peak to one and calculate each distribution
    norm = np.max(raw_mwd)
    intensities = raw_mwd / norm
    dead_intensities = np.matmul(broadenings, dead_fracs * dps) / norm
    live_intensities = np.matmul(broadenings, live_fracs * dps) / norm

    return intensities, dead_intensities, live_intensities


def _compute_dead_chain_fraction(dead_fracs, live_fracs):
    """
    Compute fraction of chains that are dead.
    """
    total = np.sum(live_fracs + dead_fracs)
    return np.sum(dead_fracs) / total

