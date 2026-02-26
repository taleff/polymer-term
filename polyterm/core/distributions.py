"""
Molecular weight distribution calculations for living polymerizations.

This module provides functions for calculating theoretical molecular weight
distributions for controlled polymerizations with termination.
"""

import numpy as np
import warnings
from scipy.stats import poisson
from scipy.integrate import quad_vec
from typing import Tuple

from .kinetics import (
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    time_to_chain_death,
)
from .broadening import emg_broadening, egh_broadening

__all__ = [
    'calculate_dp_range',
    'calculate_distribution',
    'calculate_mwd',
    'living_distribution_integrand',
]


def calculate_dp_range(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    monomer_mw: float,
    nu: float,
    nup: float
) -> np.ndarray:
    """
    Calculate intelligent DP range for distribution calculation.

    Determines an appropriate range of degrees of polymerization that
    balances computational cost with coverage of the distribution mass.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights from the experimental distribution.
    intensities : ndarray
        Intensity values at each molecular weight.
    monomer_mw : float
        Molecular weight of one monomer unit.
    nu : float
        Number average degree of polymerization.
    nup : float
        Peak degree of polymerization (living chain DP estimate).

    Returns
    -------
    ndarray
        Array of integer DP values from 1 to max_dp.

    Notes
    -----
    The maximum DP is calculated to cover >99.99% of distribution mass:
    - For living chains: Poisson(nup) has most mass within 5*sqrt(nup)
    - For dead chains: can extend to ~2*nup in worst case
    - Conservative: 3*nup + 10*sqrt(nup)

    The calculated max is bounded by the MW range of the data and
    guaranteed to be at least 2*nu.

    Examples
    --------
    >>> dps = calculate_dp_range(mws, intensities, monomer_mw=104, nu=80, nup=100)
    >>> print(f"DP range: 1 to {len(dps)}")
    """
    # For living chains: Poisson(nup) has most mass within 5*sqrt(nup)
    # For dead chains: can extend to ~2*nup in worst case (high termination)
    # Conservative: cover >99.99% of distribution mass
    intelligent_max_dp = int(3 * nup + 10 * np.sqrt(nup))

    # Don't exceed original maximum, and ensure at least 2x number average
    max_mw_based_dp = int(np.max(molecular_weights) / monomer_mw)
    max_dp = max(
        min(intelligent_max_dp, max_mw_based_dp),
        int(2 * nu)
    )

    return np.arange(1, max_dp, dtype=int)


def living_distribution_integrand(
    time: float,
    dps: np.ndarray,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    combination: bool = False,
    bn: float = 1.0
) -> np.ndarray:
    """
    Determines the living chain distribution at each point in time.
    
    This function computes the instantaneous distribution of living chains
    at a specific reduced time during the polymerization. It is used as an
    integrand for calculating the total distribution of dead chains.

    Parameters
    ----------
    time : float
        Reduced time (t' = kt * t / [I]_0 ^ (1-n)) specified
    dps : ndarray
        Degrees of polymerization at which to evaluate the distribution.
    alpha : float
        Ratio kt/kp of termination to propagation rate constants.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial initiator concentration.
    order : float
        Order of termination reaction.
    combination : bool, optional
        If True, termination occurs by combination of two chains (each
        with DP ~ nup), forming a dead chain with DP ~ 2*nup. Default is
        False (termination without combination).
    bn : float, optional
        Inverse of propagation order in living chain. Default is 1.0.

    Returns
    -------
    ndarray
        Mole fraction of chains with specified DPs that terminate at the
        given time.

    Notes
    -----
    For combination termination (typically order=2), two living chains
    combine to form one dead chain with approximately twice the molecular
    weight.
    """
    b = living_chain_concentration(init, order, time)
    nup = living_chain_dp(alpha, init_mon, init, order, time, bn)

    if combination:
        return ((b ** order) * (init ** (1 - order)) *
                poisson.pmf(dps, 2 * nup) / 2)

    return (b ** order) * (init ** (1 - order)) * poisson.pmf(dps, nup)


def calculate_distribution(
    dps: np.ndarray,
    nu: float,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    combination: bool = False,
    bn: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mole fraction distributions of living and dead chains.

    Computes the complete distribution of chain lengths for both living
    (still propagating) and dead (terminated) chains at a specified
    kinetic chain length.

    Parameters
    ----------
    dps : ndarray
        Degrees of polymerization at which to calculate mole fractions.
    nu : float
        Kinetic chain length, defined as ([M]₀ - [M]) / [I]₀. Represents
        the number average degree of polymerization.
    alpha : float
        Ratio kt/kp of rate constants.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial initiator concentration.
    order : float
        Order of termination reaction.
    combination : bool, optional
        Whether termination occurs by chain combination. Default is False.
    bn : float, optional
        Inverse of propagation order. Default is 1.0.

    Returns
    -------
    alive_fracs : ndarray
        Mole fraction of living chains at each DP.
    dead_fracs : ndarray
        Mole fraction of dead chains at each DP.

    Notes
    -----
    The sum of alive_fracs and dead_fracs integrates to 1 (within
    numerical tolerance), representing the complete distribution of
    chain lengths in the system.

    At high conversion (approaching 100%), the calculation uses a finite
    time corresponding to 99.99% chain termination to avoid numerical
    issues with infinite time integrals.
    """
    conv = init * nu / init_mon

    if np.isclose(conv, 1):
        # Use finite time for very high conversion to avoid numerical issues
        CHAIN_DEATH_FRACTION = 0.9999
        red_time = time_to_chain_death(CHAIN_DEATH_FRACTION, order, init)
    else:
        red_time = conversion_to_time(alpha, init, order, conv, bn)

    b = living_chain_concentration(init, order, red_time)
    nup = living_chain_dp(alpha, init_mon, init, order, red_time, bn)

    # Living chains follow a Poisson distribution
    alive_fracs = b * poisson.pmf(dps, nup)

    # Dead chains: integrate over all termination times
    args = (dps, alpha, init_mon, init, order, combination, bn)
    dead_fracs, _ = quad_vec(living_distribution_integrand, 0, red_time,
                             args=args)

    return np.array(alive_fracs), np.array(dead_fracs)


def calculate_mwd(
    molecular_weights: np.ndarray,
    monomer_mw: float,
    nu: float,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    sigma: float,
    tau: float = 0,
    combination: bool = False,
    bn: float = 1.0,
    live_only: bool = False
) -> np.ndarray:
    """
    Calculate complete molecular weight distribution.

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
    nu : float
        Kinetic chain length ([M]₀ - [M]) / [I]₀.
    alpha : float
        Ratio kt/kp of rate constants.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial initiator concentration.
    order : float
        Order of termination reaction.
    sigma : float
        SEC line broadening parameter (std dev in log MW space).
    sigma : float
        SEC line broadening parameter for lower molecular weight tail
        (used for exponentially modified Gaussian broadening)
    combination : bool, optional
        Whether termination occurs by combination. Default is False.
    bn : float, optional
        Inverse of propagation order. Default is 1.0.
    live_only : bool, optional
        If True, return only the living chain distribution (useful for
        examining the non-terminated portion). Default is False.

    Returns
    -------
    ndarray
        Weight fraction distribution at the specified molecular weights.
        Normalized such that integration over MW gives 1.

    Warns
    -----
    UserWarning
        If combination=True but order≠2 (chemically unusual).
        If maximum DP > 5000 (may cause slow computation).

    Notes
    -----
    The calculation proceeds in several steps:
    1. Calculate mole fractions at discrete DPs (living and dead)
    2. Apply Gaussian broadening to account for SEC instrumental effects
    3. Convert from number to weight distribution (multiply by DP)
    4. Normalize to unit area

    Examples
    --------
    >>> mws = np.logspace(3, 6, 500)  # 1k to 1M Da
    >>> distribution = calculate_mwd(
    ...     mws, monomer_mw=104, nu=80, alpha=0.01, init_mon=1.0,
    ...     init=0.01, order=1.5, sigma=0.11
    ... )
    """
    # Validate inputs
    if combination and order != 2:
        warnings.warn(
            'Combination termination with order≠2 is chemically unusual',
            UserWarning
        )

    max_dp = int(np.max(molecular_weights) / monomer_mw)

    if max_dp > 5000:
        warnings.warn(
            f'Large maximum DP ({max_dp}) may cause slow computation',
            UserWarning
        )

    # Calculate discrete DP distribution
    dps = np.arange(1, max_dp, dtype=int)
    pred_dist = calculate_distribution(
        dps, nu, alpha, init_mon, init, order, combination, bn
    )

    # Create meshgrid for broadening calculation
    dps_mesh, mws_mesh = np.meshgrid(dps, molecular_weights)
    broadenings = egh_broadening(mws_mesh, dps_mesh * monomer_mw, sigma, tau)
    
    # Convert to weight distribution
    tot_dist = (pred_dist[0] + pred_dist[1]) * dps
    raw_mwd = np.matmul(broadenings, tot_dist)

    if live_only:
        # Calculate and normalize only the living chain distribution
        live_mwd = np.matmul(broadenings, pred_dist[0] * dps)
        # Normalize by the TOTAL distribution, not just living chains
        live_mwd = live_mwd / np.trapezoid(raw_mwd, molecular_weights)
        return live_mwd

    # Normalize
    mwd = raw_mwd / np.trapezoid(raw_mwd, molecular_weights)

    return mwd
