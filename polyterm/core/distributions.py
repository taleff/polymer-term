"""
Molecular weight distribution calculations for living polymerizations.

This module provides functions for calculating theoretical molecular weight
distributions for controlled polymerizations with termination.
"""

import numpy as np
from scipy.stats import poisson, nbinom
from typing import Tuple

__all__ = [
    'calculate_dp_range',
    'get_poisson_dp_range',
    'compute_poisson_mass_fracs',
]


def get_poisson_dp_range(nup, dps, n_sigma = 6.0):
    """
    Get the DP indices where Poisson(nup) has significant probability mass.

    For Poisson(λ), the standard deviation is √λ. This function returns
    a range [min_dp, max_dp] that covers essentially all the probability
    mass (>99.9999% for n_sigma=6).

    Parameters
    ----------
    nup : float
        Poisson parameter (mean/expected DP).
    dps : np.array
        Array with examined degrees of polymerizations, should be
        monotonically increasing.
    n_sigma : float, optional
        Number of standard deviations to include. Default 6.0 covers
        >99.9999% of the distribution.

    Returns
    -------
    idx: int
        The index of the end point for the dp range
        
    """
    std = np.sqrt(max(nup, 1))
    max_dp = min(dps[-1], int(nup + n_sigma * std) + 1)

    idx = min(len(dps), max_dp - dps[0] + 1)

    return idx


def compute_poisson_mass_fracs(
    dps: np.ndarray,
    nup: float,
    n_sigma: float = 6.0
) -> np.ndarray:
    """
    Compute Poisson mass fractions efficiently over a DP array.

    Only computes PMF values in the range where they're significant,
    setting other values to zero. This is much faster than computing
    over the full DP range when nup is much smaller than max(dps).

    Parameters
    ----------
    dps : ndarray
        Array of degrees of polymerization.
    nup : float
        Poisson parameter (mean/expected DP).
    n_sigma : float, optional
        Number of standard deviations to include. Default 6.0.

    Returns
    -------
    mass_fracs : ndarray
        Poisson mass fractions (pmf * dp) at each DP in dps.
        Same shape as dps.

    Notes
    -----
    For a Poisson distribution, mass fractions are pmf(dp) * dp, which
    represents the weight contribution at each DP.

    Examples
    --------
    >>> dps = np.arange(1, 1000)
    >>> mass_fracs = compute_poisson_mass_fracs(dps, nup=100)
    >>> print(f"Non-zero range: {np.where(mass_fracs > 0)[0][[0, -1]]}")
    """
    result = np.zeros(len(dps), dtype=float)

    # Get the range where Poisson has significant values
    max_dp = get_poisson_dp_range(nup, dps, n_sigma)

    # Find indices in dps array corresponding to this range
    # dps is assumed to be contiguous starting from some value
    dp_start = int(dps[0])
    idx_end = min(len(dps), max_dp - dp_start + 1)

    if dp_start < idx_end:
        # Compute only for the relevant range
        relevant_dps = dps[dp_start:idx_end]
        result[dp_start:idx_end] = poisson.pmf(relevant_dps, nup) * relevant_dps

    return result


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

