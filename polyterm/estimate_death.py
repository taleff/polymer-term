"""
Tool for estimating the fraction of dead chains in a polymer

Fits an exponential Gaussian hybrid to a molecular weight distribution
to estimate the quantity of dead chains
"""

import numpy as np
from scipy.stats import poisson

from .core.broadening import (
    egh_broadening,
)

from .mwd import MWDResult


def estimate_death(molecular_weights, intensities, sigma, tau, monomer_mw=None):
    """
    Estimate living chain fraction by fitting the right edge of an MWD.

    Fits the right edge (high MW side) of an experimental molecular weight
    distribution to estimate the living chain contribution. Two fitting
    modes are available:

    1. With monomer_mw: Uses a Poisson distribution convolved with EGH
       broadening to account for the intrinsic width of living chain
       distributions. More accurate at low DP where Poisson width is
       comparable to instrumental broadening.

    2. Without monomer_mw: Fits a simple EGH peak function. Faster and
       suitable when Poisson broadening is negligible (high DP) or when
       monomer MW is unknown.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights from SEC/GPC measurement. Should be in increasing
        order (low to high MW).
    intensities : ndarray
        Detector response at each molecular weight (weight fractions).
    sigma : float
        Gaussian broadening parameter (standard deviation in log MW space).
        Should be obtained from calibration with narrow standards.
    tau : float
        Exponential tailing parameter for EGH broadening. Set to 0 for
        symmetric Gaussian broadening.
    monomer_mw : float, optional
        Molecular weight of one monomer unit. If provided, uses Poisson-
        broadened fitting. If None, fits a simple EGH peak.

    Returns
    -------
    MWDResult
        Dataclass containing the distribution and kinetic parameters

    Raises
    ------
    ValueError
        If sigma is not positive.
        If monomer_mw is provided but not positive.
        If molecular_weights and intensities have different lengths.

    Notes
    -----
    The procedure:
    1. Finds the peak of the experimental distribution
    2. Constructs a living chain distribution centered at the peak MW,
       scaled to match the peak height, using either:
       - Poisson + EGH (if monomer_mw provided): accounts for intrinsic
         chain length distribution width
       - Simple EGH (if monomer_mw is None): single broadened peak
    3. Subtracts living from experimental to get dead distribution

    Examples
    --------
    Fit with Poisson broadening (more accurate):

    >>> result = estimate_death(
    ...     mws, ints,
    ...     sigma=0.128,
    ...     tau=0.0456,
    ...     monomer_mw=100.0,
    ... )

    Fit with simple EGH (no monomer_mw needed):

    >>> result = estimate_death(
    ...     mws, ints,
    ...     sigma=0.128,
    ...     tau=0.0456,
    ... )
    """
    # Validate inputs
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if len(molecular_weights) != len(intensities):
        raise ValueError(
            f"Length mismatch: molecular_weights ({len(molecular_weights)}) "
            f"!= intensities ({len(intensities)})"
        )
    if (monomer_mw is not None) and (monomer_mw <= 0):
        raise ValueError("monomer_mw must be positive")

    # Normalize the peak of intensities to one
    intensities = intensities / np.max(intensities)

    # Ensure arrays are sorted by increasing MW
    sort_idx = np.argsort(molecular_weights)
    mws, ints = molecular_weights[sort_idx], intensities[sort_idx]

    # Find peak position and height
    peak_idx = np.argmax(ints)
    peak_mw = mws[peak_idx]
    peak_int = ints[peak_idx]

    if monomer_mw is not None:
        live_chain_intensities, dead_chain_fraction = _match_poisson_egh(
            mws, ints, peak_mw, peak_int, monomer_mw, sigma, tau
        )
    else:
        live_chain_intensities, dead_chain_fraction = _match_simple_egh(
            mws, ints, peak_mw, peak_int, sigma, tau
        )

    dead_chain_intensities = intensities - live_chain_intensities

    return MWDResult(
        molecular_weights, intensities, dead_chain_intensities,
        live_chain_intensities, dead_chain_fraction
    )


def _dead_fraction_from_intensities(mws, experimental, living):
    """Estimate dead chain fraction from weight-fraction intensities."""
    mole_frac_exp = experimental / mws
    mole_frac_live = living / mws
    return 1 - (np.sum(mole_frac_live) / np.sum(mole_frac_exp))


def _match_poisson_egh(mws, ints, peak_mw, peak_int,
                       monomer_mw, sigma, tau):
    """
    Construct living chain distribution using Poisson + EGH broadening.

    Centers the Poisson distribution at the experimental peak DP and
    scales to match the experimental peak height. No fitting required.
    """
    nup = peak_mw / monomer_mw
    max_dp = int(np.max(mws) / monomer_mw)
    dps = np.arange(1, max_dp, dtype=int)
    mass_fracs = poisson.pmf(dps, nup) * dps

    dps_mesh, mws_mesh = np.meshgrid(dps, mws)
    broadenings = egh_broadening(mws_mesh, dps_mesh * monomer_mw, sigma, tau)
    shape = broadenings @ mass_fracs

    # Scale so peak matches experimental peak height
    coeff = peak_int / np.max(shape) if np.max(shape) > 0 else 0.0
    live_ints = coeff * shape

    return live_ints, _dead_fraction_from_intensities(mws, ints, live_ints)


def _match_simple_egh(mws, ints, peak_mw, peak_int, sigma, tau):
    """
    Construct living chain distribution using a simple EGH peak.

    Centers at the experimental peak MW and scales to match the
    experimental peak height. No fitting required.
    """
    shape = egh_broadening(mws, peak_mw, sigma, tau)

    # Scale so peak matches experimental peak height
    coeff = peak_int / np.max(shape) if np.max(shape) > 0 else 0.0
    live_ints = coeff * shape

    return live_ints, _dead_fraction_from_intensities(mws, ints, live_ints)

