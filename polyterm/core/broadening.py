"""
SEC/GPC instrumental broadening functions.

This module provides functions for modeling instrumental broadening effects
in Size Exclusion Chromatography (SEC) / Gel Permeation Chromatography (GPC).
"""

import numpy as np
from scipy.special import erfc
from typing import Union, Optional

__all__ = [
    'gaussian_broadening',
    'emg_broadening',
    'egh_broadening',
    'compute_broadening_matrix',
]


def gaussian_broadening(
    molecular_weights: np.ndarray,
    center: Union[float, np.ndarray],
    sigma: float
) -> np.ndarray:
    """
    Calculate Gaussian line broadening function for SEC/GPC.

    Models the instrumental broadening effect in Size Exclusion Chromatography.
    The broadening is Gaussian in log(molecular weight) space.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which to evaluate the broadening function.
    center : float or ndarray
        Center(s) of the Gaussian distribution(s), representing true
        molecular weights.
    sigma : float
        Standard deviation in log(molecular weight) space. This parameter
        characterizes the instrumental broadening of the SEC/GPC system.

    Returns
    -------
    ndarray
        Normalized Gaussian broadening function values.

    Notes
    -----
    The function is normalized such that integration over all molecular
    weights equals 1. The Gaussian is applied in log space: the center
    parameter is the true average MW, but sigma describes broadening in
    log(MW). The returned Gaussian is also applicable to SEC measurements
    where the intensity of larger molecular weights are overrepresented
    """
    expon = -(np.log(molecular_weights) - np.log(center)) ** 2 / (2 * sigma ** 2)
    coeff = np.sqrt(2 * np.pi) * sigma
    return (1 / coeff) * np.exp(expon)


def emg_broadening(
    molecular_weights: np.ndarray,
    center: Union[float, np.ndarray],
    sigma: float,
    tau: float = 0.0
) -> np.ndarray:
    """
    Calculate Exponentially Modified Gaussian (EMG) broadening for SEC/GPC.

    Models asymmetric instrumental broadening common in Size Exclusion
    Chromatography. The EMG is a convolution of a Gaussian with an
    exponential decay, producing asymmetric tailing toward lower MW.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which to evaluate the broadening function.
    center : float or ndarray
        Center(s) of the distribution, representing true molecular weights.
    sigma : float
        Gaussian standard deviation in log(molecular weight) space.
    tau : float, optional
        Exponential decay parameter in log(molecular weight) space.
        Controls the asymmetric tailing toward lower MW. Default is 0.0,
        which reduces to symmetric Gaussian broadening.

    Returns
    -------
    ndarray
        EMG broadening function values. For scalar center, normalized to
        integrate to 1 over log(MW) space. For array center (meshgrid),
        returns unnormalized probability densities.

    Notes
    -----
    When tau=0, this function returns the same result as gaussian_broadening.

    For SEC instruments, tau > 0 models the common tailing observed toward
    lower molecular weights due to extra-column dispersion and other
    instrumental effects.

    The EMG is the convolution of a Gaussian with an exponential decay.
    For left-tailing (toward lower MW), the analytical formula is:

        EMG(x) = (1/2tau) * exp(sigma^2/2tau^2 - (mu-x)/tau)
                 * erfc((sigma/tau - (mu-x)/sigma) / sqrt(2))

    where x = log(MW), mu = log(center).

    Examples
    --------
    >>> mws = np.logspace(3, 5, 500)
    >>> emg = emg_broadening(mws, center=10000.0, sigma=0.10, tau=0.05)
    """
    # If tau is effectively zero or too small relative to sigma, use Gaussian
    # When sigma²/(2τ²) > ~700, exp() overflows, so use tau < sigma/37 as cutoff
    if tau <= 1e-10 or tau < sigma / 37:
        return gaussian_broadening(molecular_weights, center, sigma)

    # Work in log space
    log_mws = np.log(molecular_weights)
    log_center = np.log(center)

    # Left-tailing EMG formula (tailing toward lower MW)
    # delta = mu - x = log(center) - log(MW)
    # Positive delta means MW < center (we're on the tail side)
    delta = log_center - log_mws

    # Analytical EMG formula for left-tailing
    # z = sigma/tau - delta/sigma
    z = sigma / tau - delta / sigma

    # Compute EMG with numerical stability
    with np.errstate(over='ignore', invalid='ignore'):
        term1 = (sigma ** 2) / (2 * tau ** 2)
        term2 = -delta / tau
        result = (1 / (2 * tau)) * np.exp(term1 + term2) * erfc(z / np.sqrt(2))

    # Handle any numerical issues
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure non-negative
    result = np.maximum(result, 0.0)

    # For scalar center, normalize to integrate to 1 over log(MW) space
    is_scalar_center = np.isscalar(center) or (
        isinstance(center, np.ndarray) and center.ndim == 0
    )

    if is_scalar_center:
        integral = np.trapezoid(result, log_mws)
        if integral > 0:
            result = result / integral

    return result


def egh_broadening(
    molecular_weights: np.ndarray,
    center: Union[float, np.ndarray],
    sigma: float,
    tau: float = 0.0
) -> np.ndarray:
    """
    Calculate Exponential-Gaussian Hybrid (EGH) broadening for SEC/GPC.

    Models asymmetric instrumental broadening common in Size Exclusion
    Chromatography. The EGH is a hybrid of exponential and Gaussian
    functions, providing a numerically stable alternative to the EMG
    that is particularly effective for highly asymmetric peaks.

    Based on Lan & Jorgenson, J. Chromatogr. A 915 (2001) 1-13.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which to evaluate the broadening function.
    center : float or ndarray
        Center(s) of the distribution, representing true molecular weights.
    sigma : float
        Gaussian standard deviation in log(molecular weight) space.
    tau : float, optional
        Exponential decay parameter in log(molecular weight) space.
        Controls the asymmetric tailing toward lower MW. Must be >= 0.
        Default is 0.0, which reduces to symmetric Gaussian broadening.

    Returns
    -------
    ndarray
        EGH broadening function values. For scalar center, normalized to
        integrate to 1 over log(MW) space. For array center (meshgrid),
        returns unnormalized probability densities.

    Notes
    -----
    The EGH function is defined as (equation 12 in Lan & Jorgenson):

        f(x) = H * exp(-delta^2 / (2*sigma^2 + tau*delta))

    where delta = log(center) - log(MW). The function is zero when the
    denominator (2*sigma^2 + tau*delta) <= 0.

    When tau=0, this function returns the same result as gaussian_broadening.

    Examples
    --------
    >>> mws = np.logspace(3, 5, 500)
    >>> egh = egh_broadening(mws, center=10000.0, sigma=0.10, tau=0.05)
    """
    # If tau is effectively zero, use Gaussian
    if tau <= 1e-10:
        return gaussian_broadening(molecular_weights, center, sigma)

    # Work in log space
    log_mws = np.log(molecular_weights)
    log_center = np.log(center)

    # delta = log(center) - log(MW)
    # Positive delta means MW < center (tail side for tau > 0)
    delta = log_center - log_mws

    # Denominator: 2*sigma^2 + tau*delta
    denominator = 2 * sigma**2 + tau * delta

    # Function is zero when denominator <= 0
    # Initialize result array and only compute where valid
    result = np.zeros_like(molecular_weights, dtype=float)
    valid = denominator > 0

    # Compute EGH only where denominator is positive
    result[valid] = np.exp(-delta[valid]**2 / denominator[valid])

    # For scalar center, normalize to integrate to 1 over log(MW) space
    is_scalar_center = np.isscalar(center) or (
        isinstance(center, np.ndarray) and center.ndim == 0
    )

    if is_scalar_center:
        integral = np.trapezoid(result, log_mws)
        if integral > 0:
            result = result / integral

    return result


def compute_broadening_matrix(
    molecular_weights: np.ndarray,
    dps: np.ndarray,
    monomer_mw: float,
    sigma: float,
    tau: Optional[float] = None
) -> np.ndarray:
    """
    Compute broadening matrix for DP-to-MW transformation.

    Creates a matrix that transforms discrete degree of polymerization
    fractions into a continuous molecular weight distribution, accounting
    for SEC/GPC instrumental broadening.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which to evaluate the broadened distribution.
    dps : ndarray
        Degrees of polymerization for the discrete chain distribution.
    monomer_mw : float
        Molecular weight of one monomer unit.
    sigma : float
        SEC broadening parameter (standard deviation in log MW space).
    tau : float, optional
        Exponential tailing parameter for EGH broadening. If None or <= 0,
        symmetric Gaussian broadening is used.

    Returns
    -------
    ndarray
        Broadening matrix of shape (len(molecular_weights), len(dps)).
        When multiplied by DP fractions, produces the broadened MWD.

    Notes
    -----
    The matrix is computed by evaluating the broadening function at each
    (MW, DP) pair. For Gaussian broadening (tau=None or tau<=0), the
    function is symmetric in log space. For EGH broadening (tau>0), the
    function has asymmetric tailing toward lower molecular weights.

    Examples
    --------
    >>> mws = np.logspace(3, 5, 200)
    >>> dps = np.arange(1, 500)
    >>> matrix = compute_broadening_matrix(mws, dps, monomer_mw=104, sigma=0.1)
    >>> # Apply to DP distribution: mwd = matrix @ (fracs * dps)
    """
    dps_mesh, mws_mesh = np.meshgrid(dps, molecular_weights)
    true_mws = dps_mesh * monomer_mw

    if tau is None or tau <= 0:
        return gaussian_broadening(mws_mesh, true_mws, sigma)
    return egh_broadening(mws_mesh, true_mws, sigma, tau)

