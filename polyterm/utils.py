"""
Utility functions for molecular weight distribution analysis.

This module provides helper functions for analyzing experimental SEC/GPC data,
including peak fitting, average calculations, and goodness-of-fit metrics.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Tuple

from .core.distributions import gaussian_broadening

__all__ = [
    'calculate_number_average_dp',
    'fit_right_edge',
    'calculate_r_squared',
]


def calculate_number_average_dp(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    monomer_mw: float
) -> float:
    """
    Calculate number average degree of polymerization from SEC trace.

    Converts a weight-based SEC distribution to number-average DP using
    the relationship: Mn = (∫ w(M) dM) / (∫ w(M)/M dM).

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which the distribution is measured.
        Units must be consistent with monomer_mw.
    intensities : ndarray
        Intensity values (weight fraction) at each molecular weight.
        Should be normalized or will be treated as relative weights.
    monomer_mw : float
        Molecular weight of one monomer unit. Same units as
        molecular_weights.

    Returns
    -------
    float
        Number average degree of polymerization (dimensionless).

    Notes
    -----
    This function assumes the input intensities represent a weight
    distribution. The SEC detector response is assumed to be
    proportional to mass (e.g., RI detector, normalized response).

    Examples
    --------
    >>> mws = np.array([1000, 2000, 3000, 4000, 5000])
    >>> ints = np.array([0.1, 0.3, 0.4, 0.15, 0.05])
    >>> nu = calculate_number_average_dp(mws, ints, monomer_mw=100)
    """
    # Convert weight distribution to number distribution
    number_intensities = intensities / molecular_weights

    # Calculate number average MW
    mn = (np.trapezoid(intensities, molecular_weights) /
          np.trapezoid(number_intensities, molecular_weights))

    # Convert to DP
    return mn / monomer_mw


# Alias for backward compatibility
nu_finder = calculate_number_average_dp


def fit_right_edge(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    monomer_mw: float
) -> Tuple[float, float]:
    """
    Fit Gaussian to the right edge of SEC trace to estimate broadening.

    The right (high MW) edge of an SEC peak from a narrow distribution
    can be fit with a Gaussian to estimate the instrumental line
    broadening. This is particularly useful for well-controlled living
    polymerizations where the living chains form a sharp peak.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which the distribution is measured.
    intensities : ndarray
        Intensity values at each molecular weight.
    monomer_mw : float
        Molecular weight of one monomer unit. Same units as
        molecular_weights.

    Returns
    -------
    nup : float
        Estimated peak degree of polymerization (DP of living chains).
    sigma : float
        Estimated line broadening parameter (standard deviation in
        log MW space).

    Notes
    -----
    This method works best when:
    - The polymerization is well-controlled (narrow living chain distribution)
    - The right edge is not obscured by high MW dead chains
    - The peak maximum is well-defined

    The fitting is performed on data from the peak maximum to higher
    molecular weights, as this region most closely approximates the
    broadened living chain distribution.

    The fit minimizes the residual between the observed intensities and
    a normalized Gaussian in log-space.

    Examples
    --------
    >>> # For a narrow living polymer peak
    >>> nup, sigma = fit_right_edge(mws, intensities, monomer_mw=104)
    >>> print(f"Living chain DP: {nup:.1f}, Broadening: {sigma:.3f}")
    """
    # Extract right edge (from peak maximum onwards)
    peak_idx = np.argmax(intensities)
    edge_mws = molecular_weights[peak_idx:]
    edge_ints = intensities[peak_idx:]
    peak_intensity = np.max(intensities)

    def residual(params):
        """Calculate residual between data and Gaussian fit."""
        center, width = params
        # Coefficient to match peak height
        coeff = (peak_intensity * width * np.sqrt(2 * np.pi) /
                np.exp(-0.5 * width ** 2))
        predicted = coeff * gaussian_broadening(edge_mws, center, width)
        return 1e9 * (predicted - edge_ints)

    # Initial guess: center at first point, small broadening
    initial_guess = (edge_mws[0], 0.01)
    bounds = (0, np.inf)

    result = least_squares(residual, x0=initial_guess, bounds=bounds)

    # Convert center MW to DP
    center_mw, sigma = result['x']
    nup = center_mw / monomer_mw

    return nup, sigma


# Alias for backward compatibility
sec_right_edge_fit = fit_right_edge


def calculate_r_squared(
    observed: np.ndarray,
    predicted: np.ndarray
) -> float:
    """
    Calculate coefficient of determination (R²) for model fit.

    R² measures the proportion of variance in the observed data that is
    explained by the model predictions. Values range from -∞ to 1, where
    1 indicates perfect prediction.

    Parameters
    ----------
    observed : ndarray
        Observed (experimental) intensity values.
    predicted : ndarray
        Predicted (model) intensity values at the same points.

    Returns
    -------
    float
        Coefficient of determination. R² = 1 - (SS_res / SS_tot), where
        SS_res is the residual sum of squares and SS_tot is the total
        sum of squares.

    Notes
    -----
    R² interpretation:
    - R² = 1: Perfect fit
    - R² = 0: Model performs no better than predicting the mean
    - R² < 0: Model performs worse than predicting the mean

    For non-linear models (like the kinetic fits in this package), R²
    can sometimes be negative if the model is very poor.

    This implementation uses the standard definition:
        R² = 1 - Σ(observed - predicted)² / Σ(observed - mean(observed))²

    Examples
    --------
    >>> obs = np.array([1.0, 2.0, 3.0, 4.0])
    >>> pred = np.array([1.1, 1.9, 3.1, 3.9])
    >>> r2 = calculate_r_squared(obs, pred)
    >>> print(f"R² = {r2:.4f}")
    """
    mean_observed = np.mean(observed)
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - mean_observed) ** 2)

    # Avoid division by zero
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


# Alias for backward compatibility
corr_calc = calculate_r_squared
