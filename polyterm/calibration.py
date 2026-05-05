"""
SEC/GPC calibration utilities for instrumental broadening.

This module provides functions for calibrating SEC instrumental broadening
parameters from narrow molecular weight standards.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from .core.broadening import emg_broadening, egh_broadening
from .core.distributions import get_poisson_dp_range
from .utils import calculate_r_squared

__all__ = [
    'calibrate_emg_broadening',
    'calibrate_egh_broadening',
    'CalibrationResult',
    'compute_poisson_broadened_mwd',
]


@dataclass(frozen=True)
class CalibrationResult:
    """
    Container for broadening calibration results.

    This immutable class stores the fitted broadening parameters from
    calibrating with a narrow molecular weight standard. Works with both
    EMG (Exponentially Modified Gaussian) and EGH (Exponential-Gaussian
    Hybrid) broadening models.

    Attributes
    ----------
    sigma : float
        Gaussian standard deviation in log(MW) space.
    tau : float
        Exponential decay parameter in log(MW) space. Controls asymmetric
        tailing toward lower MW. Value of 0 indicates symmetric Gaussian.
    center : float
        Fitted peak center molecular weight.
    r_squared : float
        Coefficient of determination for the calibration fit.

    Examples
    --------
    >>> result = CalibrationResult(sigma=0.10, tau=0.05, center=10000.0, r_squared=0.998)
    >>> print(f"Broadening: sigma={result.sigma:.3f}, tau={result.tau:.3f}")
    """

    sigma: float
    tau: float
    center: float
    r_squared: float

    def __repr__(self) -> str:
        """Return string representation of calibration results."""
        return (
            f"CalibrationResult(\n"
            f"  sigma = {self.sigma:.4f}\n"
            f"  tau = {self.tau:.4f}\n"
            f"  center = {self.center:.1f}\n"
            f"  R² = {self.r_squared:.4f}\n"
            f")"
        )


def compute_poisson_broadened_mwd(molecular_weights, center_dp,
                                  monomer_mw, sigma, tau = 0.0,
                                  broadening_model = 'egh'):
    """
    Compute a Poisson-broadened molecular weight distribution.

    This function generates the theoretical MWD for a living polymer
    with Poisson chain length distribution, broadened by instrumental
    effects.
    """
    # Select broadening function
    if broadening_model.lower() == 'emg':
        broadening_func = emg_broadening
    elif broadening_model.lower() == 'egh':
        broadening_func = egh_broadening
    else:
        raise ValueError(f"Unknown broadening model: {broadening_model}")

    # Calculate DP range needed (limit to 3x center or data range)
    max_dp_from_data = int(np.max(molecular_weights) / monomer_mw) + 1
    max_dp_from_center = int(center_dp * 3) + 1
    max_dp = min(max_dp_from_data, max_dp_from_center)

    # Create DP array and compute broadening matrix
    dps = np.arange(1, max_dp, dtype=int)
    dps_mesh, mws_mesh = np.meshgrid(dps, molecular_weights)
    dp_mws = dps_mesh * monomer_mw

    broadening_matrix = broadening_func(mws_mesh, dp_mws, sigma, tau)

    # Compute Poisson mass fractions only in relevant range
    idx_end = get_poisson_dp_range(center_dp, dps, n_sigma=6.0)
    # Compute start index based on n_sigma below the mean
    std = np.sqrt(max(center_dp, 1))
    min_dp = max(1, int(center_dp - 6.0 * std))
    idx_start = max(0, min_dp - 1)  # dps starts at 1

    mass_fracs = np.zeros(len(dps), dtype=float)
    if idx_start < idx_end:
        relevant_dps = dps[idx_start:idx_end]
        mass_fracs[idx_start:idx_end] = (
            poisson.pmf(relevant_dps, center_dp) * relevant_dps
        )

    # Apply broadening
    distribution = broadening_matrix @ mass_fracs

    return distribution


def _calibrate_broadening(molecular_weights, intensities,
                          broadening_func, max_sigma, max_tau,
                          monomer_mw):
    """
    Internal helper for calibrating SEC instrumental broadening.
    """
    # Normalize intensities
    log_mws = np.log(molecular_weights)
    norm = np.trapezoid(intensities, log_mws)
    intensities_norm = intensities / norm if norm > 0 else intensities

    # Estimate initial parameters from peak location
    peak_idx = np.argmax(intensities_norm)
    center_init = molecular_weights[peak_idx]

    # Estimate sigma from peak width (FWHM)
    half_max = intensities_norm[peak_idx] / 2
    above_half = intensities_norm > half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm_log = log_mws[indices[-1]] - log_mws[indices[0]]
        sigma_init = fwhm_log / np.sqrt(8 * np.log(2))
    else:
        sigma_init = 0.1

    # Estimate tau from asymmetry (compare areas left and right of peak)
    left_area = np.trapezoid(intensities_norm[:peak_idx], log_mws[:peak_idx])
    right_area = np.trapezoid(intensities_norm[peak_idx:], log_mws[peak_idx:])
    asymmetry = left_area / (right_area + 1e-10)
    tau_init = max(0.01, min(0.1, (asymmetry - 1) * sigma_init))

    # Use Poisson-broadened fitting if monomer_mw is provided
    if monomer_mw is not None:
        return _calibrate_with_poisson(
            molecular_weights, intensities_norm, log_mws, broadening_func,
            max_sigma, max_tau, monomer_mw, center_init, sigma_init, tau_init
        )

    # Standard fitting (delta function standard)
    return _calibrate_delta_standard(
        molecular_weights, intensities_norm, log_mws, broadening_func,
        max_sigma, max_tau, center_init, sigma_init, tau_init
    )


def _calibrate_delta_standard(molecular_weights, intensities_norm,
                              log_mws, broadening_func, max_sigma,
                              max_tau, center_init, sigma_init,
                              tau_init):
    """
    Calibrate assuming the standard is a delta function (narrow standard).

    This is the traditional calibration approach that fits a single broadening
    function directly to the observed peak.
    """
    # Set parameter bounds
    bounds = [
        (0.01, max_sigma),
        (0.0, max_tau),
        (molecular_weights.min(), molecular_weights.max())
    ]

    def objective(params: Tuple[float, float, float]) -> float:
        """Compute sum of squared residuals for given parameters."""
        sigma, tau, center = params
        try:
            pred = broadening_func(molecular_weights, center, sigma, tau)
            pred_norm = np.trapezoid(pred, log_mws)
            if pred_norm > 0:
                pred = pred / pred_norm
            residuals = pred - intensities_norm
            return float(np.sum(residuals ** 2))
        except (ValueError, RuntimeWarning, FloatingPointError):
            return 1e10

    # Try multiple starting points to avoid local minima
    best_result = None
    best_cost = float('inf')

    tau_starts = [0.0, tau_init, 0.03, 0.05, 0.08]
    sigma_starts = [sigma_init * 0.8, sigma_init, sigma_init * 1.2]

    for tau_start in tau_starts:
        for sigma_start in sigma_starts:
            init_guess = [
                np.clip(sigma_start, bounds[0][0], bounds[0][1]),
                np.clip(tau_start, bounds[1][0], bounds[1][1]),
                center_init
            ]

            try:
                result = minimize(
                    objective,
                    init_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-9}
                )
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result
            except Exception:
                continue

    # Fallback if no optimization succeeded
    if best_result is None:
        init_guess = [sigma_init, 0.02, center_init]
        best_result = minimize(
            objective,
            init_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

    sigma, tau, center = best_result.x

    # Calculate R-squared for the fit
    pred = broadening_func(molecular_weights, center, sigma, tau)
    pred_norm = np.trapezoid(pred, log_mws)
    if pred_norm > 0:
        pred = pred / pred_norm
    r_squared = calculate_r_squared(intensities_norm, pred)

    return CalibrationResult(
        sigma=sigma,
        tau=tau,
        center=center,
        r_squared=r_squared
    )


def _fit_gaussian_fast(log_mws, intensities_norm):
    """
    Fast Gaussian fit using weighted moments.

    Returns center (in log space) and sigma directly from the data,
    avoiding iterative optimization.
    """
    # Weighted mean (center in log space)
    center_log = np.sum(log_mws * intensities_norm) / np.sum(intensities_norm)

    # Weighted variance
    variance = np.sum(intensities_norm * (log_mws - center_log)**2) / np.sum(intensities_norm)
    sigma = np.sqrt(variance)

    return center_log, sigma


def _calibrate_with_poisson(molecular_weights, intensities_norm,
                            log_mws, broadening_func, max_sigma,
                            max_tau, monomer_mw, _center_init,
                            _sigma_init, _tau_init):
    """
    Calibrate accounting for natural Poisson dispersity of living chains.

    For living polymerization standards, the chain length distribution
    follows a Poisson distribution. This function fits broadening parameters
    while accounting for this intrinsic width, which is important at low DP
    where Poisson width can be comparable to instrumental broadening.

    Uses analytical variance decomposition for fast initialization:
    1. Fit Gaussian to get observed center and variance
    2. Calculate Poisson variance contribution: Var_poisson ≈ 1/center_dp
    3. Estimate instrumental variance: Var_inst = Var_observed - Var_poisson
    4. Use these estimates for single-pass optimization
    """
    # Stage 1: Fast Gaussian fit using weighted moments (no iteration)
    center_log, sigma_observed = _fit_gaussian_fast(log_mws, intensities_norm)
    center_mw = np.exp(center_log)
    center_dp_init = center_mw / monomer_mw

    # Decompose observed variance into Poisson and instrumental components
    # For Poisson(λ), variance in log(MW) space ≈ 1/λ (delta method)
    var_observed = sigma_observed ** 2
    var_poisson = 1.0 / center_dp_init

    # Instrumental variance = observed - Poisson contribution
    # Ensure non-negative (can happen if DP is very low)
    var_instrumental = max(0.01**2, var_observed - var_poisson)
    sigma_init_warm = np.sqrt(var_instrumental)

    # Estimate tau from asymmetry (simple heuristic)
    peak_idx = np.argmax(intensities_norm)
    left_area = np.trapezoid(intensities_norm[:peak_idx], log_mws[:peak_idx])
    right_area = np.trapezoid(intensities_norm[peak_idx:], log_mws[peak_idx:])
    asymmetry = left_area / (right_area + 1e-10)
    tau_init_warm = max(0.0, min(0.1, (asymmetry - 1) * sigma_init_warm))

    # Limit max_dp based on estimated center (3x center covers Poisson tail)
    # This avoids creating unnecessarily large matrices
    max_dp_from_data = int(np.max(molecular_weights) / monomer_mw) + 1
    max_dp_from_center = int(center_dp_init * 3) + 1
    max_dp = min(max_dp_from_data, max_dp_from_center)

    # Pre-compute fixed DP array and meshgrid (avoids repeated allocation)
    dps = np.arange(1, max_dp, dtype=int)
    dps_mesh, mws_mesh = np.meshgrid(dps, molecular_weights)
    dp_mws = dps_mesh * monomer_mw  # Pre-compute DP molecular weights

    def objective(params: Tuple[float, float, float]) -> float:
        """Compute sum of squared residuals using Poisson-broadened model."""
        sigma, tau, center_dp = params
        try:
            # Compute broadening matrix
            broadening_matrix = broadening_func(mws_mesh, dp_mws, sigma, tau)

            # Compute Poisson mass fractions only in relevant range
            idx_end = get_poisson_dp_range(center_dp, dps, n_sigma=6.0)
            # Compute start index based on n_sigma below the mean
            std = np.sqrt(max(center_dp, 1))
            min_dp_val = max(1, int(center_dp - 6.0 * std))
            idx_start = max(0, min_dp_val - 1)  # dps starts at 1

            mass_fracs = np.zeros(len(dps), dtype=float)
            if idx_start < idx_end:
                relevant_dps = dps[idx_start:idx_end]
                mass_fracs[idx_start:idx_end] = (
                    poisson.pmf(relevant_dps, center_dp) * relevant_dps
                )

            pred = broadening_matrix @ mass_fracs
            pred_norm = np.trapezoid(pred, log_mws)
            if pred_norm > 0:
                pred = pred / pred_norm
            residuals = pred - intensities_norm
            return float(np.sum(residuals ** 2))
        except (ValueError, RuntimeWarning, FloatingPointError):
            return 1e10

    # Stage 2: Single Poisson-aware optimization from analytical warm start
    bounds = [
        (0.01, max_sigma),
        (0.0, max_tau),
        (1.0, max_dp)
    ]

    init_guess = [sigma_init_warm, tau_init_warm, center_dp_init]

    result = minimize(
        objective,
        init_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-9}
    )

    sigma, tau, center_dp = result.x

    # Calculate R-squared for the fit
    broadening_matrix = broadening_func(mws_mesh, dp_mws, sigma, tau)

    # Compute Poisson mass fractions only in relevant range
    idx_end = get_poisson_dp_range(center_dp, dps, n_sigma=6.0)
    # Compute start index based on n_sigma below the mean
    std = np.sqrt(max(center_dp, 1))
    min_dp_final = max(1, int(center_dp - 6.0 * std))
    idx_start = max(0, min_dp_final - 1)

    mass_fracs = np.zeros(len(dps), dtype=float)
    if idx_start < idx_end:
        relevant_dps = dps[idx_start:idx_end]
        mass_fracs[idx_start:idx_end] = (
            poisson.pmf(relevant_dps, center_dp) * relevant_dps
        )

    pred = broadening_matrix @ mass_fracs
    pred_norm = np.trapezoid(pred, log_mws)
    if pred_norm > 0:
        pred = pred / pred_norm
    r_squared = calculate_r_squared(intensities_norm, pred)

    # Convert center_dp back to molecular weight for result
    center_mw = center_dp * monomer_mw

    return CalibrationResult(
        sigma=sigma,
        tau=tau,
        center=center_mw,
        r_squared=r_squared
    )


def calibrate_emg_broadening(molecular_weights, intensities,
                             max_sigma = 0.5, max_tau = 0.4,
                             monomer_mw = None):
    """
    Calibrate SEC instrumental broadening from a narrow standard.

    Fits Exponentially Modified Gaussian (EMG) parameters to a narrow
    molecular weight standard to characterize instrumental broadening.
    The fitted parameters (sigma, tau) can then be used with fit_mwd
    for accurate kinetic fitting.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which the distribution is measured.
    intensities : ndarray
        Intensity values (should be a narrow, approximately symmetric
        peak for best results).
    max_sigma : float, optional
        Maximum allowed sigma value. Default is 0.5.
    max_tau : float, optional
        Maximum allowed tau value. Default is 0.2.
    monomer_mw : float, optional
        Molecular weight of one monomer unit. If provided, the calibration
        accounts for the natural Poisson dispersity of living polymerization.
        This is important when calibrating with living polymer standards,
        especially at low DP where the intrinsic Poisson width can be
        comparable to or larger than instrumental broadening.

    Returns
    -------
    CalibrationResult
        Fitted broadening parameters and fit quality metrics.

    Notes
    -----
    The fit optimizes sigma, tau, and center simultaneously to minimize
    the sum of squared residuals between the observed and predicted
    intensity profiles.

    When monomer_mw is None, the standard is assumed to be a delta function
    (infinitely narrow). This is appropriate for narrow polymer standards
    with very high DP where the Poisson width is negligible.

    When monomer_mw is provided, the fitting accounts for the Poisson
    distribution of chain lengths in living polymerization:
    - Chain lengths follow Poisson(nup) where nup is the average DP
    - The observed peak width includes both Poisson and instrumental broadening
    - The fit extracts the instrumental broadening component

    Examples
    --------
    Calibrate from a narrow polystyrene standard (high DP):

    >>> result = calibrate_emg_broadening(standard_mws, standard_intensities)
    >>> print(f"sigma={result.sigma:.3f}, tau={result.tau:.3f}")

    Calibrate from a living polymer standard (accounts for Poisson width):

    >>> result = calibrate_emg_broadening(
    ...     standard_mws, standard_intensities,
    ...     monomer_mw=104.0  # styrene
    ... )

    Use the calibration with fit_mwd:

    >>> fit_result = fit_mwd(
    ...     mws, intensities,
    ...     order=1.5,
    ...     monomer_mw=100.0,
    ...     init_mon=1.0,
    ...     sigma=result.sigma,
    ...     tau=result.tau
    ... )
    """
    return _calibrate_broadening(
        molecular_weights, intensities,
        emg_broadening, max_sigma, max_tau, monomer_mw
    )


def calibrate_egh_broadening(molecular_weights, intensities,
                             max_sigma = 0.5, max_tau = 0.4,
                            monomer_mw = None):
    """
    Calibrate SEC instrumental broadening from a narrow standard using EGH.

    Fits Exponential-Gaussian Hybrid (EGH) parameters to a narrow
    molecular weight standard to characterize instrumental broadening.
    The EGH model is numerically stable and particularly effective for
    highly asymmetric peaks.

    Based on Lan & Jorgenson, J. Chromatogr. A 915 (2001) 1-13.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which the distribution is measured.
    intensities : ndarray
        Intensity values (should be a narrow, approximately symmetric
        peak for best results).
    max_sigma : float, optional
        Maximum allowed sigma value. Default is 0.5.
    max_tau : float, optional
        Maximum allowed tau value. Default is 0.2.
    monomer_mw : float, optional
        Molecular weight of one monomer unit. If provided, the calibration
        accounts for the natural Poisson dispersity of living polymerization.
        This is important when calibrating with living polymer standards,
        especially at low DP where the intrinsic Poisson width can be
        comparable to or larger than instrumental broadening.

    Returns
    -------
    CalibrationResult
        Fitted broadening parameters and fit quality metrics.

    Notes
    -----
    The fit optimizes sigma, tau, and center simultaneously to minimize
    the sum of squared residuals between the observed and predicted
    intensity profiles.

    When monomer_mw is None, the standard is assumed to be a delta function
    (infinitely narrow). This is appropriate for narrow polymer standards
    with very high DP where the Poisson width is negligible.

    When monomer_mw is provided, the fitting accounts for the Poisson
    distribution of chain lengths in living polymerization:
    - Chain lengths follow Poisson(nup) where nup is the average DP
    - The observed peak width includes both Poisson and instrumental broadening
    - The fit extracts the instrumental broadening component

    Examples
    --------
    Calibrate from a narrow polystyrene standard (high DP):

    >>> result = calibrate_egh_broadening(standard_mws, standard_intensities)
    >>> print(f"sigma={result.sigma:.3f}, tau={result.tau:.3f}")

    Calibrate from a living polymer standard (accounts for Poisson width):

    >>> result = calibrate_egh_broadening(
    ...     standard_mws, standard_intensities,
    ...     monomer_mw=104.0  # styrene
    ... )

    Use the calibration with fit_mwd:

    >>> fit_result = fit_mwd(
    ...     mws, intensities,
    ...     order=1.5,
    ...     monomer_mw=100.0,
    ...     init_mon=1.0,
    ...     sigma=result.sigma,
    ...     tau=result.tau
    ... )
    """
    return _calibrate_broadening(
        molecular_weights, intensities,
        egh_broadening, max_sigma, max_tau, monomer_mw
    )

