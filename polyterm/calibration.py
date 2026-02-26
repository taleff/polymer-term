"""
SEC/GPC calibration utilities for instrumental broadening.

This module provides functions for calibrating SEC instrumental broadening
parameters from narrow molecular weight standards.
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Type, TypeVar

import numpy as np
from scipy.optimize import minimize

from .core.broadening import emg_broadening, egh_broadening
from .utils import calculate_r_squared

__all__ = [
    'calibrate_emg_broadening',
    'calibrate_egh_broadening',
    'EMGCalibrationResult',
    'EGHCalibrationResult',
]


@dataclass(frozen=True)
class EMGCalibrationResult:
    """
    Container for EMG calibration results.

    This immutable class stores the fitted broadening parameters from
    calibrating with a narrow molecular weight standard.

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
    >>> result = EMGCalibrationResult(sigma=0.10, tau=0.05, center=10000.0, r_squared=0.998)
    >>> print(f"Broadening: sigma={result.sigma:.3f}, tau={result.tau:.3f}")
    """

    sigma: float
    tau: float
    center: float
    r_squared: float

    def __repr__(self) -> str:
        """Return string representation of calibration results."""
        return (
            f"EMGCalibrationResult(\n"
            f"  sigma = {self.sigma:.4f}\n"
            f"  tau = {self.tau:.4f}\n"
            f"  center = {self.center:.1f}\n"
            f"  R² = {self.r_squared:.4f}\n"
            f")"
        )


@dataclass(frozen=True)
class EGHCalibrationResult:
    """
    Container for EGH calibration results.

    This immutable class stores the fitted broadening parameters from
    calibrating with a narrow molecular weight standard using the
    Exponential-Gaussian Hybrid (EGH) model.

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
    >>> result = EGHCalibrationResult(sigma=0.10, tau=0.05, center=10000.0, r_squared=0.998)
    >>> print(f"Broadening: sigma={result.sigma:.3f}, tau={result.tau:.3f}")
    """

    sigma: float
    tau: float
    center: float
    r_squared: float

    def __repr__(self) -> str:
        """Return string representation of calibration results."""
        return (
            f"EGHCalibrationResult(\n"
            f"  sigma = {self.sigma:.4f}\n"
            f"  tau = {self.tau:.4f}\n"
            f"  center = {self.center:.1f}\n"
            f"  R² = {self.r_squared:.4f}\n"
            f")"
        )


# Type variable for calibration result classes
T = TypeVar('T', EMGCalibrationResult, EGHCalibrationResult)


def _calibrate_broadening(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    broadening_func: Callable[[np.ndarray, float, float, float], np.ndarray],
    result_class: Type[T],
    max_sigma: float,
    max_tau: float
) -> T:
    """
    Internal helper for calibrating SEC instrumental broadening.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which the distribution is measured.
    intensities : ndarray
        Intensity values from a narrow standard.
    broadening_func : callable
        Broadening function (emg_broadening or egh_broadening).
    result_class : type
        Result dataclass to return (EMGCalibrationResult or EGHCalibrationResult).
    max_sigma : float
        Maximum allowed sigma value.
    max_tau : float
        Maximum allowed tau value.

    Returns
    -------
    result_class instance
        Fitted broadening parameters and fit quality metrics.
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

    return result_class(
        sigma=sigma,
        tau=tau,
        center=center,
        r_squared=r_squared
    )


def calibrate_emg_broadening(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    max_sigma: float = 0.5,
    max_tau: float = 0.2
) -> EMGCalibrationResult:
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

    Returns
    -------
    EMGCalibrationResult
        Fitted broadening parameters and fit quality metrics.

    Notes
    -----
    The fit optimizes sigma, tau, and center simultaneously to minimize
    the sum of squared residuals between the observed and predicted
    intensity profiles.

    For best results, use a narrow molecular weight standard (low PDI)
    with a well-defined peak that spans the molecular weight range of
    interest for your polymer samples.

    Examples
    --------
    Calibrate from a narrow polystyrene standard:

    >>> result = calibrate_emg_broadening(standard_mws, standard_intensities)
    >>> print(f"sigma={result.sigma:.3f}, tau={result.tau:.3f}")

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
        emg_broadening, EMGCalibrationResult,
        max_sigma, max_tau
    )


def calibrate_egh_broadening(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    max_sigma: float = 0.5,
    max_tau: float = 0.2
) -> EGHCalibrationResult:
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

    Returns
    -------
    EGHCalibrationResult
        Fitted broadening parameters and fit quality metrics.

    Notes
    -----
    The fit optimizes sigma, tau, and center simultaneously to minimize
    the sum of squared residuals between the observed and predicted
    intensity profiles.

    For best results, use a narrow molecular weight standard (low PDI)
    with a well-defined peak that spans the molecular weight range of
    interest for your polymer samples.

    Examples
    --------
    Calibrate from a narrow polystyrene standard:

    >>> result = calibrate_egh_broadening(standard_mws, standard_intensities)
    >>> print(f"sigma={result.sigma:.3f}, tau={result.tau:.3f}")

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
        egh_broadening, EGHCalibrationResult,
        max_sigma, max_tau
    )
