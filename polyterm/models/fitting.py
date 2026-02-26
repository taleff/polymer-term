"""
Functional API for kinetic model fitting.

This module provides a pure functional interface for fitting molecular weight
distributions to kinetic models.
"""

from typing import Optional, Dict
from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize, least_squares
from scipy.sparse.linalg import LinearOperator

from ..core.kinetics import (
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    monomer_conversion,
    time_to_chain_death,
)
from ..core.distributions import (
    calculate_dp_range,
    calculate_mwd,
)
from ..core.broadening import (
    compute_broadening_matrix,
    gaussian_broadening,
    emg_broadening,
    egh_broadening,
)
from .estimation import estimate_alpha
from ..utils import (
    calculate_number_average_dp,
    fit_right_edge,
    calculate_r_squared,
    prepare_fit_data,
)


def _get_quadrature_points(n_points: int, time_end: float) -> tuple:
    """
    Get Gauss-Legendre quadrature nodes and weights scaled to [0, time_end].

    Parameters
    ----------
    n_points : int
        Number of quadrature points.
    time_end : float
        Upper limit of integration interval [0, time_end].

    Returns
    -------
    nodes : ndarray
        Quadrature nodes in [0, time_end].
    weights : ndarray
        Corresponding quadrature weights.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    scaled_nodes = (nodes + 1) * time_end / 2
    scaled_weights = weights * time_end / 2
    return scaled_nodes, scaled_weights


def _precompute_poisson_matrix(
    times: np.ndarray,
    dps: np.ndarray,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    bn: float
) -> np.ndarray:
    """
    Precompute Poisson PMFs at all quadrature time points.

    Parameters
    ----------
    times : ndarray
        Quadrature time points.
    dps : ndarray
        Degrees of polymerization.
    alpha : float
        Ratio kt/kp.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial initiator concentration.
    order : float
        Termination order.
    bn : float
        Inverse propagation order.

    Returns
    -------
    poisson_matrix : ndarray, shape (n_times, n_dps)
        poisson_matrix[i, j] = poisson.pmf(dps[j], nup(times[i]))
    """
    # Compute nup at each time point
    nups = np.array([
        living_chain_dp(alpha, init_mon, init, order, t, bn)
        for t in times
    ])

    # Vectorized Poisson computation: shape (n_times, n_dps)
    poisson_matrix = poisson.pmf(dps[np.newaxis, :], nups[:, np.newaxis])

    return poisson_matrix


def _precompute_poisson_matrix_combination(
    times: np.ndarray,
    dps: np.ndarray,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    bn: float
) -> np.ndarray:
    """
    Precompute Poisson PMFs for combination termination (2*nup).

    Same as _precompute_poisson_matrix but uses 2*nup for the Poisson
    parameter, as combination termination produces chains with ~2x the DP.

    Parameters
    ----------
    times : ndarray
        Quadrature time points.
    dps : ndarray
        Degrees of polymerization.
    alpha : float
        Ratio kt/kp.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial initiator concentration.
    order : float
        Termination order.
    bn : float
        Inverse propagation order.

    Returns
    -------
    poisson_matrix : ndarray, shape (n_times, n_dps)
        poisson_matrix[i, j] = poisson.pmf(dps[j], 2*nup(times[i]))
    """
    # Compute nup at each time point
    nups = np.array([
        living_chain_dp(alpha, init_mon, init, order, t, bn)
        for t in times
    ])

    # Use 2*nup for combination termination
    poisson_matrix = poisson.pmf(dps[np.newaxis, :], 2 * nups[:, np.newaxis])

    return poisson_matrix


def _compute_dead_fracs_quadrature(
    times: np.ndarray,
    weights: np.ndarray,
    poisson_matrix: np.ndarray,
    init: float,
    order: float,
    combination: bool
) -> np.ndarray:
    """
    Compute dead chain fractions using fixed quadrature.

    Parameters
    ----------
    times : ndarray
        Quadrature time points.
    weights : ndarray
        Quadrature weights.
    poisson_matrix : ndarray, shape (n_times, n_dps)
        Precomputed Poisson PMFs.
    init : float
        Initial initiator concentration.
    order : float
        Termination order.
    combination : bool
        Whether termination is by combination.

    Returns
    -------
    dead_fracs : ndarray, shape (n_dps,)
        Mole fraction of dead chains at each DP.
    """
    # Compute b(t) at each time point
    b_vals = np.array([
        living_chain_concentration(init, order, t) for t in times
    ])

    # Compute integrand weights: (b^order) * (init^(1-order))
    integrand_weights = (b_vals ** order) * (init ** (1 - order))

    # For combination termination, divide by 2
    if combination:
        integrand_weights = integrand_weights / 2

    # Weighted sum: (n_times,) @ (n_times, n_dps) -> (n_dps,)
    dead_fracs = (weights * integrand_weights) @ poisson_matrix

    return dead_fracs


__all__ = [
    'fit_mwd',
    'FitResult',
    'fit_living_peak',
    'LivingPeakResult',
    'estimate_living_fraction',
    'LivingFractionResult',
]


@dataclass(frozen=True)
class FitResult:
    """
    Container for model fitting results.

    This immutable class stores all outputs from a kinetic model fit,
    including fitted parameters, goodness-of-fit metrics, and the
    predicted distribution.

    Attributes
    ----------
    alpha : float
        Fitted ratio of termination to propagation rate constants (kt/kp).
    init : float
        Fitted or specified initial initiator concentration.
    order : float
        Termination reaction order used in the fit.
    sigma : float
        SEC line broadening parameter (fitted or fixed).
    tau : float
        SEC tailing parameter (0 if Gaussian broadening used).
    conversion : float
        Monomer conversion at the end of polymerization (0 to 1).
    r_squared : float
        Coefficient of determination for the fit.
    molecular_weights : ndarray
        Molecular weights at which the fit was performed.
    predicted_intensities : ndarray
        Model-predicted intensities at molecular_weights.
    dead_chain_intensities : ndarray
        Predicted intensities from dead chains only.
    dead_chain_fraction : float
        Fraction of chains that have terminated.
    fun : float
        Final objective function value.
    jac : ndarray
        Jacobian at the solution.
    hess_inv : LinearOperator
        Inverse Hessian approximation at the solution.
    fit_message : str
        Status message from the optimizer.
    """

    alpha: float
    init: float
    order: float
    sigma: float
    tau: float
    conversion: float
    r_squared: float
    molecular_weights: np.ndarray
    predicted_intensities: np.ndarray
    dead_chain_intensities: np.ndarray
    dead_chain_fraction: float
    fun: float
    jac: np.ndarray
    hess_inv: LinearOperator
    fit_message: str

    def __repr__(self) -> str:
        """Return string representation of fit results."""
        lines = [
            "FitResult(",
            f"  alpha = {self.alpha:.6f}",
            f"  [I]_0 = {self.init:.6f}",
            f"  order = {self.order:.3f}",
            f"  sigma = {self.sigma:.6f}",
        ]
        if self.tau > 0:
            lines.append(f"  tau = {self.tau:.6f}")
        lines.extend([
            f"  conversion = {self.conversion:.4f}",
            f"  R^2 = {self.r_squared:.6f}",
            f"  dead chains = {self.dead_chain_fraction:.4f}",
            ")",
        ])
        return "\n".join(lines)


@dataclass(frozen=True)
class LivingPeakResult:
    """
    Container for living peak fitting results.

    This immutable class stores all outputs from fitting the right edge
    of a molecular weight distribution to determine the living chain peak.

    Attributes
    ----------
    living_intensities : ndarray
        Intensities of the living chain distribution (the fitted broadening
        function scaled to match the right edge of the experimental data).
    dead_intensities : ndarray
        Intensities of the dead chain distribution, calculated as the
        difference between experimental and living distributions.
    dead_chain_fraction : float
        Fraction of chains that have terminated, calculated from the
        integrated areas of dead vs total distributions.
    living_peak_mw : float
        Molecular weight at the peak of the living chain distribution
        (the fitted center parameter).
    r_squared : float
        Coefficient of determination for the right edge fit.
    molecular_weights : ndarray
        Molecular weights at which distributions are evaluated.
    broadening_type : str
        Type of broadening function used ('gaussian', 'emg', or 'egh').
    sigma : float
        Gaussian broadening parameter used.
    tau : float
        Tailing parameter used (0 for Gaussian).
    coefficient : float
        Scaling coefficient from the fit.
    fit_message : str
        Status message from the optimizer.
    """

    living_intensities: np.ndarray
    dead_intensities: np.ndarray
    dead_chain_fraction: float
    living_peak_mw: float
    r_squared: float
    molecular_weights: np.ndarray
    broadening_type: str
    sigma: float
    tau: float
    coefficient: float
    fit_message: str

    def __repr__(self) -> str:
        """Return string representation of living peak fit results."""
        lines = [
            "LivingPeakResult(",
            f"  living_peak_mw = {self.living_peak_mw:.1f}",
            f"  dead_chain_fraction = {self.dead_chain_fraction:.4f}",
            f"  R^2 = {self.r_squared:.6f}",
            f"  broadening = {self.broadening_type}",
            f"  sigma = {self.sigma:.6f}",
        ]
        if self.tau > 0:
            lines.append(f"  tau = {self.tau:.6f}")
        lines.append(")")
        return "\n".join(lines)


@dataclass(frozen=True)
class LivingFractionResult:
    """
    Container for living fraction estimation results.

    This immutable class stores all outputs from fitting a Poisson-broadened
    distribution to the right edge of a molecular weight distribution to
    determine the living chain fraction.

    Attributes
    ----------
    living_peak_mw : float
        Molecular weight at the center of the living chain distribution
        (the fitted center parameter converted from DP).
    dead_chain_fraction : float
        Fraction of chains that have terminated, calculated from the
        integrated areas of dead vs total distributions using mole fractions.
    living_distribution : ndarray
        Weight fraction distribution of living chains at each molecular
        weight. Calculated using Poisson distribution of DPs with
        instrumental broadening applied to each DP.
    dead_distribution : ndarray
        Weight fraction distribution of dead chains, calculated as the
        difference between experimental and living distributions.
    molecular_weights : ndarray
        Molecular weights at which distributions are evaluated.
    sigma : float
        Gaussian broadening parameter used.
    tau : float
        Tailing parameter used (0 for Gaussian).
    monomer_mw : float
cd         Molecular weight of one monomer unit.
    living_peak_dp : float
        Degree of polymerization at the center of the living distribution.
    coefficient : float
        Scaling coefficient from the fit.
    r_squared : float
        Coefficient of determination for the right edge fit.
    fit_message : str
        Status message from the optimizer.
    """

    living_peak_mw: float
    dead_chain_fraction: float
    living_distribution: np.ndarray
    dead_distribution: np.ndarray
    molecular_weights: np.ndarray
    sigma: float
    tau: float
    monomer_mw: float
    living_peak_dp: float
    coefficient: float
    r_squared: float
    fit_message: str

    def __repr__(self) -> str:
        """Return string representation of living fraction results."""
        lines = [
            "LivingFractionResult(",
            f"  living_peak_mw = {self.living_peak_mw:.1f}",
            f"  living_peak_dp = {self.living_peak_dp:.1f}",
            f"  dead_chain_fraction = {self.dead_chain_fraction:.4f}",
            f"  R^2 = {self.r_squared:.6f}",
            f"  sigma = {self.sigma:.6f}",
        ]
        if self.tau > 0:
            lines.append(f"  tau = {self.tau:.6f}")
        lines.append(")")
        return "\n".join(lines)


def estimate_living_fraction(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    sigma: float,
    tau: float,
    monomer_mw: float,
) -> LivingFractionResult:
    """
    Estimate living chain fraction using Poisson-broadened distribution fit.

    Fits the right edge (high MW side) of an experimental molecular weight
    distribution using a physically accurate model: a Poisson distribution
    of degrees of polymerization, converted to mass fractions and broadened
    by instrumental effects. This accounts for the intrinsic width of living
    chain distributions, which is important at low DP where the Poisson
    distribution has significant width.

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
    monomer_mw : float
        Molecular weight of one monomer unit.

    Returns
    -------
    LivingFractionResult
        Contains living and dead distributions, dead chain fraction,
        living peak MW/DP, and fit quality metrics.

    Raises
    ------
    ValueError
        If sigma is not positive.
        If monomer_mw is not positive.
        If molecular_weights and intensities have different lengths.

    Notes
    -----
    The fitting procedure:
    1. Finds the peak of the experimental distribution
    2. Extracts the right edge (from peak to high MW)
    3. For each trial center DP (nup):
       a. Generate Poisson(nup) mole fractions at each DP
       b. Multiply by DP to get mass fractions
       c. Apply EGH/Gaussian broadening to each DP
       d. Sum to get the living chain distribution
    4. Fits center DP and coefficient to match the right edge
    5. Subtracts living from experimental to get dead distribution

    This method is more accurate than fitting a pure Gaussian/EGH function
    because it accounts for the intrinsic Poisson distribution of living
    chain lengths. At low DP, this width can be comparable to or larger
    than instrumental broadening.

    Examples
    --------
    Fit with EGH broadening (calibrated parameters):

    >>> from polyterm import estimate_living_fraction
    >>> result = estimate_living_fraction(
    ...     mws, ints,
    ...     sigma=0.128,
    ...     tau=0.0456,
    ...     monomer_mw=100.0
    ... )
    >>> print(f"Living peak MW = {result.living_peak_mw:.0f}")
    >>> print(f"Living peak DP = {result.living_peak_dp:.1f}")
    >>> print(f"Dead fraction = {result.dead_chain_fraction:.2%}")
    """
    # Validate inputs
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if monomer_mw <= 0:
        raise ValueError("monomer_mw must be positive")
    if len(molecular_weights) != len(intensities):
        raise ValueError(
            f"Length mismatch: molecular_weights ({len(molecular_weights)}) "
            f"!= intensities ({len(intensities)})"
        )

    # Ensure arrays are sorted by increasing MW
    sort_idx = np.argsort(molecular_weights)
    mws = molecular_weights[sort_idx]
    ints = intensities[sort_idx]

    # Find peak and extract right edge (from peak to high MW)
    peak_idx = np.argmax(ints)
    edge_mws = mws[peak_idx:]
    edge_ints = ints[peak_idx:]

    # Calculate DP range needed for broadening
    max_dp = int(np.max(mws) / monomer_mw) + 1
    dps = np.arange(1, max_dp, dtype=int)

    # Pre-compute broadening matrix for efficiency
    # Shape: (len(mws), len(dps))
    dps_mesh, mws_mesh = np.meshgrid(dps, mws)
    true_mws_mesh = dps_mesh * monomer_mw
    broadening_matrix = egh_broadening(mws_mesh, true_mws_mesh, sigma, tau)

    # Also compute for edge only
    _, edge_mws_mesh = np.meshgrid(dps, edge_mws)
    edge_broadening_matrix = egh_broadening(edge_mws_mesh, true_mws_mesh[:len(edge_mws), :], sigma, tau)

    def compute_living_distribution(nup: float, coeff: float) -> np.ndarray:
        """Compute living chain distribution for given center DP."""
        # Poisson mole fractions at each DP
        mole_fracs = poisson.pmf(dps, nup)
        # Convert to mass fractions (multiply by DP)
        mass_fracs = mole_fracs * dps
        # Apply broadening and sum
        living_dist = broadening_matrix @ mass_fracs
        return coeff * living_dist

    def compute_edge_distribution(nup: float, coeff: float) -> np.ndarray:
        """Compute living distribution on edge only for fitting."""
        mole_fracs = poisson.pmf(dps, nup)
        mass_fracs = mole_fracs * dps
        living_dist = edge_broadening_matrix @ mass_fracs
        return coeff * living_dist

    # Fit center DP and coefficient using least squares
    def residual(params):
        nup, coeff = params
        if nup <= 0 or coeff <= 0:
            return np.full_like(edge_ints, 1e10)
        predicted = compute_edge_distribution(nup, coeff)
        return predicted - edge_ints

    # Initial guess: center at peak MW converted to DP
    peak_mw = mws[peak_idx]
    initial_nup = peak_mw / monomer_mw

    # Estimate initial coefficient
    initial_dist = compute_edge_distribution(initial_nup, 1.0)
    peak_dist_val = initial_dist[0] if len(initial_dist) > 0 else 1.0
    initial_coeff = edge_ints[0] / peak_dist_val if peak_dist_val > 0 else 0.01

    initial_guess = [initial_nup, initial_coeff]
    bounds = ([1, 0], [max_dp, np.inf])

    result = least_squares(
        residual,
        x0=initial_guess,
        bounds=bounds,
        method='trf'
    )

    fitted_nup = result.x[0]
    fitted_coeff = result.x[1]
    fitted_center_mw = fitted_nup * monomer_mw

    # Generate living distribution over full MW range
    living_mass_frac = compute_living_distribution(fitted_nup, fitted_coeff)

    # Convert to mole fractions for dead fraction calculation
    # Number (mole) fraction = weight fraction / MW
    mole_frac_exp = ints / mws
    mole_frac_living = living_mass_frac / mws

    # Calculate dead chain fraction from integrated mole fractions
    total_moles = np.sum(mole_frac_exp)
    living_moles = np.sum(mole_frac_living)
    dead_chain_fraction = 1 - (living_moles / total_moles) if total_moles > 0 else 0.0
    dead_chain_fraction = max(0, min(1, dead_chain_fraction))

    # Calculate dead distribution (weight fractions)
    dead_mass_frac = ints - living_mass_frac
    dead_mass_frac = np.maximum(dead_mass_frac, 0)

    # Calculate R-squared for the right edge fit
    edge_predicted = compute_edge_distribution(fitted_nup, fitted_coeff)
    r_squared = calculate_r_squared(edge_ints, edge_predicted)

    # Build status message
    fit_message = f"cost={result.cost:.6e}, nfev={result.nfev}, success={result.success}"

    return LivingFractionResult(
        living_peak_mw=fitted_center_mw,
        dead_chain_fraction=dead_chain_fraction,
        living_distribution=living_mass_frac,
        dead_distribution=dead_mass_frac,
        molecular_weights=mws,
        sigma=sigma,
        tau=tau,
        monomer_mw=monomer_mw,
        living_peak_dp=fitted_nup,
        coefficient=fitted_coeff,
        r_squared=r_squared,
        fit_message=fit_message,
    )


def fit_living_peak(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    broadening_type: str,
    sigma: float,
    tau: float = 0.0,
) -> LivingPeakResult:
    """
    Fit the right edge of a MWD to determine living chain distribution.

    This function fits the high-MW edge of an experimental molecular weight
    distribution using a broadening function (Gaussian, EMG, or EGH) to
    extract the living chain peak. The living distribution is represented
    by the fitted broadening function, and the dead distribution is the
    difference between the experimental data and the living portion.

    This approach is useful when the broadening parameters (sigma, tau)
    are already known from calibration with narrow standards, and you
    want to separate living and dead chain contributions.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights from SEC/GPC measurement. Should be in increasing
        order (low to high MW).
    intensities : ndarray
        Detector response at each molecular weight (weight fractions).
    broadening_type : str
        Type of broadening function to use. Must be one of:
        - 'gaussian': Symmetric Gaussian broadening
        - 'emg': Exponentially Modified Gaussian (asymmetric tailing)
        - 'egh': Exponential-Gaussian Hybrid (asymmetric, more stable)
    sigma : float
        Gaussian broadening parameter (standard deviation in log MW space).
        Should be obtained from calibration with narrow standards.
    tau : float, optional
        Exponential tailing parameter for EMG/EGH broadening. Default is 0.0.
        Ignored for 'gaussian' broadening type.

    Returns
    -------
    LivingPeakResult
        Contains living and dead distributions, dead chain fraction,
        living peak MW, and fit quality metrics.

    Raises
    ------
    ValueError
        If broadening_type is not one of 'gaussian', 'emg', or 'egh'.
        If sigma is not positive.
        If molecular_weights and intensities have different lengths.

    Notes
    -----
    The fitting procedure:
    1. Finds the peak of the experimental distribution
    2. Extracts the right edge (from peak to high MW)
    3. Fits center (peak MW) and coefficient to match the right edge
    4. Uses fitted parameters to generate full living distribution
    5. Subtracts living from experimental to get dead distribution

    The dead chain fraction is calculated as the integrated area of the
    dead distribution divided by the total integrated area, using
    number-average (mole fraction) intensities.

    Examples
    --------
    Fit with EGH broadening (calibrated parameters):

    >>> from polyterm import fit_living_peak
    >>> result = fit_living_peak(
    ...     mws, ints,
    ...     broadening_type='egh',
    ...     sigma=0.128,
    ...     tau=0.0456
    ... )
    >>> print(f"Living peak MW = {result.living_peak_mw:.0f}")
    >>> print(f"Dead fraction = {result.dead_chain_fraction:.2%}")

    Use with pre-calibrated instrument:

    >>> from polyterm import calibrate_egh_broadening, fit_living_peak
    >>> cal = calibrate_egh_broadening(standard_mws, standard_ints)
    >>> result = fit_living_peak(
    ...     sample_mws, sample_ints,
    ...     broadening_type='egh',
    ...     sigma=cal.sigma,
    ...     tau=cal.tau
    ... )
    """
    # Validate inputs
    valid_types = ('gaussian', 'emg', 'egh')
    if broadening_type not in valid_types:
        raise ValueError(
            f"broadening_type must be one of {valid_types}, got '{broadening_type}'"
        )
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if len(molecular_weights) != len(intensities):
        raise ValueError(
            f"Length mismatch: molecular_weights ({len(molecular_weights)}) "
            f"!= intensities ({len(intensities)})"
        )

    # For gaussian, ignore tau
    if broadening_type == 'gaussian':
        tau = 0.0

    # Ensure arrays are sorted by increasing MW
    sort_idx = np.argsort(molecular_weights)
    mws = molecular_weights[sort_idx]
    ints = intensities[sort_idx]

    # Select broadening function
    if broadening_type == 'gaussian':
        def broadening_func(m, center):
            return gaussian_broadening(m, center, sigma)
    elif broadening_type == 'emg':
        def broadening_func(m, center):
            return emg_broadening(m, center, sigma, tau)
    else:  # egh
        def broadening_func(m, center):
            return egh_broadening(m, center, sigma, tau)

    # Find peak and extract right edge (from peak to high MW)
    peak_idx = np.argmax(ints)
    edge_mws = mws[peak_idx:]
    edge_ints = ints[peak_idx:]

    # Fit center and coefficient using least squares
    def residual(params):
        center, coeff = params
        if center <= 0 or coeff <= 0:
            return np.full_like(edge_ints, 1e10)
        predicted = coeff * broadening_func(edge_mws, center)
        return predicted - edge_ints

    # Initial guess: center at peak MW, coefficient from peak intensity
    peak_mw = mws[peak_idx]
    peak_intensity = ints[peak_idx]
    # Estimate coefficient from peak height ratio
    initial_broadening_at_peak = broadening_func(
        np.array([peak_mw]), peak_mw
    )[0]
    initial_coeff = peak_intensity / initial_broadening_at_peak if initial_broadening_at_peak > 0 else 0.01

    initial_guess = [peak_mw, initial_coeff]
    bounds = ([mws[0], 0], [mws[-1], np.inf])

    result = least_squares(
        residual,
        x0=initial_guess,
        bounds=bounds,
        method='trf'
    )

    fitted_center = result.x[0]
    fitted_coeff = result.x[1]

    # Generate living distribution over full MW range
    living_mass_frac = fitted_coeff * broadening_func(mws, fitted_center)

    # Convert to mole fractions for proper dead fraction calculation
    # Number (mole) fraction = weight fraction / MW
    mole_frac_exp = ints / mws
    mole_frac_living = living_mass_frac / mws

    # Normalize for comparison
    mole_frac_exp_norm = mole_frac_exp / np.max(mole_frac_exp) if np.max(mole_frac_exp) > 0 else mole_frac_exp
    mole_frac_living_norm = mole_frac_living / np.max(mole_frac_living) if np.max(mole_frac_living) > 0 else mole_frac_living

    # Dead distribution is the difference (can be negative near peak)
    mole_frac_dead_norm = mole_frac_exp_norm - mole_frac_living_norm

    # Calculate dead chain fraction from integrated areas
    total_area = np.sum(mole_frac_exp_norm)
    living_area = np.sum(mole_frac_living_norm)
    dead_area = np.sum(np.maximum(mole_frac_dead_norm, 0))  # Only positive contributions

    if total_area > 0:
        dead_chain_fraction = dead_area / total_area
        # Alternative: based on living fraction
        dead_chain_fraction_alt = 1 - (living_area / total_area) if living_area < total_area else 0
        # Use the more conservative estimate
        dead_chain_fraction = max(0, min(1, dead_chain_fraction_alt))
    else:
        dead_chain_fraction = 0.0

    # Calculate dead intensities in weight fraction space
    dead_intensities = ints - living_mass_frac
    # Ensure non-negative for visualization (keep actual values for analysis)
    dead_intensities_viz = np.maximum(dead_intensities, 0)

    # Calculate R-squared for the right edge fit
    edge_predicted = fitted_coeff * broadening_func(edge_mws, fitted_center)
    r_squared = calculate_r_squared(edge_ints, edge_predicted)

    # Build status message
    fit_message = f"cost={result.cost:.6e}, nfev={result.nfev}, success={result.success}"

    return LivingPeakResult(
        living_intensities=living_mass_frac,
        dead_intensities=dead_intensities_viz,
        dead_chain_fraction=dead_chain_fraction,
        living_peak_mw=fitted_center,
        r_squared=r_squared,
        molecular_weights=mws,
        broadening_type=broadening_type,
        sigma=sigma,
        tau=tau,
        coefficient=fitted_coeff,
        fit_message=fit_message,
    )


def fit_mwd(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    order: float,
    monomer_mw: float,
    init_mon: float,
    *,
    sigma: Optional[float] = None,
    tau: Optional[float] = None,
    conversion: Optional[float] = None,
    init: Optional[float] = None,
    combination: bool = False,
    bn: float = 1.0,
    max_fit_points: int = 500,
    n_quadrature_points: int = 100,
) -> FitResult:
    """
    Fit kinetic model to a molecular weight distribution.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights from SEC/GPC measurement.
    intensities : ndarray
        Detector response at each molecular weight.
    order : float
        Termination reaction order (e.g., 1.0, 1.5, 2.0).
    monomer_mw : float
        Molecular weight of one monomer unit.
    init_mon : float
        Initial monomer concentration.
    sigma : float, optional
        SEC broadening parameter. If None, will be fitted using Gaussian
        broadening. If provided, broadening is fixed at this value.
    tau : float, optional
        SEC tailing parameter for EGH (exponentially modified Gaussian)
        broadening. Requires sigma to be specified. If None or 0, symmetric
        Gaussian broadening is used.
    conversion : float, optional
        Monomer conversion (0 to 1). If None, will be fitted.
    init : float, optional
        Initial initiator concentration. If None, will be fitted.
    combination : bool, optional
        Whether termination occurs by chain combination. Default False.
    bn : float, optional
        Inverse of propagation order. Default 1.0.
    max_fit_points : int, optional
        Maximum points for fitting (downsamples if needed). Default 500.
    n_quadrature_points : int, optional
        Number of Gauss-Legendre quadrature points for integration.
        Higher values improve accuracy but slow computation. Default 100.

    Returns
    -------
    FitResult
        Fitted parameters, predicted distribution, and metrics.

    Raises
    ------
    ValueError
        If tau is provided but sigma is None (EGH requires known sigma).
        If order is not positive.
        If conversion is outside [0, 1].

    Examples
    --------
    Fit with sigma estimated (Gaussian broadening):

    >>> result = fit_mwd(
    ...     mws, intensities,
    ...     order=1.5,
    ...     monomer_mw=104.15,
    ...     init_mon=1.0
    ... )
    >>> print(f"alpha = {result.alpha:.4f}, R^2 = {result.r_squared:.4f}")

    Fit with calibrated broadening (EGH):

    >>> result = fit_mwd(
    ...     mws, intensities,
    ...     order=1.5,
    ...     monomer_mw=104.15,
    ...     init_mon=1.0,
    ...     sigma=0.05,
    ...     tau=0.02
    ... )

    Batch processing with functools.partial:

    >>> from functools import partial
    >>> fit_my_instrument = partial(
    ...     fit_mwd,
    ...     monomer_mw=104.15,
    ...     init_mon=1.0,
    ...     sigma=0.05,
    ...     tau=0.02
    ... )
    >>> results = [fit_my_instrument(m.mws, m.ints, order=1.5) for m in samples]
    """
    # Validate inputs
    if tau is not None and tau > 0 and sigma is None:
        raise ValueError("tau requires sigma to be specified")
    if order <= 0:
        raise ValueError("order must be positive")
    if conversion is not None and not (0 <= conversion <= 1):
        raise ValueError("conversion must be between 0 and 1")
    if init is not None and init <= 0:
        raise ValueError("init must be positive")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be positive")

    # Determine fitting mode
    fit_sigma = (sigma is None)
    tau_val = tau if (tau is not None and tau > 0) else 0.0

    # Prepare data
    fit_mws, fit_ints = prepare_fit_data(molecular_weights, intensities, max_fit_points)

    # Calculate distribution characteristics for initial guesses
    nu = calculate_number_average_dp(fit_mws, fit_ints, monomer_mw)
    nup, sigma_est = fit_right_edge(fit_mws, fit_ints, monomer_mw)
    nup = max(nup, nu * 1.001)  # Ensure nup > nu for stability

    # Calculate DP range
    dps = calculate_dp_range(fit_mws, fit_ints, monomer_mw, nu, nup)

    # Build parameter specification
    param_spec = _build_param_spec(
        init_mon, order, nu, nup, sigma_est,
        sigma=sigma, conversion=conversion, init=init,
        fit_sigma=fit_sigma
    )

    # Create objective function
    objective = _create_objective(
        fit_mws, fit_ints, dps, monomer_mw, init_mon, order,
        sigma=sigma, tau=tau_val, combination=combination, bn=bn,
        param_spec=param_spec, fit_sigma=fit_sigma,
        n_quadrature_points=n_quadrature_points
    )

    # Run optimization
    opt_result = _optimize(objective, param_spec)

    # Build and return result
    return _build_fit_result(
        opt_result, fit_mws, fit_ints, dps, monomer_mw, init_mon, order,
        sigma=sigma, tau=tau_val, combination=combination, bn=bn,
        param_spec=param_spec, fit_sigma=fit_sigma
    )


def _build_param_spec(
    init_mon: float,
    order: float,
    nu: float,
    nup: float,
    sigma_est: float,
    *,
    sigma: Optional[float],
    conversion: Optional[float],
    init: Optional[float],
    fit_sigma: bool
) -> Dict:
    """
    Build parameter specification for optimization.

    Returns dict with 'names', 'init_guess', 'bounds', and fixed values.
    """
    # Determine which parameters to fit
    param_names = ['alpha']
    if fit_sigma:
        param_names.append('sigma')
    if init is None:
        param_names.append('init')
    if conversion is None:
        param_names.append('time')

    # Estimate initial values
    if conversion is not None:
        mon_frac = 1 - conversion
    elif init is not None:
        mon_frac = max(0.001, 1 - nu * init / init_mon)
    else:
        mon_frac = 0.9  # Low conversion assumption

    init_est = init if init is not None else (init_mon / nu) * (1 - mon_frac)
    alpha_est = estimate_alpha(order, mon_frac, init_est, init_mon, nu, nup)

    # For fixed broadening, start with lower alpha to avoid overestimation
    if not fit_sigma:
        alpha_est = alpha_est * 0.1

    conv_est = conversion if conversion is not None else (1 - mon_frac)
    time_est = conversion_to_time(alpha_est, init_est, order, conv_est, 1.0)

    # Build bounds
    max_reasonable_dp = 10000
    max_starting_ratio = 0.05

    max_alpha = (max_starting_ratio * (init_mon**(2-order)) *
                 (max_reasonable_dp**(order-1)))
    min_alpha = max(1e-8, 0.01 * (init_mon**(2-order)) *
                    (max_reasonable_dp**(order-2)))

    init_guess = [alpha_est]
    bounds = [(min_alpha, max_alpha)]

    if fit_sigma:
        init_guess.append(sigma_est)
        bounds.append((0.01, 0.5))

    if 'init' in param_names:
        init_guess.append(init_est)
        bounds.append((init_mon / max_reasonable_dp, init_mon))

    if 'time' in param_names:
        init_guess.append(time_est)
        init_for_bound = init if init is not None else (init_mon / max_reasonable_dp)
        max_time = time_to_chain_death(0.9999, init_for_bound, order) * 10
        bounds.append((0, max_time))

    # Clip initial guess to bounds
    for i in range(len(init_guess)):
        lower, upper = bounds[i]
        init_guess[i] = np.clip(init_guess[i], lower, upper)

    return {
        'names': param_names,
        'init_guess': init_guess,
        'bounds': bounds,
        '_fixed_init': init,
        '_fixed_conversion': conversion,
    }


def _create_objective(
    fit_mws: np.ndarray,
    fit_ints: np.ndarray,
    dps: np.ndarray,
    monomer_mw: float,
    init_mon: float,
    order: float,
    *,
    sigma: Optional[float],
    tau: float,
    combination: bool,
    bn: float,
    param_spec: Dict,
    fit_sigma: bool,
    n_quadrature_points: int = 100
) -> callable:
    """
    Create objective function for optimization.

    Returns a callable that takes parameter vector and returns sum of squares.
    """
    param_names = param_spec['names']
    fixed_init = param_spec['_fixed_init']
    fixed_conversion = param_spec['_fixed_conversion']

    # Pre-compute broadening matrix if sigma is fixed
    if not fit_sigma:
        broadenings = compute_broadening_matrix(fit_mws, dps, monomer_mw, sigma, tau)
        cache = {'sigma': sigma, 'broadening': broadenings}
    else:
        cache = {'sigma': None, 'broadening': None}

    def objective(x):
        params = dict(zip(param_names, x))

        alpha = params['alpha']
        sig = params.get('sigma', sigma)
        init_val = params.get('init', fixed_init)

        try:
            # Update broadening cache if sigma changed (only when fitting sigma)
            if fit_sigma:
                if cache['sigma'] is None or sig != cache['sigma']:
                    # When fitting sigma, we use Gaussian (tau=0)
                    cache['broadening'] = compute_broadening_matrix(
                        fit_mws, dps, monomer_mw, sig, 0.0
                    )
                    cache['sigma'] = sig

            if 'time' in param_names:
                pred = _calculate_mwd_internal(
                    alpha, init_val, dps, fit_mws, cache['broadening'],
                    init_mon, order, combination, bn,
                    time=params['time'],
                    n_quadrature_points=n_quadrature_points
                )
            else:
                pred = _calculate_mwd_internal(
                    alpha, init_val, dps, fit_mws, cache['broadening'],
                    init_mon, order, combination, bn,
                    conv=fixed_conversion,
                    n_quadrature_points=n_quadrature_points
                )

            residuals = pred - fit_ints
            result = np.sum(residuals ** 2)

            if not np.isfinite(result):
                return 1e10

            return result

        except (ValueError, RuntimeWarning, FloatingPointError):
            return 1e10

    return objective


def _calculate_mwd_internal(
    alpha: float,
    init: float,
    dps: np.ndarray,
    mws: np.ndarray,
    broadenings: np.ndarray,
    init_mon: float,
    order: float,
    combination: bool,
    bn: float,
    time: Optional[float] = None,
    conv: Optional[float] = None,
    n_quadrature_points: int = 100
) -> np.ndarray:
    """
    Calculate MWD given either time or conversion.

    Internal function used during optimization. Uses fixed Gauss-Legendre
    quadrature with precomputed Poisson distributions for efficiency.

    Parameters
    ----------
    alpha : float
        Ratio kt/kp.
    init : float
        Initial initiator concentration.
    dps : ndarray
        Degrees of polymerization.
    mws : ndarray
        Molecular weights for output.
    broadenings : ndarray
        Broadening matrix.
    init_mon : float
        Initial monomer concentration.
    order : float
        Termination order.
    combination : bool
        Whether termination is by combination.
    bn : float
        Inverse propagation order.
    time : float, optional
        Reduced time (if known).
    conv : float, optional
        Conversion (if time not known).
    n_quadrature_points : int, optional
        Number of Gauss-Legendre quadrature points. Default 100.

    Returns
    -------
    ndarray
        Normalized MWD at the specified molecular weights.
    """
    # Guard against invalid parameters
    if not (np.isfinite(alpha) and np.isfinite(init)):
        return np.full(len(mws), np.nan)

    if alpha <= 0 or init <= 0:
        return np.full(len(mws), np.nan)

    # Determine time from conversion if needed
    if time is None:
        if conv is None:
            return np.full(len(mws), np.nan)
        if np.isclose(conv, 1):
            time = time_to_chain_death(0.9999, init, order)
        else:
            time = conversion_to_time(alpha, init, order, conv, bn)

    # Guard against runaway integration
    if not np.isfinite(time) or time < 0:
        return np.full(len(mws), np.nan)

    # Fixed quadrature: precompute Poisson distributions
    quad_times, quad_weights = _get_quadrature_points(n_quadrature_points, time)

    # Handle combination termination: need 2*nup for dead chains
    if combination:
        poisson_matrix = _precompute_poisson_matrix_combination(
            quad_times, dps, alpha, init_mon, init, order, bn
        )
    else:
        poisson_matrix = _precompute_poisson_matrix(
            quad_times, dps, alpha, init_mon, init, order, bn
        )

    dead_fracs = _compute_dead_fracs_quadrature(
        quad_times, quad_weights, poisson_matrix, init, order, combination
    )

    # Living chains at final time
    b = living_chain_concentration(init, order, time)
    nup = living_chain_dp(alpha, init_mon, init, order, time, bn)
    alive_fracs = b * poisson.pmf(dps, nup)

    total_fracs = alive_fracs + dead_fracs

    pred_mwd = np.matmul(broadenings, total_fracs * dps)
    norm = np.trapezoid(pred_mwd, np.log(mws))
    if norm == 0 or not np.isfinite(norm):
        return np.full(len(mws), np.nan)
    return pred_mwd / norm


def _optimize(objective: callable, param_spec: Dict) -> Dict:
    """
    Run optimization using L-BFGS-B.

    Returns dict with optimization results.
    """
    options = {'maxiter': 1000, 'ftol': 1e-9}

    result = minimize(
        objective,
        param_spec['init_guess'],
        method='L-BFGS-B',
        bounds=param_spec['bounds'],
        options=options
    )

    return {
        'x': result.x,
        'fun': result.fun,
        'jac': result.jac,
        'hess_inv': result.hess_inv,
        'message': f"{result.message} (success={result.success}, nit={result.nit})"
    }


def _build_fit_result(
    opt_result: Dict,
    fit_mws: np.ndarray,
    fit_ints: np.ndarray,
    dps: np.ndarray,
    monomer_mw: float,
    init_mon: float,
    order: float,
    *,
    sigma: Optional[float],
    tau: float,
    combination: bool,
    bn: float,
    param_spec: Dict,
    fit_sigma: bool
) -> FitResult:
    """
    Build FitResult from optimization output.
    """
    param_names = param_spec['names']
    x = opt_result['x']
    params = dict(zip(param_names, x))

    alpha = params['alpha']
    sigma_val = params.get('sigma', sigma)
    init_val = params.get('init', param_spec['_fixed_init'])

    # Calculate conversion
    if 'time' in param_names:
        time = params['time']
        conversion_val = 1 - monomer_conversion(
            time, 1/alpha, 1, 1, init_val, order, bn
        )
    else:
        conversion_val = param_spec['_fixed_conversion']
        time = conversion_to_time(alpha, init_val, order, conversion_val, bn)

    # Compute broadening matrix for final predictions
    broadenings = compute_broadening_matrix(fit_mws, dps, monomer_mw, sigma_val, tau)

    # Calculate predicted intensities
    pred_ints = _calculate_mwd_internal(
        alpha, init_val, dps, fit_mws, broadenings,
        init_mon, order, combination, bn,
        time=time
    )

    # Calculate dead chain intensities for visualization
    nu = conversion_val * (init_mon / init_val)
    total_ints = calculate_mwd(
        fit_mws, monomer_mw, nu,
        alpha, init_mon, init_val, order, sigma_val, tau,
        combination, bn
    )

    #height_diff = np.max(pred_ints) / np.max(total_ints) if np.max(total_ints) > 0 else 1.0

    live_ints = calculate_mwd(
        fit_mws, monomer_mw, nu,
        alpha, init_mon, init_val, order, sigma_val, tau,
        combination, bn, live_only=True
    )

    dead_ints = (total_ints - live_ints)# * height_diff

    # Calculate R-squared
    r_squared = calculate_r_squared(fit_ints, pred_ints)

    # Calculate dead chain fraction
    b = living_chain_concentration(init_val, order, time)
    dead_chain_fraction = 1 - b / init_val

    return FitResult(
        alpha=alpha,
        init=init_val,
        order=order,
        sigma=sigma_val,
        tau=tau,
        conversion=conversion_val,
        r_squared=r_squared,
        molecular_weights=fit_mws,
        predicted_intensities=total_ints,
        dead_chain_intensities=dead_ints,
        dead_chain_fraction=dead_chain_fraction,
        fun=opt_result['fun'],
        jac=opt_result['jac'],
        hess_inv=opt_result['hess_inv'],
        fit_message=opt_result['message']
    )
