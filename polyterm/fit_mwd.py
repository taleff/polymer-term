"""
Functional API for kinetic model fitting.

This module provides a pure functional interface for fitting molecular weight
distributions to kinetic models.
"""

import numpy as np
from scipy.optimize import differential_evolution

from .kinetics.models import (
    validate_kinetics,
    find_chain_death_time,
    DEFAULT_COMBINATION,
    MONOMER_CONVERSION,
    STANDARD_KINETICS,
)

from .core.mwd_computation import (
    poisson_distribution,
    compute_dead_chain_fracs,
    compute_live_chain_fracs,
    compute_mwd_from_fracs,
    compute_dead_chain_fraction,
    conversion_to_reduced_time,
    CHAIN_DEATH_FRACTION,
)

from .core.distributions import get_poisson_dp_range

from .core.broadening import compute_broadening_matrix

from .core.initial_guess import estimate_initial_alpha

from .core.utils import (
    calculate_number_average_dp,
    fit_right_edge,
    calculate_r_squared,
)

from .mwd import MWDResult

__all__ = [
    'fit_mwd',
    ]


def fit_mwd(molecular_weights, intensities, order, monomer_mw,
            init_mon, *, sigma=None, tau=None, conversion=None,
            full_time=False, init=None, combination=None, bn=1.0,
            max_fit_points=500, n_quadrature_points=500,
            kinetics=STANDARD_KINETICS,
            distribution=poisson_distribution, seed=42):
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
        Cannot be used together with full_time.
    full_time : bool, optional
        If True, assume the reaction has reached full termination
        (99.99% of chains dead). The time is computed via
        find_chain_death_time and conversion is not fitted. The
        actual monomer conversion at that time is reported in the
        result. Cannot be used together with conversion. Default
        False.
    init : float, optional
        Initial initiator concentration. If None, will be fitted.
    combination : float or None, optional
        Fraction of termination events that proceed by combination,
        between 0.0 (pure disproportionation) and 1.0 (pure
        combination). If None, uses the kinetics dict's
        ``default_combination`` value, or 0.0 if not present.
    bn : float, optional
        Inverse of propagation order. Default 1.0.
    max_fit_points : int, optional
        Maximum points for fitting (downsamples if needed). Default 500.
    n_quadrature_points : int, optional
        Number of Gauss-Legendre quadrature points for integration.
        Higher values improve accuracy but slow computation. Default 100.
    kinetics : dict, optional
        Dictionary of kinetic functions. Default is STANDARD_KINETICS
        (standard chain-growth polymerization with n-th order termination).
        Pre-built models are available:
        - STANDARD_KINETICS: Standard polymerization kinetics
        - ROMP_FIRST_ORDER_KINETICS: ROMP with first-order rate law
        - ROMP_SECOND_ORDER_KINETICS: ROMP with second-order rate law
    distribution : callable, optional
        Function with signature ``distribution(dps, nup)`` that returns
        the probability mass at each degree of polymerization ``dps``
        given kinetic chain length ``nup``. Default is Poisson PMF.
    seed : int or None, optional
        Random seed for the differential evolution optimizer. Default 42
        for reproducibility. Set to None for non-deterministic runs.

    Returns
    -------
    MWDResult
        Dataclass containing the distribution and kinetic parameters.

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
    if len(molecular_weights) != len(intensities):
        raise ValueError("molecular_weights and intensities must have same length")
    if len(molecular_weights) == 0:
        raise ValueError("Input arrays cannot be empty")
    if tau is not None and tau > 0 and sigma is None:
        raise ValueError("tau requires sigma to be specified")
    if order <= 0:
        raise ValueError("order must be positive")
    if combination is None:
        combination = kinetics.get(DEFAULT_COMBINATION, 0.0)
    if not (0 <= combination <= 1):
        raise ValueError("combination must be between 0 and 1")
    if full_time and conversion is not None:
        raise ValueError(
            "Cannot specify both full_time=True and conversion. "
            "full_time computes the time at which 99.99% of chains "
            "have terminated, bypassing conversion entirely."
        )
    if conversion is not None and not (0 <= conversion <= 1):
        raise ValueError("conversion must be between 0 and 1")
    if init is not None and init <= 0:
        raise ValueError("init must be positive")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be positive")
    if tau is not None and tau <= 0:
        raise ValueError("tau must be positive")

    # Prepare data
    fit_mws, fit_ints = _prepare_fit_data(
        molecular_weights, intensities, max_fit_points
    )

    # Calculate distribution characteristics for initial guesses
    nu = calculate_number_average_dp(fit_mws, fit_ints, monomer_mw)
    nup, sigma_est = fit_right_edge(fit_mws, fit_ints, monomer_mw)
    nup = max(nup, nu * 1.001)  # Ensure nup > nu for stability

    # Calculate DP range (add a twice buffer to nup to ensure complete
    # calculation range)
    max_dp = int(np.max(molecular_weights) / monomer_mw)
    dps = np.arange(1, max_dp, dtype=int)
    idx_end = get_poisson_dp_range(2*nup, dps)
    dps = dps[:idx_end]

    # When full_time is set, fix conversion so it's excluded from fitting
    effective_conversion = 1.0 if full_time else conversion

    # Build parameter specification
    param_spec = _build_param_spec(
        init_mon, order, nu, nup, sigma_est, kinetics,
        sigma=sigma, conversion=effective_conversion, init=init
    )

    # Create objective function
    objective = _create_objective(
        fit_mws, fit_ints, dps, monomer_mw, init_mon, order, kinetics,
        sigma=sigma, tau=tau, conversion=effective_conversion, init=init,
        full_time=full_time, combination=combination, bn=bn,
        param_spec=param_spec, n_quadrature_points=n_quadrature_points,
        distribution=distribution
    )

    # Run optimization
    opt_result = _optimize(objective, param_spec, seed=seed)

    # Build and return result
    return _build_fit_result(
        opt_result, fit_mws, fit_ints, dps, monomer_mw, init_mon, order, kinetics,
        sigma=sigma, tau=tau, conversion=effective_conversion, init=init,
        full_time=full_time, combination=combination, bn=bn,
        param_spec=param_spec, n_quadrature_points=n_quadrature_points,
        distribution=distribution
    )


def _prepare_fit_data(molecular_weights, intensities, max_points = 500):
    """
    Prepare MWD data for fitting.

    Downsamples if needed and normalizes the distribution
    for efficient and stable optimization.
    """
    mws = np.asarray(molecular_weights)
    ints = np.asarray(intensities)

    # Downsample if needed
    if len(mws) > max_points:
        # Use uniform spacing in indices
        indices = np.linspace(0, len(mws) - 1, max_points, dtype=int)
        mws = mws[indices]
        ints = ints[indices]

    ints = ints / np.max(ints)

    return mws, ints


def _build_param_spec(init_mon, order, nu, nup, sigma_est, kinetics, *,
                       sigma, conversion, init):
    """Build parameter specification for optimization."""
    # Determine which parameters to fit
    param_names = ['alpha']
    if init is None:
        param_names.append('init')
    if conversion is None:
        param_names.append('conv')
    if sigma is None:
        param_names.append('sigma')

    # Estimate initial values from data shape.
    # nup (right-edge DP) estimates the living chain length, giving
    # init_est ≈ [M]_0 / nup. The ratio nu/nup estimates conversion
    # (fraction of maximum DP achieved). This avoids the old
    # "low conversion assumption" which broke high-conversion samples.
    if conversion is not None:
        mon_frac_est = 1 - conversion
        conv_est = conversion
    elif init is not None:
        mon_frac_est = max(0.001, 1 - nu * init / init_mon)
        conv_est = 1 - mon_frac_est
    else:
        conv_est = nu / nup
        mon_frac_est = 1 - conv_est

    init_est = init if init is not None else init_mon / nup

    # Build bounds (model-agnostic)
    max_reasonable_dp = 10000

    # Initial alpha estimate from observed distribution shape
    alpha_est = estimate_initial_alpha(
        order, mon_frac_est, init_est, init_mon, nu, nup
    )
    # Ensure estimate is positive and finite
    if not (np.isfinite(alpha_est) and alpha_est > 0):
        alpha_est = init_est / init_mon

    # Alpha bounds: 3 orders of magnitude on each side of the
    # data-driven estimate in log10 space. Using log-space ensures
    # uniform sampling across decades, which is critical for scale
    # parameters that can span many orders of magnitude.
    log_alpha_est = np.log10(alpha_est)
    log_min_alpha = log_alpha_est - 3
    log_max_alpha = log_alpha_est + 3

    # Ensure the lower bound extends into the living-polymer regime.
    # For ROMP-like models, alpha ~ init/init_mon marks the threshold
    # between mild and complete termination. The bounds must include
    # this region so the optimizer can find low-termination solutions.
    log_living_ref = np.log10(max(init_est / init_mon, 1e-10))
    log_min_alpha = min(log_min_alpha, log_living_ref - 1)

    init_guess = [log_alpha_est]
    bounds = [(log_min_alpha, log_max_alpha)]

    # Build init_guess/bounds in same order as param_names
    if 'init' in param_names:
        init_guess.append(init_est)
        # Data-driven bounds: 2 orders of magnitude each side of the
        # estimate, clipped to physical limits. The old bounds
        # [init_mon/10000, init_mon] were far too wide, letting DE
        # converge to the trivial [I]_0 -> [M]_0 basin.
        init_lower = max(init_est / 100, init_mon / max_reasonable_dp)
        init_upper = min(init_est * 100, init_mon)
        bounds.append((init_lower, init_upper))

    if 'conv' in param_names:
        # Conversion has natural bounds [0.01, 0.999]
        init_guess.append(conv_est)
        bounds.append((0.01, 0.999))

    if 'sigma' in param_names:
        init_guess.append(sigma_est)
        bounds.append((0.01, 0.5))

    # Clip initial guess to bounds
    for i in range(len(init_guess)):
        lower, upper = bounds[i]
        init_guess[i] = np.clip(init_guess[i], lower, upper)

    return {
        'names': param_names,
        'init_guess': init_guess,
        'bounds': bounds,
    }


def _create_objective(fit_mws, fit_ints, dps, monomer_mw, init_mon, order,
                      kinetics, *, sigma, tau, conversion, init, full_time,
                      combination, bn, param_spec, n_quadrature_points=100,
                      distribution=poisson_distribution):
    """Create objective function for optimization."""
    # Pre-compute broadening matrix if sigma is fixed
    if sigma is not None:
        fixed_broadenings = compute_broadening_matrix(
            fit_mws, dps, monomer_mw, sigma, tau
        )
    else:
        fixed_broadenings = None

    # Scale factor for the objective function, use variance of the
    # data as normalization (standard for least squares)
    data_variance = np.var(fit_ints)
    scale_factor = 1.0 / (data_variance + 1e-15)

    def objective(x):
        params = dict(zip(param_spec['names'], x))

        alpha_val = 10 ** params['alpha']
        sigma_val = params.get('sigma', sigma)
        init_val = params.get('init', init)

        # Determine conversion value (either from params or fixed)
        conv_val = params.get('conv', conversion)

        try:
            # Use pre-computed broadening or compute if sigma is being fitted
            if fixed_broadenings is not None:
                current_broadenings = fixed_broadenings
            else:
                current_broadenings = compute_broadening_matrix(
                    fit_mws, dps, monomer_mw, sigma_val, 0.0
                )

            # Compute time: either from chain death fraction (full_time)
            # or from monomer conversion via the kinetics model.
            if full_time:
                time = find_chain_death_time(
                    kinetics, alpha_val, init_mon, init_val, order,
                    CHAIN_DEATH_FRACTION, bn
                )
            else:
                time = conversion_to_reduced_time(
                    kinetics, alpha_val, init_mon, init_val, order,
                    conv_val, bn
                )

            if not np.isfinite(time) or time < 0:
                return 1e10

            # Compute MWD using helper functions
            pred = _compute_mwd_for_fit(
                time, alpha_val, init_val, dps, current_broadenings,
                init_mon, order, combination, bn, n_quadrature_points, kinetics,
                distribution
            )

            if pred is None:
                return 1e10

            # Scale residuals by data variance to get meaningful loss values
            residuals = pred - fit_ints
            result = scale_factor * np.sum(residuals ** 2)

            return result

        except (ValueError, RuntimeWarning, FloatingPointError):
            return 1e10

    return objective


def _compute_mwd_for_fit(time, alpha, init, dps, broadenings, init_mon,
                         order, combination, bn, n_quadrature_points, kinetics,
                         distribution=poisson_distribution):
    """Compute MWD for fitting (returns only total intensities)."""
    # Guard against invalid parameters
    if not (np.isfinite(alpha) and np.isfinite(init)):
        return None
    if alpha <= 0 or init <= 0:
        return None
    if not np.isfinite(time) or time < 0:
        return None

    # Compute fractions using helper functions
    dead_fracs = compute_dead_chain_fracs(
        time, dps, alpha, init_mon, init, order, bn,
        combination, n_quadrature_points, kinetics, distribution
    )
    live_fracs = compute_live_chain_fracs(
        time, dps, alpha, init_mon, init, order, bn, kinetics, distribution
    )

    # Compute MWD
    intensities, _, _ = compute_mwd_from_fracs(
        dead_fracs, live_fracs, dps, broadenings
    )

    if not np.all(np.isfinite(intensities)):
        return None

    return intensities


def _optimize(objective, param_spec, seed=42):
    """
    Run global optimization using differential evolution with
    local refinement.

    Uses scipy's differential_evolution for robust global search
    across the full parameter space, seeded with the data-driven
    initial guess, followed by L-BFGS-B polishing to refine the
    solution.

    Parameters
    ----------
    objective : callable
        Objective function to minimize.
    param_spec : dict
        Parameter specification with 'init_guess', 'bounds', and 'names'.
    seed : int or None
        Random seed for reproducibility.

    Returns dict with optimization results.
    """
    de_kwargs = dict(
        bounds=param_spec['bounds'],
        x0=param_spec['init_guess'],
        seed=seed,
        tol=1e-4,
        atol=0,
        maxiter=150,
        popsize=15,
        mutation=(0.5, 1.5),
        init='latinhypercube',
    )

    result = differential_evolution(objective, polish=True, **de_kwargs)

    # Guard against polishing pushing into an invalid region.
    # Re-evaluate the objective at the reported optimum to confirm
    # it is not a penalty value (1e10). If it is, re-run without
    # polishing to recover the best valid DE point.
    if objective(result.x) >= 1e9:
        result = differential_evolution(objective, polish=False, **de_kwargs)

    return {
        'x': result.x,
        'fun': result.fun,
        'jac': getattr(result, 'jac', None),
        'hess_inv': getattr(result, 'hess_inv', None),
        'message': f"{result.message} (success={result.success}, nit={result.nit})"
    }


def _build_fit_result(opt_result, fit_mws, fit_ints, dps, monomer_mw,
                      init_mon, order, kinetics, *, sigma, tau, conversion,
                      init, full_time, combination, bn, param_spec,
                      n_quadrature_points=100,
                      distribution=poisson_distribution):
    """Build FitResult from optimization output."""
    param_names = param_spec['names']
    x = opt_result['x']
    params = dict(zip(param_names, x))

    alpha = 10 ** params['alpha']
    sigma_val = params.get('sigma', sigma)
    init_val = params.get('init', init)

    # Calculate time and conversion
    if 'conv' in param_names:
        conversion_val = params['conv']
    else:
        conversion_val = conversion

    # Compute time: either from chain death fraction (full_time)
    # or from monomer conversion via the kinetics model.
    try:
        if full_time:
            time = find_chain_death_time(
                kinetics, alpha, init_mon, init_val, order,
                CHAIN_DEATH_FRACTION, bn
            )
            # Compute the actual monomer conversion at this time
            conversion_val = kinetics[MONOMER_CONVERSION](
                alpha, init_mon, init_val, order, time, bn
            )
        else:
            time = conversion_to_reduced_time(
                kinetics, alpha, init_mon, init_val, order, conversion_val, bn
            )
    except (ValueError, ZeroDivisionError):
        time = np.nan

    # Validate time
    if not np.isfinite(time) or time < 0:
        raise ValueError(
            f"Fitting converged to parameters that produce invalid time: "
            f"alpha={alpha:.6g}, init={init_val:.6g}, conversion={conversion_val:.6g}. "
            f"The kinetics model returned time={time}. "
            f"Try fixing 'init' to your known initiator concentration."
        )

    # Compute fractions using helper functions
    try:
        dead_fracs = compute_dead_chain_fracs(
            time, dps, alpha, init_mon, init_val, order, bn,
            combination, n_quadrature_points, kinetics, distribution
        )
        live_fracs = compute_live_chain_fracs(
            time, dps, alpha, init_mon, init_val, order, bn, kinetics,
            distribution
        )
    except ValueError as e:
        raise ValueError(
            f"Fitting converged to parameters that produce invalid kinetics: "
            f"alpha={alpha:.6g}, init={init_val:.6g}, time={time:.6g}. "
            f"Original error: {e}"
        ) from e

    # Compute broadening matrix
    broadenings = compute_broadening_matrix(
        fit_mws, dps, monomer_mw, sigma_val, tau
    )

    # Compute MWD
    intensities, dead_ints, live_ints = compute_mwd_from_fracs(
        dead_fracs, live_fracs, dps, broadenings
    )

    # Check for degenerate result (empty MWD)
    if np.max(intensities) <= 0:
        raise ValueError(
            f"Fitting converged to parameters that produce an empty MWD: "
            f"alpha={alpha:.6g}, init={init_val:.6g}, "
            f"conversion={conversion_val:.6g}. "
            f"This typically means the optimizer could not find valid "
            f"parameters within the search bounds. Check that the kinetics "
            f"model is appropriate for your data."
        )

    # Compute dead chain fraction and R-squared
    dead_fraction = compute_dead_chain_fraction(dead_fracs, live_fracs)
    r_squared = calculate_r_squared(fit_ints, intensities)

    return MWDResult(
        fit_mws, intensities, dead_ints, live_ints,
        dead_fraction, alpha, init_val, order, sigma_val,
        tau if tau is not None else 0.0,
        conversion_val, r_squared
    )

