"""
Functional API for fitting ATRP molecular weight distributions.

This module provides fit_atrp_mwd for fitting experimental SEC/GPC data
to the extended ATRP model of Mastan, Zhou, Zhu (Macromol. Theory Simul.
2014, 23, 227, 10.1002/mats.201300166) implemented in
``polyterm.core.atrp``.  It mirrors :func:`polyterm.fit_mwd.fit_mwd` in
interface and return type, but the forward model is the full extended
ATRP model with explicit activator / deactivator kinetics instead of the
generic kinetics-dict based formulation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from .core.atrp import atrp_distribution, _atrp_rhs
from .core.broadening import compute_broadening_matrix
from .calculate_mwd import _compute_mwd_from_fracs, _compute_dead_chain_fraction
from .utils import calculate_number_average_dp, calculate_r_squared
from .fit_mwd import _prepare_fit_data
from .mwd.mwd import MWDResult

__all__ = ['fit_atrp_mwd']


# ATRP termination is bimolecular in radicals.  Stored on the result so
# downstream code sees the same `order` convention as fit_mwd.
_ATRP_TERMINATION_ORDER = 2.0


def _apply_combination(dead, f):
    """Mix disproportionation and combination termination modes.

    ``atrp_distribution`` returns a dead chain distribution d(r) computed
    from Eq. (24) of the paper, where each termination event produces
    one dead chain at the dying radical's DP (pure disproportionation).
    For a fraction ``f`` of events proceeding by combination instead,
    two radicals merge into a single chain at the sum of their DPs.

    If ``p(r) = d(r) / N`` is the normalized dying-radical DP distribution
    and ``N = sum_r d(r)`` is the expected number of termination events
    per initial chain, then the mixed dead distribution is::

        dead_mixed(r) = (1-f) d(r) + f (N/2) (p * p)(r)
                      = (1-f) d(r) + (f / (2 N)) (d * d)(r)

    since ``np.convolve(d, d)[r] = N^2 (p * p)(r)``.  This preserves
    total monomer mass ``sum_r r * dead(r)`` across any choice of ``f``.
    """
    if f <= 0.0:
        return dead
    n_deaths = dead.sum()
    if n_deaths <= 1e-15:
        return dead
    n_dp = dead.size
    combo = np.convolve(dead, dead)[:n_dp] / (2.0 * n_deaths)
    return (1.0 - f) * dead + f * combo


def _forward_mwd(dps_atrp, k_t, time, *, mon, init, init_c, init_xc,
                 k_p, k_a, k_d, f, segments):
    """Run the extended ATRP model and return (living, dead) on dps_atrp[1:].

    The DP=0 entry is discarded because the broadening matrix is built
    on dps starting at 1 (DP=0 has zero weight contribution and produces
    a log(0) singularity in the broadening centers).
    """
    living, dead = atrp_distribution(
        dps_atrp, time, mon, init, init_c, init_xc,
        k_p, k_a, k_d, k_t, segments=segments,
    )
    dead = _apply_combination(dead, f)
    return living[1:], dead[1:]


def _conversion_at_time(time, *, mon, init, init_c, init_xc,
                        k_p, k_a, k_d, k_t):
    """Monomer conversion at ``time`` for a given k_t (single ODE integration)."""
    y0 = [mon, init_c, init_xc, init, 0.0, 0.0]
    sol = solve_ivp(
        _atrp_rhs, [0.0, time], y0,
        args=(k_p, k_a, k_d, k_t),
        method='BDF', rtol=1e-9, atol=1e-14,
    )
    if not sol.success:
        return float('nan')
    return float(1.0 - sol.y[0, -1] / mon)


def fit_atrp_mwd(mws, ints, monomer_mw, mon, init, init_c, init_xc,
                 k_p, K_ATRP, k_d, f, sigma, tau, *,
                 segments=200, max_fit_points=500):
    """
    Fit an experimental MWD to the extended ATRP kinetic model.

    Fits the termination rate constant :math:`k_t` (and polymerization
    time) to reproduce an experimental SEC/GPC distribution using the
    extended ATRP model of Mastan, Zhou, Zhu (Macromol. Theory Simul.
    2014, 23, 227).  Instrumental broadening is applied the same way
    as :func:`polyterm.fit_mwd.fit_mwd`, via
    :func:`polyterm.core.broadening.compute_broadening_matrix`.

    Parameters
    ----------
    mws : ndarray
        Measured molecular weights from SEC/GPC.
    ints : ndarray
        Detector response at each molecular weight.
    monomer_mw : float
        Molecular weight of one monomer unit (same units as ``mws``).
    mon : float
        Initial monomer concentration ``[M]_0`` (mol/L).
    init : float
        Initial dormant chain concentration ``[PX]_0`` (mol/L).
    init_c : float
        Initial activator catalyst concentration ``[C]_0`` (mol/L).
    init_xc : float
        Initial deactivator catalyst concentration ``[XC]_0`` (mol/L).
    k_p : float
        Propagation rate constant (L mol\u207b\u00b9 s\u207b\u00b9).
    K_ATRP : float
        ATRP equilibrium constant ``k_a / k_d``.  The activation rate
        constant used in the forward model is computed as
        ``k_a = K_ATRP * k_d``.
    k_d : float
        Deactivation rate constant (L mol\u207b\u00b9 s\u207b\u00b9).
    f : float
        Fraction of termination events that proceed by combination, in
        ``[0, 1]``.  ``f = 0`` is pure disproportionation, ``f = 1`` is
        pure combination.
    sigma : float
        SEC broadening parameter (standard deviation in log(MW) space).
    tau : float
        SEC tailing parameter for EGH broadening. Use ``0`` for a
        symmetric Gaussian.
    segments : int, optional
        Number of segments for the extended ATRP model's time
        discretization. Default 200 (reasonable speed/accuracy tradeoff
        for fitting; use 500+ for a final, publication-grade evaluation).
    max_fit_points : int, optional
        Maximum number of MWD points used in the fit (downsampled if
        needed). Default 500.

    Returns
    -------
    MWDResult
        Same dataclass returned by :func:`fit_mwd`.  The fields are:

        - ``alpha``: ``k_t / k_p`` at the fitted solution.
        - ``init``: the supplied ``init`` (not fitted).
        - ``order``: always ``2.0`` (ATRP termination is bimolecular).
        - ``sigma``, ``tau``: the supplied broadening parameters.
        - ``conversion``: monomer conversion at the fitted time.
        - ``r_squared``: coefficient of determination of the fit.

    Examples
    --------
    >>> result = fit_atrp_mwd(
    ...     mws, ints, monomer_mw=100.12,
    ...     mon=4.7, init=4.7/200, init_c=4.7/200, init_xc=4.7/2000,
    ...     k_p=834, K_ATRP=1e-6, k_d=1e6, f=0.0,
    ...     sigma=0.05, tau=0.02,
    ... )
    >>> print(f"k_t = {result.alpha * 834:.3e}  R^2 = {result.r_squared:.4f}")
    """
    # --- Input validation -------------------------------------------------
    if len(mws) != len(ints):
        raise ValueError("mws and ints must have the same length")
    if len(mws) == 0:
        raise ValueError("Input arrays cannot be empty")
    if sigma is None or sigma <= 0:
        raise ValueError("sigma must be positive")
    if tau is None or tau < 0:
        raise ValueError("tau must be non-negative")
    if not (0.0 <= f <= 1.0):
        raise ValueError("f must be in [0, 1]")
    if k_p <= 0 or K_ATRP <= 0 or k_d <= 0:
        raise ValueError("k_p, K_ATRP, and k_d must be positive")
    if mon <= 0 or init <= 0 or init_c <= 0 or init_xc <= 0:
        raise ValueError(
            "mon, init, init_c, and init_xc must all be positive"
        )

    k_a = K_ATRP * k_d

    # --- Prepare experimental data ---------------------------------------
    fit_mws, fit_ints = _prepare_fit_data(mws, ints, max_fit_points)

    # --- DP grid ---------------------------------------------------------
    # atrp_distribution requires a contiguous integer DP grid starting at 0
    # because its internal per-segment convolutions index by DP.  The
    # broadening matrix is built on dps_sec = dps_atrp[1:] to avoid the
    # log(0) singularity at DP=0 (which carries zero weight anyway).
    max_dp = max(int(np.ceil(2.0 * np.max(mws) / monomer_mw)), 10)
    dps_atrp = np.arange(0, max_dp, dtype=int)
    dps_sec = dps_atrp[1:]

    # Broadening is fixed across iterations, precompute once.
    broadenings = compute_broadening_matrix(
        fit_mws, dps_sec, monomer_mw, sigma, tau
    )

    # --- Initial guesses -------------------------------------------------
    # For quasi-living ATRP, only a tiny fraction of chains are active at
    # any time.  The ATRP equilibrium PX + C <-> P* + XC gives an active
    # fraction (relative to dormant) of approximately
    #     [P*]/[PX] ~ K_ATRP * [C] / [XC]
    # at steady state, so the per-chain propagation rate is
    #     d(nu)/dt ~ k_p * [M]_0 * (K_ATRP * [C] / [XC]),
    # which is 4-6 orders of magnitude smaller than the naive k_p*[M]_0.
    # This formula ignores the persistent-radical slowdown, so the true
    # time is typically 2-10x longer.  Used only as the center of the
    # log-time scan below.
    nu = calculate_number_average_dp(fit_mws, fit_ints, monomer_mw)
    active_frac = K_ATRP * init_c / init_xc
    eff_prop_rate = k_p * mon * active_frac
    time_center = max(nu / max(eff_prop_rate, 1e-30), 1.0)
    # k_t for typical vinyl-monomer ATRP is in the 1e7-1e8 range.
    kt_guess = 1e7

    bounds = [(3.0, 12.0), (-2.0, 10.0)]

    # --- Objective function ----------------------------------------------
    data_variance = float(np.var(fit_ints))
    scale_factor = 1.0 / (data_variance + 1e-15)

    def objective(x):
        log_kt, log_time = x
        k_t = 10.0 ** log_kt
        time = 10.0 ** log_time
        try:
            living_sec, dead_sec = _forward_mwd(
                dps_atrp, k_t, time,
                mon=mon, init=init, init_c=init_c, init_xc=init_xc,
                k_p=k_p, k_a=k_a, k_d=k_d, f=f, segments=segments,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            return 1e10

        intensities, _, _ = _compute_mwd_from_fracs(
            dead_sec, living_sec, dps_sec, broadenings
        )
        if not np.all(np.isfinite(intensities)):
            return 1e10
        residuals = intensities - fit_ints
        return scale_factor * float(np.sum(residuals ** 2))

    # --- Coarse log-time scan --------------------------------------------
    # The objective surface has a very narrow valley in log_time at the
    # correct DP (width ~0.1 decades for tight broadening).  The analytic
    # time_center above is typically within 1-2 decades of the true
    # optimum but not inside the valley, and L-BFGS-B's default step
    # sizes step right over it.  Scanning a band around time_center at
    # the initial log(k_t) lands us in the valley before gradient-based
    # refinement.
    log_kt0 = np.log10(kt_guess)
    log_tc = np.log10(time_center)
    scan_lo = max(bounds[1][0], log_tc - 1.5)
    scan_hi = min(bounds[1][1], log_tc + 1.5)
    scan_grid = np.linspace(scan_lo, scan_hi, 31)
    scan_losses = np.array([objective([log_kt0, lt]) for lt in scan_grid])
    log_t0 = float(scan_grid[int(np.argmin(scan_losses))])

    x0 = np.array([log_kt0, log_t0], dtype=float)

    opt = minimize(
        objective, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-9}
    )

    # --- Build result at the optimum -------------------------------------
    log_kt, log_time = opt.x
    k_t_fit = 10.0 ** log_kt
    time_fit = 10.0 ** log_time
    alpha_fit = k_t_fit / k_p

    living_sec, dead_sec = _forward_mwd(
        dps_atrp, k_t_fit, time_fit,
        mon=mon, init=init, init_c=init_c, init_xc=init_xc,
        k_p=k_p, k_a=k_a, k_d=k_d, f=f, segments=segments,
    )
    intensities, dead_ints, live_ints = _compute_mwd_from_fracs(
        dead_sec, living_sec, dps_sec, broadenings
    )

    conversion_val = _conversion_at_time(
        time_fit, mon=mon, init=init, init_c=init_c, init_xc=init_xc,
        k_p=k_p, k_a=k_a, k_d=k_d, k_t=k_t_fit,
    )

    dead_fraction = _compute_dead_chain_fraction(dead_sec, living_sec)
    r_squared = calculate_r_squared(fit_ints, intensities)

    return MWDResult(
        fit_mws, intensities, dead_ints, live_ints,
        dead_fraction, alpha_fit, init, _ATRP_TERMINATION_ORDER,
        sigma, tau, conversion_val, r_squared,
    )
