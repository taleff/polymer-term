"""
Molecular weight distribution calculations for ATRP

This module provides functions for calculating theoretical molecular weight
distributions for atom transfer radical polymerizations
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammaln, gammainc

__all__ = [
    'single_cstr_distribution',
    'atrp_distribution',
]


def _logsumexp_rows(x):
    """Row-wise logsumexp on a 2D array. Faster than scipy.special.logsumexp
    because it skips the array_api_compat dispatch overhead, which dominates
    the single_cstr_distribution hot path."""
    m = np.max(x, axis=1, keepdims=True)
    # Guard against rows that are all -inf (would give nan from (-inf)-(-inf)).
    finite = np.isfinite(m)
    m_safe = np.where(finite, m, 0.0)
    out = np.log(np.sum(np.exp(x - m_safe), axis=1)) + m_safe[:, 0]
    return np.where(finite[:, 0], out, -np.inf)


def _choose_i_max(max_dp, z, phi_d):
    """Tight truncation bound for the sums in Eq. (24).

    The summand of the dormant series, log T(r, i), has the saddle point
    i* ~ sqrt(r * z * phi_d) for large r (from d/di of the log-term).
    The Poisson upper-tail in the dead series becomes negligible beyond
    i ~ z + 6 sqrt(z).  Combining both bounds with a safety margin:
    """
    sp = np.sqrt(max(max_dp * z * max(phi_d, 0.0), 0.0))
    pois = z + 6.0 * np.sqrt(max(z, 0.0))
    return int(max(sp * 6.0 + pois + 30.0, 15.0))


def _single_cstr_with_tables(dps, phi_p, phi_d, z, log_phip,
                             lgamma_r_plus_1, lgamma_rpi_full, lgamma_i_full,
                             i_max_global):
    """Core single-CSTR distribution reusing precomputed gammaln tables.

    Parameters
    ----------
    dps : np.ndarray
        Integer DPs.
    phi_p, phi_d, z : float
        Segment kinetic parameters.
    log_phip : float
        Precomputed log(phi_p) (or -inf if phi_p==0).
    lgamma_r_plus_1 : np.ndarray, shape (n_dp,)
        Precomputed gammaln(dps + 1).
    lgamma_rpi_full : np.ndarray, shape (n_dp, i_max_global)
        Precomputed gammaln(dps[:, None] + i_arr_full[None, :]) for
        i_arr_full = np.arange(1, i_max_global + 1).
    lgamma_i_full : np.ndarray, shape (i_max_global,)
        Precomputed gammaln(i_arr_full). lgamma_i_full[k-1] = gammaln(k).
    i_max_global : int
        Number of columns in the precomputed tables.
    """
    phi_t = 1 - phi_p - phi_d

    # Pick a tight per-segment truncation index.  The saddle-point estimate
    # shrinks i_max from ~max_dp to ~sqrt(max_dp * z * phi_d) + O(sqrt(z)),
    # which is a >10x reduction for typical ATRP segments and is the single
    # biggest speedup in the hot loop.
    max_dp = int(dps[-1]) if dps.size else 0
    i_max = min(_choose_i_max(max_dp, z, phi_d), i_max_global)

    # Slice the precomputed gammaln tables down to the per-segment width.
    log_binom = (
        lgamma_rpi_full[:, :i_max]
        - lgamma_r_plus_1[:, None]
        - lgamma_i_full[None, :i_max]
    )
    # Use float i indices that match gammaln indexing: i_arr[k] = k + 1.
    i_arr = np.arange(1, i_max + 1, dtype=float)
    lgamma_i_plus_1 = lgamma_i_full[1:i_max + 1] if i_max + 1 <= i_max_global \
        else gammaln(i_arr + 1)
    dps_log_phip = dps[:, None] * log_phip  # (n_dp, 1)

    # --- Dormant (living) distribution ------------------------------------
    if z > 0 and phi_d > 0 and phi_p > 0:
        log_zphid = np.log(z * phi_d)
        log_term_d = (
            log_binom
            + i_arr[None, :] * log_zphid
            - lgamma_i_plus_1[None, :]
            - z
            + dps_log_phip
        )
        living = np.exp(_logsumexp_rows(log_term_d))
    else:
        living = np.zeros(dps.size, dtype=float)
    if dps[0] == 0:
        living[0] += np.exp(-z)

    # --- Terminated (dead) distribution -----------------------------------
    if phi_d > 0 and phi_t > 0 and phi_p > 0:
        log_phid = np.log(phi_d)
        q_tail = gammainc(i_arr, z)  # P(Poisson(z) >= j)
        with np.errstate(divide='ignore'):
            log_q = np.where(q_tail > 0, np.log(q_tail), -np.inf)
        log_term_t = (
            log_binom
            + i_arr[None, :] * log_phid
            + log_q[None, :]
            + dps_log_phip
        )
        dead = (phi_t / phi_d) * np.exp(_logsumexp_rows(log_term_t))
    else:
        dead = np.zeros(dps.size, dtype=float)

    return living, dead


def single_cstr_distribution(dps, phi_p, phi_d, z):
    """
    Get the living and dead distribution for constant polymerization params

    Given a CSTR-like setup (where the concentration of all species
    in a polymerization are constant), what is the molecular weight
    distribution of the final polymer?

    Implements Equation (24) of Mastan, Zhou, Zhu, Macromol. Theory
    Simul. 2014, 23, 227 (10.1002/mats.201300166).

    Parameters
    ----------
    dps : np.ndarray
        The degrees of polymerization at which to determine the number
        of chains.
    phi_p : float
        The probability of a chain enchaining a monomers (as opposed
        to terminating or deactivating)
    phi_d : float
        The probability of a chain deactivating (as opposed to
        to terminating or enchaining a monomer)
    z : float
        The average number of activation deactivation cycles that
        occurs within the time

    Returns
    -------
    tuple
        The proportion of chains present at each degree of
        polymerization specified by the dps parameter split into
        the living portion and dead portion
    """
    dps = np.asarray(dps)
    max_dp = int(dps[-1]) if dps.size else 0
    i_max = _choose_i_max(max_dp, z, phi_d)
    i_arr_full = np.arange(1, i_max + 1, dtype=float)
    lgamma_rpi_full = gammaln(dps[:, None] + i_arr_full[None, :])
    lgamma_r_plus_1 = gammaln(dps + 1).astype(float)
    lgamma_i_full = gammaln(i_arr_full)
    log_phip = np.log(phi_p) if phi_p > 0 else -np.inf
    return _single_cstr_with_tables(
        dps, phi_p, phi_d, z, log_phip,
        lgamma_r_plus_1, lgamma_rpi_full, lgamma_i_full, i_max,
    )


def _atrp_rhs(t, y, k_p, k_a, k_d, k_t):
    """ODE right-hand side for ATRP (Eqs. A1-A6 of the paper)."""
    M, C, XC, PX, Pr, _ = y
    r_act = k_a * PX * C
    r_deact = k_d * Pr * XC
    r_term = k_t * Pr * Pr
    return [
        -k_p * M * Pr,                # d[M]/dt
        r_deact - r_act,              # d[C]/dt
        r_act - r_deact,              # d[XC]/dt
        r_deact - r_act,              # d[PX]/dt   (dormant chains)
        r_act - r_deact - r_term,     # d[P*]/dt   (active radicals)
        r_term,                       # d[P]/dt    (dead chains)
    ]


def atrp_distribution(dps, time, mon, init, init_c, init_xc, k_p, k_a,
                      k_d, k_t, segments=100):
    """
    Calculates the molecular weight distribution of an ATRP with
    termination

    Implements the discretized/extended derivation of Mastan, Zhou, Zhu,
    Macromol. Theory Simul. 2014, 23, 227 (10.1002/mats.201300166),
    Section 2.2 and Equations 46-48.  The kinetic ODEs (Eqs. A1-A6) are
    integrated to obtain a time-dependent profile of every species.  The
    polymerization time is split into ``segments`` intervals, each
    interval is treated as a single CSTR with frozen concentrations
    (using the start-of-interval values, as in the paper), and the
    overall dormant and dead distributions are accumulated by convolving
    each interval's growth distribution with the cumulative distribution
    from the previous intervals.

    Parameters
    ----------
    dps : np.ndarray
        The degrees of polymerization at which to determine the number
        of chains.  Must be a contiguous integer range starting at zero
        (e.g. ``np.arange(0, dp_max + 1)``) so that the convolutions
        across intervals remain meaningful.
    time : float
        The time at which the polymerization finishes (s).
    mon : float
        The initial concentration of monomer (mol/L).
    init : float
        The initial concentration of initiator / dormant chains [PX]_0
        (mol/L).
    init_c : float
        The initial concentration of activator catalyst [C]_0 (mol/L).
    init_xc : float
        The initial concentration of deactivator catalyst [XC]_0
        (mol/L).
    k_p : float
        The rate constant of propagation (L mol^-1 s^-1).
    k_a : float
        The rate constant of activation (L mol^-1 s^-1).
    k_d : float
        The rate constant of deactivation (L mol^-1 s^-1).
    k_t : float
        The rate constant of termination (L mol^-1 s^-1).
    segments : int
        The number of discrete segments to split the time space into
        for calculating the MWD. The higher the number of segments,
        the more accurate the calculation.

    Returns
    -------
    tuple
        The proportion of chains present at each degree of
        polymerization specified by the dps parameter split into
        the living portion and dead portion
    """
    dps = np.asarray(dps)
    if init_xc is None:
        init_xc = 0.1 * init_c

    n_dp = dps.size

    # Degenerate case: zero polymerization time -- no chains have grown,
    # so all dormant mass sits at DP 0 and there are no dead chains.
    if time <= 0.0:
        living = np.zeros(n_dp)
        dead = np.zeros(n_dp)
        if n_dp > 0 and dps[0] == 0:
            living[0] = 1.0
        return living, dead

    # Integrate the kinetic ODEs once over [0, time], evaluating the
    # state at the segment boundaries.  BDF handles the radical-balance
    # stiffness that arises once the persistent-radical effect kicks in.
    #
    # Use log-spaced segment boundaries so that the early-time dynamics
    # (where [P*] decays rapidly due to the persistent-radical effect)
    # are resolved with many small segments while the slow late-time
    # plateau uses fewer wide segments.  The first boundary sits at
    # time * 1e-6, which is small enough that no appreciable growth
    # occurs before it for any realistic ATRP recipe.
    t_eval = np.concatenate((
        [0.0],
        np.logspace(np.log10(time * 1e-6), np.log10(time), segments),
    ))
    y0 = [mon, init_c, init_xc, init, 0.0, 0.0]
    sol = solve_ivp(
        _atrp_rhs, [0.0, time], y0,
        t_eval=t_eval, args=(k_p, k_a, k_d, k_t),
        method='BDF', rtol=1e-9, atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError(f"ATRP ODE integration failed: {sol.message}")

    # --- Precompute per-segment kinetic parameters and a global i_max ---
    # Doing this in vectorized form lets us pick a single i_max_global for
    # the precomputed gammaln tables (avoiding per-segment gammaln calls in
    # the inner loop) while still letting each segment slice down to a
    # tighter per-segment i_max.
    Mv, Cv, XCv, _PXv, Prv, _Pdv = sol.y[:, :segments]
    dtv = np.diff(t_eval)

    rate_pv = k_p * Mv
    rate_dv = k_d * XCv
    rate_tv = k_t * Prv
    denomv = rate_pv + rate_dv + rate_tv
    valid = (denomv > 0.0) & (dtv > 0.0)

    phi_pv = np.where(valid, rate_pv / np.where(valid, denomv, 1.0), 0.0)
    phi_dv = np.where(valid, rate_dv / np.where(valid, denomv, 1.0), 0.0)
    zv = np.where(valid, k_a * Cv * dtv, 0.0)

    max_dp = int(dps[-1])
    # Worst-case i_max across all segments; slightly padded so that the
    # per-segment slice always fits without a gammaln refetch.
    max_z = float(np.max(zv)) if zv.size else 0.0
    max_phid = float(np.max(phi_dv)) if phi_dv.size else 0.0
    i_max_global = max(_choose_i_max(max_dp, max_z, max_phid), 15)

    i_arr_full = np.arange(1, i_max_global + 1, dtype=float)
    lgamma_rpi_full = gammaln(dps[:, None] + i_arr_full[None, :])
    lgamma_r_plus_1 = gammaln(dps.astype(float) + 1)
    lgamma_i_full = gammaln(i_arr_full)

    living = None
    dead = None

    for j in range(segments):
        if not valid[j]:
            continue
        z = float(zv[j])
        # Segments with vanishing z (log-spaced early intervals, or
        # depleted catalyst near the end) are effectively identity maps:
        # all chains stay at DP 0 dormant and no dead chains form.  Skip
        # them to avoid wasted work and an unnecessary convolution.
        if z < 1e-12:
            continue

        phi_p = float(phi_pv[j])
        phi_d = float(phi_dv[j])
        log_phip = np.log(phi_p) if phi_p > 0 else -np.inf

        nd_j, nt_j = _single_cstr_with_tables(
            dps, phi_p, phi_d, z, log_phip,
            lgamma_r_plus_1, lgamma_rpi_full, lgamma_i_full, i_max_global,
        )

        if living is None:
            # First non-trivial segment seeds the cumulative distribution.
            living = nd_j
            dead = nt_j
            continue

        # Eqs. 46-47 of the paper.  A chain reaching DP r at the end of
        # segment j either:
        #   (a) was dormant after segment j-1 with DP s and added (r-s)
        #       monomers as a dormant chain in segment j, or
        #   (b) was already dead at the end of segment j-1, or
        #   (c) was dormant after segment j-1 with DP s and was killed
        #       after adding (r-s) monomers in segment j.
        # Convolutions implement (a) and (c); (b) is the carry-over of
        # the previous dead distribution.  Truncating the convolution to
        # n_dp drops mass that would land beyond the requested DP grid.
        dead = dead + np.convolve(living, nt_j)[:n_dp]
        living = np.convolve(living, nd_j)[:n_dp]

    if living is None:
        # No segment contributed (e.g. zero time): everything is still
        # dormant at DP 0.
        living = np.zeros(n_dp)
        dead = np.zeros(n_dp)
        living[0] = 1.0

    return living, dead

