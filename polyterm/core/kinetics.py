"""
Core kinetic calculations for "living polymerizations" with termination.

This module provides fundamental kinetic functions for modeling controlled
chain growth polymerizations with chain termination. All calculations assume
fast initiation and no chain transfer.

IMPORTANT: All input parameters must use consistent units. For example, if
monomer molecular weight is in g/mol, all mass units must be in grams, all
concentration units must use the same volume basis, and all rate constants
must be compatible with those concentration units.
"""

import numpy as np
from scipy.special import expi, gamma, gammaincc
from scipy.integrate import quad
from mpmath import expint
from typing import Union

__all__ = [
    'monomer_conversion',
    'living_chain_concentration',
    'living_chain_dp',
    'conversion_to_time',
    'time_to_chain_death',
]


def monomer_conversion(
    times: Union[float, np.ndarray],
    kp: float,
    kt: float,
    init_mon: float,
    init: float,
    order: float,
    bn: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Calculates the concentration of monomer during a controlled chain growth
    polymerization with termination. The polymerization is assumed to be
    first order in monomer and the reaction starts at t=0.

    Parameters
    ----------
    times : float or ndarray
        Time(s) at which to calculate monomer concentration. Units must be
        consistent with rate constants.
    kp : float
        Propagation rate constant for the reaction P_i* + M → P_{i+1}*.
        Units must be consistent with kt, init_mon, and init.
    kt : float
        Termination rate constant for the reaction P_i* → P_i.
        Units must be consistent with kp, init_mon, and init.
    init_mon : float
        Initial monomer concentration. Units must be consistent with other
        concentration parameters.
    init : float
        Initial initiator (living chain) concentration. Units must be
        consistent with other concentration parameters.
    order : float
        Order of the termination reaction with respect to living chains.
        Common values: 1 (first order), 2 (second order).
    bn : float, optional
        Inverse of the propagation reaction order in living chain.
        Default is 1 (first order in living chain). For some anionic
        polymerizations, bn ≠ 1.

    Returns
    -------
    float or ndarray
        Monomer concentration at the specified time(s). Same shape as input
        times, same units as init_mon.

    Notes
    -----
    The function handles three special cases analytically:
    - order = 1: First-order termination
    - order = 1 + 1/bn: Special case with analytical solution
    - Other orders: General case

    Examples
    --------
    >>> times = np.array([0, 10, 20, 30])
    >>> mon_conc = monomer_conversion(times, kp=100, kt=0.1, init_mon=1.0,
    ...                            init=0.01, order=1.0)
    """
    if order == 1:
        expon = (kp * bn / kt) * (init ** (1/bn)) * (np.exp(-kt * times / bn) - 1)
        return init_mon * np.exp(expon)

    elif np.isclose(order, 1 + (1/bn)):
        expon = 1 + (init ** (1/bn)) * kt * times / bn
        return init_mon * np.power(expon, -(kp * bn / kt))

    else:
        inset = init ** (1 - order) + kt * (order - 1) * times
        in_exp = bn - (1 / (order - 1))
        out_exp = (kp / (kt * (order - 1 - (1/bn)))) * \
                  (init ** (1 + (1/bn) - order) - inset ** in_exp)
        return init_mon * np.exp(out_exp)


def living_chain_concentration(
    init: float,
    order: float,
    time: float
) -> float:
    """
    Calculate living chain concentration at a given reduced time.

    Computes [P*] based on reduced time t' = kt * t / [I]_0 ^ (1-n),
    where kt is the termination rate constant, [I]_0 is the initiator
    concentration, and n is the order.

    Parameters
    ----------
    init : float
        Initial living chain (initiator) concentration. Units must be
        consistent throughout the calculation.
    order : float
        Order of the termination reaction with respect to living chains.
    time : float
        Reduced time (t' = kt * t / [I]_0 ^ (1-n)), dimensionless.

    Returns
    -------
    float
        Concentration of living chains at the specified time. Same units
        as init.

    Notes
    -----
    This function is also used internally with the name _b_val for
    historical reasons.
    """
    if order == 1:
        return init * np.exp(-time)
    elif order == 2:
        return init / (1 + time)
    else:
        return init * (((order - 1) * time + 1) ** (1 / (1 - order)))


def _nup_integrand(
    t: float,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    bn: float
) -> float:
    """Integrand for calculating nup when bn ≠ 1."""
    term1 = monomer_conversion(t, 1/alpha, 1, init_mon, init, order, bn)
    term2 = np.power(living_chain_concentration(init, order, t), (1 - bn) / bn)
    return term1 * term2 / alpha


def living_chain_dp(
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    time: float,
    bn: float = 1.0
) -> float:
    """
    Calculates degree of polymerization of living chains.

    Computes the average degree of polymerization (DP) of the living chains
    at a given reduced time, accounting for the consumption of monomer
    and the decrease in living chain concentration.

    Parameters
    ----------
    alpha : float
        Ratio of termination to propagation rate constants (kt/kp).
        Dimensionless.
    init_mon : float
        Initial monomer concentration. Units must be consistent.
    init : float
        Initial living chain concentration. Units must be consistent
        with init_mon.
    order : float
        Order of the termination reaction.
    time : float
        Reduced time (t' = kt * t  / [I]_0 ^ (1-n)), dimensionless.
    bn : float, optional
        Inverse of the propagation reaction order in living chain.
        Default is 1.0.

    Returns
    -------
    float
        Average degree of polymerization of living chains (dimensionless).

    Notes
    -----
    This function is also named nup or ν'. It represents the average chain
    length of chains that are still actively propagating.

    The calculation method depends on the termination order:
    - order = 1: Analytical solution using exponential integral
    - order = 2: Analytical solution
    - order > 2: Analytical solution using incomplete gamma function
    - 1 < order < 2: Numerical solution using generalized exponential integral
    - bn ≠ 1: Numerical integration required
    """
    # If bn ≠ 1, numerical integration is required
    if bn != 1:
        result, _ = quad(_nup_integrand, 0, time,
                        args=(alpha, init_mon, init, order, bn))
        return result

    # Analytical solutions for bn = 1
    if order == 1:
        # Suppress warnings for edge cases (very large time values)
        with np.errstate(invalid='ignore'):
            return ((init_mon / alpha) * np.exp(-init / alpha) *
                    (expi(init / alpha) - expi(np.exp(-time) * init / alpha)))

    if order == 2:
        # Handle singularity at alpha = 1
        # Limit as alpha -> 1: nu = (init_mon / init) * ln(1 + time)
        if np.isclose(alpha, 1.0, rtol=1e-6):
            return (init_mon / init) * np.log(1 + time)
        return ((init_mon / init / (alpha - 1)) *
                ((1 + time) ** ((alpha - 1) / alpha) - 1))

    # For other orders, use more complex analytical forms
    ratio = init ** (2 - order) / alpha / (order - 2)

    if order > 2:
        coeff = (np.exp(ratio) * (alpha ** ((1 - order) / (2 - order))) *
                ((order - 2) ** (1 / (order - 2))) * (init_mon / alpha))
        right = (gamma((1 - order) / (2 - order)) *
                (gammaincc((1 - order) / (2 - order), ratio) -
                 gammaincc((1 - order) / (2 - order),
                          ratio * ((order - 1) * time + 1) **
                          ((2 - order) / (1 - order)))))
        return coeff * right

    # For 1 < order < 2, use generalized exponential integral
    base = 1 / (2 - order)
    inside = 1 + (order - 1) * time
    power = (2 - order) / (1 - order)
    sol = (init_mon * np.exp(ratio, dtype=complex) *
           np.power(init, 1 - order, dtype=complex) *
           (expint(base, ratio) - inside * expint(base, ratio * (inside ** power)))
           / alpha / (order - 2))
    return complex(sol).real


def conversion_to_time(
    alpha: float,
    init: float,
    order: float,
    conversion: float,
    bn: float = 1.0
) -> float:
    """
    Convert monomer conversion to reduced polymerization time.

    Calculates the reduced time t' = kt * t / [I]_0 ^ (1-n) required to
    reach a specified monomer conversion, given the kinetic parameters.

    Parameters
    ----------
    alpha : float
        Ratio of termination to propagation rate constants (kt/kp).
    init : float
        Initial living chain concentration.
    order : float
        Order of the termination reaction.
    conversion : float
        Monomer conversion, between 0 (no reaction) and 1 (complete
        conversion).
    bn : float, optional
        Inverse of the propagation reaction order in living chain.
        Default is 1.0.

    Returns
    -------
    float
        Reduced time (t' = kt * t / [I]_0 ^ (1-n)) required to reach the
        specified conversion.

    Notes
    -----
    When bn ≠ 1, the function internally adjusts the parameters to use
    equivalent expressions with bn = 1.
    """
    # Adjust parameters if bn ≠ 1
    if bn != 1:
        init = init ** (1 / bn)
        alpha = alpha / bn
        order = 1 + bn * (order - 1)

    mon_frac = 1 - conversion

    if order == 1:
        # Suppress warnings for edge cases (conversion very close to 0 or 1)
        with np.errstate(invalid='ignore'):
            return -np.log((alpha / init) * np.log(mon_frac) + 1) * bn
    elif order == 2:
        return ((mon_frac ** -alpha) - 1) * bn
    else:
        return (((1 - alpha * (order - 2) * np.log(mon_frac) /
                  (init ** (2 - order))) **
                 ((1 - order) / (2 - order)) - 1) / (order - 1)) * bn


def time_to_chain_death(
    chain_conversion: float,
    init: float,
    order: float,
) -> float:
    """
    Computes the reduced time required to reach a specifed conversion of
    living chain ends into dead chain ends.

    Parameters
    ----------
    chain_conversion : float
        Ratio of termination to propagation rate constants (kt/kp).
    init : float
        Initial living chain concentration.
    order : float
        Order of the termination reaction.

    Returns
    -------
    float
        The reduced time at which the specified conversion of living chains
        is achieved
    """
    if order == 1:
        return -np.log(1-chain_conversion)
    else:
        exp = 1 - order
        return ((1-chain_conversion) ** exp - 1) / (-exp)

