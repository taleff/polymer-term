"""
Pre-built kinetics models and utilities.

This module provides the standard kinetics model with unified function
signatures, validation utilities, and helpers for working with custom
kinetics.
"""

import numpy as np

from .kinetics import (
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    monomer_conversion,
)

__all__ = [
    # Key constants
    'LIVING_CHAIN_CONC',
    'LIVING_CHAIN_DP',
    'CONVERSION_TO_TIME',
    'MONOMER_CONVERSION',
    'CHAIN_DEATH_RATE',
    'REQUIRED_KEYS',
    # Standard kinetics
    'STANDARD_KINETICS',
    # ROMP kinetics
    'ROMP_FIRST_ORDER_KINETICS',
    'ROMP_SECOND_ORDER_KINETICS',
    # Functions
    'validate_kinetics',
    'find_chain_death_time',
]


# Key constants to avoid typos
LIVING_CHAIN_CONC = 'living_chain_concentration'
LIVING_CHAIN_DP = 'living_chain_dp'
CONVERSION_TO_TIME = 'conversion_to_time'
MONOMER_CONVERSION = 'monomer_conversion'
CHAIN_DEATH_RATE = 'chain_death_rate'

REQUIRED_KEYS = {
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    CHAIN_DEATH_RATE,
}


def find_chain_death_time(kinetics, alpha, init_mon, init, order,
                          death_fraction=0.9999, bn=1.0):
    """
    Find reduced time at which death_fraction of chains have terminated.

    Uses bisection on living_chain_concentration since it's monotonically
    decreasing. Replaces the old time_to_chain_death function.

    Parameters
    ----------
    kinetics : dict
        Kinetics dictionary with LIVING_CHAIN_CONC function.
    alpha : float
        Ratio kt/kp of rate constants.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial living chain concentration.
    order : float
        Termination reaction order.
    death_fraction : float, optional
        Fraction of chains that have terminated. Default 0.9999.
    bn : float, optional
        Inverse of propagation order. Default 1.0.

    Returns
    -------
    float
        Reduced time at which the specified death fraction is reached.
    """
    from scipy.optimize import brentq

    target_conc = init * (1 - death_fraction)

    def objective(time):
        return kinetics[LIVING_CHAIN_CONC](alpha, init_mon, init, order, time, bn) - target_conc

    # Find upper bound where concentration is below target
    t_max = 1.0
    max_iterations = 100
    for _ in range(max_iterations):
        if objective(t_max) <= 0:
            break
        t_max *= 10
    else:
        # Concentration doesn't decrease enough - use a large time value
        raise ValueError(
            f"Living chain concentration does not decrease to {death_fraction*100}% "
            f"death. Check kinetics parameters: alpha={alpha}, init_mon={init_mon}, "
            f"init={init}. For some kinetics models (e.g., ROMP), this can occur "
            f"when alpha*init_mon < init."
        )

    return brentq(objective, 0, t_max)


def validate_kinetics(kinetics):
    """
    Validate a kinetics dictionary has all required keys.

    Parameters
    ----------
    kinetics : dict
        Dictionary mapping key constants to kinetic functions.

    Raises
    ------
    ValueError
        If required keys are missing, with helpful message about typos.
    """
    missing = REQUIRED_KEYS - set(kinetics.keys())
    if missing:
        extra = set(kinetics.keys()) - REQUIRED_KEYS
        msg = f"Kinetics dict missing required keys: {missing}"
        if extra:
            msg += f". Found unexpected keys (typo?): {extra}"
        raise ValueError(msg)

    return kinetics


# ======================== Default Conditions ========================
# These are the kinetic equations associated with a second order
# polymerization rate law (first order in monomer and active chain
# end) and n order in termination rate rate (n order in just the
# active chain end)

# Wrapper functions with unified signatures
def _std_living_chain_conc(alpha, init_mon, init, order, time, bn=1.0):
    return living_chain_concentration(init, order, time)


def _std_living_chain_dp(alpha, init_mon, init, order, time, bn=1.0):
    return living_chain_dp(alpha, init_mon, init, order, time, bn)


def _std_conversion_to_time(alpha, init_mon, init, order, conversion, bn=1.0):
    return conversion_to_time(alpha, init, order, conversion, bn)


def _std_monomer_conversion(alpha, init_mon, init, order, time, bn=1.0):
    # Convert reduced time to actual time before calling monomer_conversion.
    # Reduced time t' = init^(order-1) * t (with kt=1), so t = t' / init^(order-1).
    if bn != 1:
        eff_order = 1 + bn * (order - 1)
        eff_init = init ** (1 / bn)
        actual_time = time / (eff_init ** (eff_order - 1))
    else:
        actual_time = time / (init ** (order - 1))
    return monomer_conversion(actual_time, 1/alpha, 1, init_mon, init, order, bn)


def _std_chain_death_rate(alpha, init_mon, init, order, time, bn=1.0):
    """Rate of chain death for standard kinetics: [P*]^order * init^(1-order)."""
    b = living_chain_concentration(init, order, time)
    return (b ** order) * (init ** (1 - order))


# Default kinetics using current implementation
STANDARD_KINETICS = {
    LIVING_CHAIN_CONC: _std_living_chain_conc,
    LIVING_CHAIN_DP: _std_living_chain_dp,
    CONVERSION_TO_TIME: _std_conversion_to_time,
    MONOMER_CONVERSION: _std_monomer_conversion,
    CHAIN_DEATH_RATE: _std_chain_death_rate,
}

# ==================== ROMP First Order Conditions ====================
# These are the kinetic equations associated with ROMP such that the
# termination reaction occurs during the catalytic cycle. Here we
# assume a first order rate law for polymerization and an identical
# one for termination

# Functions for ROMP death with bromopyridine abstraction
def _romp_first_order_living_chain_conc(alpha, init_mon, init, order, time, bn):
    return init + alpha * init_mon * (np.exp(-time/alpha)-1)


def _romp_first_order_living_chain_dp(alpha, init_mon, init, order, time, bn):
    return (1/alpha) * np.log(init/(init+alpha*init_mon*(np.exp(-time/alpha)-1)))


def _romp_first_order_conversion_to_time(alpha, init_mon, init, order, conversion, bn):
    return -alpha * np.log(1-conversion)


def _romp_first_order_monomer_conversion(alpha, init_mon, init, order, time, bn):
    return init_mon * np.exp(-time/alpha)


def _romp_first_order_chain_death_rate(alpha, init_mon, init, order, time, bn):
    """Rate of chain death for ROMP first-order: -d[P*]/dt = init_mon * exp(-t/alpha)."""
    return init_mon * np.exp(-time/alpha)


# Default kinetics using current implementation
ROMP_FIRST_ORDER_KINETICS = {
    LIVING_CHAIN_CONC: _romp_first_order_living_chain_conc,
    LIVING_CHAIN_DP: _romp_first_order_living_chain_dp,
    CONVERSION_TO_TIME: _romp_first_order_conversion_to_time,
    MONOMER_CONVERSION: _romp_first_order_monomer_conversion,
    CHAIN_DEATH_RATE: _romp_first_order_chain_death_rate,
}

# =================== ROMP Second Order Conditions ===================
# These are the kinetic equations associated with ROMP such that the
# termination reaction occurs during the catalytic cycle. Here we
# assume a second order rate law for polymerization and an identical
# one for termination

# Functions for ROMP death with no bromopyridine abstraction
def _romp_second_order_time(alpha, init_mon, init, time):
    return np.exp((alpha*init_mon-init)*time/alpha)


def _romp_second_order_conc(alpha, init_mon, init):
    return alpha*init_mon - init


def _romp_second_order_living_chain_conc(alpha, init_mon, init, order, time, bn):
    numer = init * _romp_second_order_conc(alpha, init_mon, init)
    denom = alpha * init_mon * _romp_second_order_time(alpha, init_mon, init, time) - init
    result = numer / denom
    return result


def _romp_second_order_living_chain_dp(alpha, init_mon, init, order, time, bn):
    conc = _romp_second_order_living_chain_conc(alpha, init_mon, init, order, time, bn)
    ratio = init / conc
    return (1 / alpha) * np.log(ratio)


def _romp_second_order_conversion_to_time(alpha, init_mon, init, order, conversion, bn):
    conc_diff = _romp_second_order_conc(alpha, init_mon, init)
    denom = init - alpha * conversion * init_mon
    numer = init * (1 - conversion)
    return (alpha / conc_diff) * np.log(numer / denom)


def _romp_second_order_monomer_conversion(alpha, init_mon, init, order, time, bn):
    return (1/alpha) * (_romp_second_order_conc(alpha, init_mon, init)
                        +_romp_second_order_living_chain_conc(alpha, init_mon, init, order, time, bn))


def _romp_second_order_chain_death_rate(alpha, init_mon, init, order, time, bn):
    """Rate of chain death for ROMP second-order: -d[P*]/dt ∝ [P*] * [M]."""
    b = _romp_second_order_living_chain_conc(alpha, init_mon, init, order, time, bn)
    m = _romp_second_order_monomer_conversion(alpha, init_mon, init, order, time, bn)
    return b * m


# Default kinetics using current implementation
ROMP_SECOND_ORDER_KINETICS = {
    LIVING_CHAIN_CONC: _romp_second_order_living_chain_conc,
    LIVING_CHAIN_DP: _romp_second_order_living_chain_dp,
    CONVERSION_TO_TIME: _romp_second_order_conversion_to_time,
    MONOMER_CONVERSION: _romp_second_order_monomer_conversion,
    CHAIN_DEATH_RATE: _romp_second_order_chain_death_rate,
}

