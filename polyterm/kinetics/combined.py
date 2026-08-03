"""
Combined first + second order termination kinetics.

Kinetic equations for a polymerization with simultaneous first-order
termination (e.g. CRT in ATRP) and second-order termination
(disproportionation + combination).

The model uses three rate ratios:
  alpha_1 = k_{t1}/k_p  (first-order)
  alpha_2 = k_{t2}/k_p  (second-order disproportionation)
  alpha_3 = k_{t3}/k_p  (second-order combination)

Parameterized by:
  r = alpha_1 / (alpha_1 + (alpha_2+alpha_3)*[I]_0)
    = fraction of initial termination rate that is first-order
  alpha = alpha_2 + alpha_3 (passed as the alpha argument in functions)
  combination = alpha_3 / (alpha_2+alpha_3) (handled by fit_mwd)

Uses the tau substitution from derivation.tex where
  tau = (r2+r3)*(1-exp(-k_{t1}*t))
  R = r2+r3 = (1-r)/r
All quantities expressed without knowing k_p individually.
"""

import numpy as np
from scipy.integrate import quad

from .models import (
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    CHAIN_DEATH_RATE,
    SECOND_ORDER_DEATH_RATE,
)

__all__ = [
    'make_combined_kinetics',
]


def make_combined_kinetics(first_order_ratio):
    """Create kinetics dict for combined first + second order termination.

    Parameters
    ----------
    first_order_ratio : float
        Fraction of the initial termination rate due to first-order
        mechanism: r = alpha_1 / (alpha_1 + (alpha_2+alpha_3)*[I]_0).
        Must be between 0 (exclusive) and 1 (exclusive).
        Use STANDARD_KINETICS with order=2 for pure second-order,
        or order=1 for pure first-order.

    Returns
    -------
    dict
        Kinetics dictionary. The ``alpha`` parameter in the returned
        functions represents alpha_2 + alpha_3 (total second-order rate
        ratio). Use the ``combination`` parameter in ``fit_mwd`` to
        split between disproportionation and combination.
    """
    r = first_order_ratio
    if r <= 0 or r >= 1:
        raise ValueError(
            f"first_order_ratio must be between 0 and 1 (exclusive), "
            f"got {r}. Use STANDARD_KINETICS with order=2 for pure "
            f"second-order or order=1 for pure first-order."
        )

    # R = (1-r)/r is a constant for a given first_order_ratio.
    # It equals [I]_0*(alpha_2+alpha_3)/alpha_1 in physical terms.
    R = (1 - r) / r

    def _combined_living_chain_conc(alpha, init_mon, init, order, time, bn):
        # b = [I]_0 * (R - tau) / (R * (1 + tau))
        # Clamp to zero for tau >= R (all chains dead)
        if time >= R:
            return 0.0
        return init * (R - time) / (R * (1 + time))

    def _combined_living_chain_dp(alpha, init_mon, init, order, time, bn):
        # nup = ([M]_0 * R) / (alpha * [I]_0)
        #       * integral_0^tau (1+s)^{-1/alpha} / (R-s) ds
        if time <= 0:
            return 0.0
        coeff = init_mon * R / (alpha * init)
        # Use substitution u = R - s to help quad near the singularity
        # at s = R. The integrand in u-space is:
        #   (1 + R - u)^{-1/alpha} / u, from u = R down to u = R - tau
        result, _ = quad(
            lambda u: (1 + R - u) ** (-1.0 / alpha) / u,
            R - time, R,
            limit=100,
        )
        return coeff * result

    def _combined_conversion_to_time(alpha, init_mon, init, order, conversion,
                                     bn):
        # [M]/[M]_0 = (1+tau)^{-1/alpha}
        # tau = (1-x)^{-alpha} - 1
        # Guard against overflow for high alpha and high conversion
        log_val = -alpha * np.log(1 - conversion)
        if log_val > 700:
            return np.inf
        return np.exp(log_val) - 1

    def _combined_monomer_conversion(alpha, init_mon, init, order, time, bn):
        # [M] = [M]_0 * (1+tau)^{-1/alpha}
        return init_mon * (1 + time) ** (-1.0 / alpha)

    def _combined_chain_death_rate(alpha, init_mon, init, order, time, bn):
        # Total death rate in tau: -db/dtau = [I]_0*(1+R) / (R*(1+tau)^2)
        if time >= R:
            return 0.0
        return init * (1 + R) / (R * (1 + time) ** 2)

    def _combined_second_order_death_rate(alpha, init_mon, init, order, time,
                                          bn):
        # Second-order component: [I]_0*(R-tau) / (R*(1+tau)^2)
        if time >= R:
            return 0.0
        return init * (R - time) / (R * (1 + time) ** 2)

    return {
        LIVING_CHAIN_CONC: _combined_living_chain_conc,
        LIVING_CHAIN_DP: _combined_living_chain_dp,
        CONVERSION_TO_TIME: _combined_conversion_to_time,
        MONOMER_CONVERSION: _combined_monomer_conversion,
        CHAIN_DEATH_RATE: _combined_chain_death_rate,
        SECOND_ORDER_DEATH_RATE: _combined_second_order_death_rate,
    }
