"""
Initial parameter estimation for kinetic fitting.

This module provides functions for estimating kinetic parameters from
molecular weight distribution characteristics. These estimates serve as
initial guesses for optimization or as quick approximations when full
fitting is not needed.
"""

import numpy as np
from scipy.special import lambertw
from typing import Optional

__all__ = [
    'estimate_initial_alpha',
]


def estimate_initial_alpha(
    order: float,
    mon_frac: float,
    init: float,
    init_mon: float,
    nu: float,
    nup: float
) -> float:
    """
    Estimate alpha (kt/kp) from distribution characteristics.

    Provides initial guesses for optimization based on the observed
    molecular weight distribution shape. The estimation uses analytical
    approximations that depend on the termination order.

    Parameters
    ----------
    order : float
        Order of the termination reaction (e.g., 1.0, 1.5, 2.0).
    mon_frac : float
        Remaining monomer fraction (1 - conversion). Must be between 0 and 1.
    init : float
        Initial initiator concentration estimate.
    init_mon : float
        Initial monomer concentration.
    nu : float
        Number average degree of polymerization from distribution.
    nup : float
        Peak degree of polymerization (living chain DP estimate).

    Returns
    -------
    float
        Estimated alpha (kt/kp) value suitable as optimization starting point.

    Notes
    -----
    The estimation method varies by termination order:
    - order = 1: Uses exponential decay relationships
    - order = 2: Uses Lambert W function for analytical solution
    - Other orders: Uses power-law approximations

    These estimates are approximate and intended only as starting points
    for nonlinear optimization.

    Examples
    --------
    >>> # Estimate alpha for a second-order termination system
    >>> alpha = estimate_initial_alpha(
    ...     order=2.0,
    ...     mon_frac=0.1,  # 90% conversion
    ...     init=0.01,
    ...     init_mon=1.0,
    ...     nu=80,
    ...     nup=100
    ... )
    """
    # High conversion case (mon_frac near 0)
    if np.isclose(mon_frac, 0):
        if order == 1:
            return 0.00001
        elif order == 2:
            ratio = init_mon / init / nup
            alpha = 1 - ratio
            if np.isclose(alpha, 1.0):
                alpha = 0.999
            return alpha
        else:
            return abs(init ** (2 - order) / (order - 2))

    # General case
    if order == 1:
        return np.average([init * ((nup/nu) - 1), -init/np.log(mon_frac)])
    elif order == 2:
        ratio = init_mon / init / nup
        alpha = 1 - ratio + (
            lambertw(
                ratio * np.log(mon_frac) * (mon_frac ** ratio),
                -1
            ) / np.log(mon_frac)
        )
        alpha = alpha.real
        if np.isclose(alpha, 1.0):
            alpha = 0.999
        return alpha
    else:
        return (0.5 * init ** (2 - order) / (order - 2) / np.log(mon_frac))
