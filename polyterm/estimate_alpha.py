"""
Tool for estimating alpha from polymerization conversion points

Fits a curve to the number average molecular weight of the living
portion of the molecular weight distribution versus conversion to
extract the ratio of termination to polymerization rate constants
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .core.utils import calculate_r_squared
from .kinetics.models import (
    STANDARD_KINETICS,
    CONVERSION_TO_TIME,
    LIVING_CHAIN_DP,
)


def _predicted_living_dps(alpha, convs, init_mon, init, order, bn, kinetics):
    """Predict living chain DPs at each conversion for a given alpha."""
    conv_to_time = kinetics[CONVERSION_TO_TIME]
    living_dp = kinetics[LIVING_CHAIN_DP]

    dps = np.empty(len(convs))
    for i, conv in enumerate(convs):
        time = conv_to_time(alpha, init_mon, init, order, conv, bn)
        dps[i] = living_dp(alpha, init_mon, init, order, time, bn)
    return dps


def estimate_alpha(convs, living_mns, monomer_mw, init_mon, init, order,
                   bn=1.0, kinetics=STANDARD_KINETICS):
    """
    Estimate alpha by fitting the conversion evolution of living Mn.

    Uses the conversion evolution of the number average molecular
    weight of the living portion of a MWD to estimate alpha (kt/kp).
    The living Mn may be estimated using the peak of the MWD, however
    this approximation will be worse at higher conversions.

    Parameters
    ----------
    convs : 1darray
        Monomer conversions at which the living Mn values were measured.
    living_mns : 1darray
        Number average molecular weights of the living portion of the
        MWD (may be estimated with the peak molecular weight).
    monomer_mw : float
        Molecular weight of one monomer unit.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial initiator concentration.
    order : float
        Termination reaction order.
    bn : float, optional
        Inverse of the propagation reaction order in living chain.
        Default is 1.0.
    kinetics : dict, optional
        Dictionary of kinetic functions. If None, uses STANDARD_KINETICS
        (standard chain-growth polymerization with n-th order termination).
        Pre-built models are available:
        - STANDARD_KINETICS: Standard polymerization kinetics
        - ROMP_FIRST_ORDER_KINETICS: ROMP with first-order rate law
        - ROMP_SECOND_ORDER_KINETICS: ROMP with second-order rate law

    Returns
    -------
    dict
        Dictionary containing:
        - alpha : float - Estimated kt/kp ratio
        - predicted_mns : ndarray - Predicted living Mn at each conversion
        - r_squared : float - Coefficient of determination of the fit

    Raises
    ------
    ValueError
        If monomer_mw is not positive.
        If convs and living_mns have different lengths.

    Examples
    --------
    >>> result = estimate_alpha(
    ...     convs, living_mns,
    ...     monomer_mw=100.0, init_mon=1.0,
    ...     init=0.01, order=1.5,
    ... )
    >>> result['alpha']
    """
    # Validate inputs
    convs = np.asarray(convs, dtype=float)
    living_mns = np.asarray(living_mns, dtype=float)
    if len(convs) != len(living_mns):
        raise ValueError(
            f"Length mismatch: convs ({len(convs)}) "
            f"!= living_mns ({len(living_mns)})"
        )
    if monomer_mw <= 0:
        raise ValueError("monomer_mw must be positive")

    # Convert observed Mn to DP
    observed_dps = living_mns / monomer_mw

    # Objective: sum of squared residuals in DP space
    def objective(log_alpha):
        alpha = np.exp(log_alpha)
        predicted = _predicted_living_dps(
            alpha, convs, init_mon, init, order, bn, kinetics
        )
        if np.any(~np.isfinite(predicted)):
            return 1e30
        return np.sum((predicted - observed_dps) ** 2)

    # Search in log space with physically reasonable bounds
    # alpha = kt/kp typically ranges from 1e-6 to 10
    result = minimize_scalar(
        objective,
        bounds=(-15, 3),
        method='bounded',
    )

    alpha_fit = np.exp(result.x)

    # Compute predicted Mn for output
    predicted_dps = _predicted_living_dps(
        alpha_fit, convs, init_mon, init, order, bn, kinetics
    )
    predicted_mns = predicted_dps * monomer_mw

    r_squared = calculate_r_squared(living_mns, predicted_mns)

    return {
        'alpha': alpha_fit,
        'predicted_mns': predicted_mns,
        'r_squared': r_squared,
    }
