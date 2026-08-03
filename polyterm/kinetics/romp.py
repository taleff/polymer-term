"""
ROMP kinetics models.

Pre-built kinetics dictionaries for ring-opening metathesis polymerization
with first-order and second-order rate laws, plus the multi-pathway
make_romp_kinetics factory.
"""

import numpy as np

from .models import (
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    CHAIN_DEATH_RATE,
    SECOND_ORDER_DEATH_RATE,
    DEFAULT_COMBINATION,
)

__all__ = [
    'ROMP_FIRST_ORDER_KINETICS',
    'ROMP_SECOND_ORDER_KINETICS',
    'make_romp_kinetics',
]


# ==================== ROMP First Order Conditions ====================
# Kinetic equations for ROMP where termination occurs during the
# catalytic cycle. Assumes a first order rate law for polymerization
# and an identical one for termination.

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


ROMP_FIRST_ORDER_KINETICS = {
    LIVING_CHAIN_CONC: _romp_first_order_living_chain_conc,
    LIVING_CHAIN_DP: _romp_first_order_living_chain_dp,
    CONVERSION_TO_TIME: _romp_first_order_conversion_to_time,
    MONOMER_CONVERSION: _romp_first_order_monomer_conversion,
    CHAIN_DEATH_RATE: _romp_first_order_chain_death_rate,
}


# =================== ROMP Second Order Conditions ===================
# Kinetic equations for ROMP where termination occurs during the
# catalytic cycle. Assumes a second order rate law for polymerization
# and an identical one for termination.

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
    """Rate of chain death for ROMP second-order: -d[P*]/dt ~ [P*] * [M]."""
    b = _romp_second_order_living_chain_conc(alpha, init_mon, init, order, time, bn)
    m = _romp_second_order_monomer_conversion(alpha, init_mon, init, order, time, bn)
    return b * m


ROMP_SECOND_ORDER_KINETICS = {
    LIVING_CHAIN_CONC: _romp_second_order_living_chain_conc,
    LIVING_CHAIN_DP: _romp_second_order_living_chain_dp,
    CONVERSION_TO_TIME: _romp_second_order_conversion_to_time,
    MONOMER_CONVERSION: _romp_second_order_monomer_conversion,
    CHAIN_DEATH_RATE: _romp_second_order_chain_death_rate,
}


# ========== ROMP Multi-Pathway Termination Kinetics ==========
# Propagation: -d[M]/dt = kp * [M] * [P*]
# Termination: -d[P*]/dt = kt1*[M]*[P*] + kt2*[P*] + kt3*[P*]^2
#
# alpha_i = kt_i / kp
#
# Uses reduced time tau = kp*t as the internal time variable.
# During propagation, solves the coupled ODE system numerically.
# After monomer is consumed, uses the analytical Bernoulli solution
# for the aging phase (d[P*]/dtau = -a2*P - a3*P^2).
# This naturally captures post-propagation bimolecular coupling.

def make_romp_kinetics(*, alpha1=None, alpha2=None, alpha3=None):
    """Create kinetics for monomer-dependent + first-order + second-order termination.

    The rate equations are:
        Propagation:  -d[M]/dt  = kp * [M] * [P*]
        Termination:  -d[P*]/dt = kt1*[M]*[P*] + kt2*[P*] + kt3*[P*]^2

    Specify exactly two of the three alpha values (alpha_i = kt_i/kp).
    The unspecified alpha becomes the fitted parameter, passed as the
    ``alpha`` argument in the returned kinetics functions.

    The termination associated with kt3 ([P*]^2 term) undergoes
    combination. The returned kinetics dict sets ``default_combination``
    to 1.0, so ``fit_mwd`` / ``calculate_mwd`` will automatically apply
    combination to the alpha3 pathway unless overridden.

    This model uses reduced time tau = kp*t as its internal time
    variable. During propagation, it solves the coupled ODE system
    for [M], [P*], and nup numerically. After monomer is consumed,
    the Bernoulli analytical solution is used for post-propagation
    aging (bimolecular coupling and first-order decay).

    Parameters
    ----------
    alpha1 : float or None
        kt1/kp, monomer-dependent termination (kt1*[M]*[P*]).
    alpha2 : float or None
        kt2/kp, first-order termination (kt2*[P*]).
    alpha3 : float or None
        kt3/kp, second-order / combination termination (kt3*[P*]^2).

    Returns
    -------
    dict
        Kinetics dictionary compatible with ``fit_mwd`` and ``calculate_mwd``.
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    # Validate: exactly two must be specified
    provided = {
        'alpha1': alpha1, 'alpha2': alpha2, 'alpha3': alpha3
    }
    specified = {k: v for k, v in provided.items() if v is not None}
    unspecified = [k for k, v in provided.items() if v is None]

    if len(specified) != 2:
        raise ValueError(
            f"Specify exactly two of alpha1, alpha2, alpha3. "
            f"Got {len(specified)}: {list(specified.keys())}"
        )
    for name, val in specified.items():
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")

    fitted_name = unspecified[0]

    def _resolve(alpha):
        a1 = alpha if fitted_name == 'alpha1' else alpha1
        a2 = alpha if fitted_name == 'alpha2' else alpha2
        a3 = alpha if fitted_name == 'alpha3' else alpha3
        return a1, a2, a3

    # Single-entry cache for the ODE solution, keyed on full parameters
    _cache = {}

    def _get_solution(a1, a2, a3, init_mon, init):
        """Solve the coupled propagation ODE and cache the result.

        Solves the system:
            d[M]/dtau   = -[M]*[P*]
            d[P*]/dtau  = -(a1*[M]*[P*] + a2*[P*] + a3*[P*]^2)
            d(nup)/dtau = [M]

        Propagation ends when [M] drops below 1e-8 * [M]_0, at which
        point the aging phase uses the analytical Bernoulli solution.
        """
        key = (a1, a2, a3, init_mon, init)
        if key in _cache:
            return _cache[key]
        _cache.clear()

        def ode(tau, y):
            M, P = y[0], y[1]
            if P <= 0:
                return [0.0, 0.0, 0.0]
            dM = -M * P
            dP = -(a1 * M * P + a2 * P + a3 * P * P)
            dnup = M
            return [dM, dP, dnup]

        # Stop when monomer is consumed
        def monomer_gone(tau, y):
            return y[0] - 1e-8 * init_mon
        monomer_gone.terminal = True
        monomer_gone.direction = -1

        # Stop if all chains die during propagation
        def chains_gone(tau, y):
            return y[1] - 1e-10 * init
        chains_gone.terminal = True
        chains_gone.direction = -1

        # Generous upper bound for propagation time
        tau_est = init_mon / max(init, 1e-30)
        tau_max = 200 * tau_est

        sol = solve_ivp(
            ode, [0, tau_max], [init_mon, init, 0.0],
            events=[monomer_gone, chains_gone],
            dense_output=True,
            method='LSODA',
            rtol=1e-10, atol=1e-13,
        )

        tau_end = sol.t[-1]
        y_end = sol.sol(tau_end)

        result = {
            'sol': sol,
            'tau_end': tau_end,
            'P_end': max(y_end[1], 0.0),
            'nup_end': y_end[2],
            'x_end': 1.0 - max(y_end[0], 0.0) / init_mon,
        }
        _cache[key] = result
        return result

    def _aging_P(P0, a2, a3, dtau):
        """Bernoulli analytical solution for [P*] during aging.

        Solves d[P*]/dtau = -(a2*[P*] + a3*[P*]^2) exactly:
          Mixed:       P = 1 / ((1/P0 + a3/a2)*exp(a2*dt) - a3/a2)
          Pure 1st:    P = P0 * exp(-a2*dt)
          Pure 2nd:    P = P0 / (1 + a3*P0*dt)
        """
        if dtau <= 0 or P0 <= 0:
            return max(P0, 0.0)
        if a2 > 1e-30 and a3 > 1e-30:
            ratio = a3 / a2
            denom = (1.0 / P0 + ratio) * np.exp(a2 * dtau) - ratio
            if denom <= 0:
                return 0.0
            return 1.0 / denom
        elif a3 <= 1e-30:
            return P0 * np.exp(-a2 * dtau)
        else:
            return P0 / (1.0 + a3 * P0 * dtau)

    def _eval_at_tau(a1, a2, a3, init_mon, init, tau):
        """Get [M], [P*], and nup at reduced time tau."""
        if tau <= 0:
            return init_mon, init, 0.0
        s = _get_solution(a1, a2, a3, init_mon, init)
        if tau <= s['tau_end']:
            y = s['sol'].sol(tau)
            return max(y[0], 0.0), max(y[1], 0.0), y[2]
        # Aging phase: no propagation, only termination
        dtau = tau - s['tau_end']
        P = _aging_P(s['P_end'], a2, a3, dtau)
        return 0.0, max(P, 0.0), s['nup_end']

    def _romp_md_living_chain_conc(alpha, init_mon, init, order, time, bn=1.0):
        a1, a2, a3 = _resolve(alpha)
        _, P, _ = _eval_at_tau(a1, a2, a3, init_mon, init, time)
        return P

    def _romp_md_living_chain_dp(alpha, init_mon, init, order, time, bn=1.0):
        a1, a2, a3 = _resolve(alpha)
        _, _, nup = _eval_at_tau(a1, a2, a3, init_mon, init, time)
        return nup

    def _romp_md_conversion_to_time(alpha, init_mon, init, order, conversion,
                                     bn=1.0):
        """Map monomer conversion to reduced time tau via root-finding."""
        if conversion <= 0:
            return 0.0
        a1, a2, a3 = _resolve(alpha)
        s = _get_solution(a1, a2, a3, init_mon, init)

        # If requested conversion exceeds what propagation achieves,
        # return the propagation endpoint
        if conversion >= s['x_end']:
            return s['tau_end']

        # Find tau where x(tau) = conversion
        def objective(tau):
            M, _, _ = _eval_at_tau(a1, a2, a3, init_mon, init, tau)
            return (1.0 - M / init_mon) - conversion

        return brentq(objective, 0, s['tau_end'], rtol=1e-10)

    def _romp_md_monomer_conversion(alpha, init_mon, init, order, time,
                                     bn=1.0):
        """Compute [M] at reduced time tau = kp*t."""
        a1, a2, a3 = _resolve(alpha)
        scalar = np.isscalar(time)
        time = np.atleast_1d(np.asarray(time, dtype=float))
        result = np.array([
            _eval_at_tau(a1, a2, a3, init_mon, init, t)[0] for t in time
        ])
        return float(result[0]) if scalar else result

    def _romp_md_chain_death_rate(alpha, init_mon, init, order, time,
                                   bn=1.0):
        """Total death rate -d[P*]/dtau at reduced time tau."""
        a1, a2, a3 = _resolve(alpha)
        M, P, _ = _eval_at_tau(a1, a2, a3, init_mon, init, time)
        if P <= 0:
            return 0.0
        return a1 * M * P + a2 * P + a3 * P * P

    def _romp_md_second_order_death_rate(alpha, init_mon, init, order, time,
                                          bn=1.0):
        """Bimolecular death rate component: a3*[P*]^2."""
        a1, a2, a3 = _resolve(alpha)
        _, P, _ = _eval_at_tau(a1, a2, a3, init_mon, init, time)
        if P <= 0:
            return 0.0
        return a3 * P * P

    return {
        LIVING_CHAIN_CONC: _romp_md_living_chain_conc,
        LIVING_CHAIN_DP: _romp_md_living_chain_dp,
        CONVERSION_TO_TIME: _romp_md_conversion_to_time,
        MONOMER_CONVERSION: _romp_md_monomer_conversion,
        CHAIN_DEATH_RATE: _romp_md_chain_death_rate,
        SECOND_ORDER_DEATH_RATE: _romp_md_second_order_death_rate,
        DEFAULT_COMBINATION: 1.0,
    }
