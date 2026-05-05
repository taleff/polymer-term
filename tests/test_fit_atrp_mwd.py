"""
Tests for the fit_atrp_mwd functional API.

Round-trip tests use the recipes from Figure 10 of Mastan, Zhou, Zhu
(Macromol. Theory Simul. 2014, 23, 227), which fixes k_p=834,
k_a=1, k_d=1e6 (K_ATRP=1e-6), k_t=1e8 at 50% conversion with two
catalyst loadings controlled by alpha_paper = k_t*[XC]/(k_d*[PX]):

    [M]:[RX]:[C]:[XC] = 200:1:alpha_paper:0.1*alpha_paper

alpha_paper=10 is a well-controlled polymerization with narrow
dispersity; alpha_paper=0.01 is dominated by termination.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from polyterm import fit_atrp_mwd
from polyterm.core.atrp import atrp_distribution, _atrp_rhs
from polyterm.core.broadening import compute_broadening_matrix
from polyterm.calculate_mwd import _compute_mwd_from_fracs


# Figure 10 kinetic constants
MONOMER_MW = 100.12
MON = 4.7
INIT = MON / 200
K_P = 834.0
K_ATRP = 1e-6
K_D = 1e6
K_T_TRUE = 1e8


def _time_to_half_conversion(init_c, init_xc):
    """Integrate the ATRP ODE to find when [M] drops to [M]_0/2."""
    y0 = [MON, init_c, init_xc, INIT, 0.0, 0.0]
    k_a = K_ATRP * K_D

    def residual(log_t):
        t = 10.0 ** log_t
        sol = solve_ivp(
            _atrp_rhs, [0.0, t], y0,
            args=(K_P, k_a, K_D, K_T_TRUE),
            method='BDF', rtol=1e-9, atol=1e-14,
        )
        return sol.y[0, -1] - MON / 2.0

    return 10.0 ** brentq(residual, -2.0, 8.0)


def _synth_mwd(init_c, init_xc, time_true, max_dp, mws, sigma, tau):
    """Produce a broadened synthetic MWD from the true parameters."""
    k_a = K_ATRP * K_D
    dps_atrp = np.arange(0, max_dp, dtype=int)
    living, dead = atrp_distribution(
        dps_atrp, time_true, MON, INIT, init_c, init_xc,
        K_P, k_a, K_D, K_T_TRUE, segments=150,
    )
    dps_sec = dps_atrp[1:]
    broadenings = compute_broadening_matrix(
        mws, dps_sec, MONOMER_MW, sigma, tau
    )
    intensities, _, _ = _compute_mwd_from_fracs(
        dead[1:], living[1:], dps_sec, broadenings
    )
    return intensities / intensities.max()


@pytest.mark.parametrize("alpha_paper,max_dp,mw_max", [
    (10.0,   300, 40_000),
    (0.01,   400, 50_000),
])
def test_round_trip_figure_10(alpha_paper, max_dp, mw_max):
    """fit_atrp_mwd should recover k_t from a synthetic Figure 10 MWD.

    The synthetic data is generated with the true kinetic constants and
    a Gaussian SEC broadening, then fed back into fit_atrp_mwd.  The
    fitted k_t must land within a factor of two of the true value and
    the fit must achieve R^2 > 0.99.
    """
    init_c = alpha_paper * MON / 200
    init_xc = alpha_paper * MON / 2000

    time_true = _time_to_half_conversion(init_c, init_xc)

    sigma = 0.03
    tau = 0.0
    mws = np.linspace(200, mw_max, 200)
    ints = _synth_mwd(init_c, init_xc, time_true, max_dp, mws, sigma, tau)

    result = fit_atrp_mwd(
        mws, ints, monomer_mw=MONOMER_MW,
        mon=MON, init=INIT, init_c=init_c, init_xc=init_xc,
        k_p=K_P, K_ATRP=K_ATRP, k_d=K_D, f=0.0,
        sigma=sigma, tau=tau, segments=100,
    )

    k_t_fit = result.alpha * K_P
    assert result.r_squared > 0.99, f"poor fit: R^2={result.r_squared}"
    assert 0.5 * K_T_TRUE < k_t_fit < 2.0 * K_T_TRUE, (
        f"k_t not recovered: true={K_T_TRUE:.2e}, fit={k_t_fit:.2e}"
    )
    # Figure 10 is at 50% conversion.
    assert 0.35 < result.conversion < 0.65, (
        f"conversion off target: {result.conversion:.3f}"
    )
    # order is hard-coded to 2 for ATRP termination
    assert result.order == 2.0
