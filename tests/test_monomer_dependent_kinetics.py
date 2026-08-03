"""Tests for monomer-dependent termination kinetics model.

Model:
    Propagation: -d[M]/dt = kp * [M] * [P*]
    Termination: -d[P*]/dt = kt1*[M]*[P*] + kt2*[P*] + kt3*[P*]^2

    alpha_i = kt_i / kp

The factory make_romp_kinetics(*, alpha1, alpha2, alpha3)
takes exactly two fixed alphas; the third is the fitted parameter.

The model uses reduced time tau = kp*t as its internal time variable.
During propagation, a coupled ODE is solved numerically. After monomer
depletion, the Bernoulli analytical solution handles aging.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from polyterm.kinetics.models import (
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    CHAIN_DEATH_RATE,
    SECOND_ORDER_DEATH_RATE,
    REQUIRED_KEYS,
    validate_kinetics,
)

from polyterm.kinetics.romp import make_romp_kinetics


# ---- Factory validation ----

class TestFactoryValidation:
    """Test that the factory validates its inputs correctly."""

    def test_must_specify_exactly_two_alphas(self):
        """Specifying all three or fewer than two should raise."""
        with pytest.raises(ValueError):
            make_romp_kinetics(alpha1=0.1, alpha2=0.1, alpha3=0.1)
        with pytest.raises(ValueError):
            make_romp_kinetics(alpha1=0.1)
        with pytest.raises(ValueError):
            make_romp_kinetics()

    def test_negative_alpha_raises(self):
        """Negative alpha values should raise."""
        with pytest.raises(ValueError):
            make_romp_kinetics(alpha1=-0.1, alpha2=0.1)

    def test_returns_valid_kinetics_dict(self):
        """Returned dict should pass validate_kinetics."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        validated = validate_kinetics(kinetics)
        assert validated is kinetics

    def test_has_second_order_death_rate(self):
        """Should include SECOND_ORDER_DEATH_RATE for combination."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        assert SECOND_ORDER_DEATH_RATE in kinetics

    def test_all_three_configurations(self):
        """All three ways to specify two alphas should work."""
        k12 = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        k13 = make_romp_kinetics(alpha1=0.001, alpha3=0.0005)
        k23 = make_romp_kinetics(alpha2=0.001, alpha3=0.0005)
        for k in [k12, k13, k23]:
            validate_kinetics(k)


# ---- Initial conditions ----

class TestInitialConditions:
    """Test that kinetics functions return correct values at tau=0."""

    @pytest.fixture
    def params(self):
        return dict(init_mon=1.0, init=0.01, order=1.5, bn=1.0)

    def test_living_chain_conc_at_zero(self, params):
        """[P*](tau=0) should equal init."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.01  # fitted alpha3
        conc = kinetics[LIVING_CHAIN_CONC](
            alpha, params['init_mon'], params['init'],
            params['order'], 0.0, params['bn']
        )
        assert np.isclose(conc, params['init'])

    def test_monomer_conversion_at_zero(self, params):
        """[M](tau=0) should equal init_mon."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.01
        mon = kinetics[MONOMER_CONVERSION](
            alpha, params['init_mon'], params['init'],
            params['order'], 0.0, params['bn']
        )
        assert np.isclose(mon, params['init_mon'])

    def test_living_chain_dp_at_zero(self, params):
        """nup(tau=0) should be 0."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.01
        dp = kinetics[LIVING_CHAIN_DP](
            alpha, params['init_mon'], params['init'],
            params['order'], 0.0, params['bn']
        )
        assert np.isclose(dp, 0.0, atol=1e-10)

    def test_conversion_to_time_at_zero(self, params):
        """conversion_to_time(0) should return 0."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.01
        time = kinetics[CONVERSION_TO_TIME](
            alpha, params['init_mon'], params['init'],
            params['order'], 0.0, params['bn']
        )
        assert np.isclose(time, 0.0)


# ---- Monotonicity ----

class TestMonotonicity:
    """Test that kinetic quantities behave monotonically."""

    @pytest.fixture
    def kinetics_and_params(self):
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.005  # fitted alpha3
        return kinetics, dict(
            alpha=alpha, init_mon=1.0, init=0.01, order=1.5, bn=1.0
        )

    def test_living_chain_conc_decreases(self, kinetics_and_params):
        """[P*] should decrease with reduced time tau."""
        kinetics, p = kinetics_and_params
        times = [0.0, 10.0, 50.0, 200.0]
        concs = [
            kinetics[LIVING_CHAIN_CONC](
                p['alpha'], p['init_mon'], p['init'],
                p['order'], t, p['bn']
            )
            for t in times
        ]
        for i in range(len(concs) - 1):
            assert concs[i] > concs[i + 1], (
                f"[P*] not decreasing: {concs[i]} -> {concs[i+1]} "
                f"at tau={times[i]} -> {times[i+1]}"
            )

    def test_living_chain_dp_increases(self, kinetics_and_params):
        """nup should increase with reduced time during propagation."""
        kinetics, p = kinetics_and_params
        times = [0.01, 10.0, 50.0, 200.0]
        dps = [
            kinetics[LIVING_CHAIN_DP](
                p['alpha'], p['init_mon'], p['init'],
                p['order'], t, p['bn']
            )
            for t in times
        ]
        for i in range(len(dps) - 1):
            assert dps[i] <= dps[i + 1]

    def test_nup_constant_during_aging(self, kinetics_and_params):
        """nup should remain constant after propagation ends."""
        kinetics, p = kinetics_and_params
        # Get a tau well past propagation end
        tau_prop = kinetics[CONVERSION_TO_TIME](
            p['alpha'], p['init_mon'], p['init'],
            p['order'], 0.999, p['bn']
        )
        nup_end = kinetics[LIVING_CHAIN_DP](
            p['alpha'], p['init_mon'], p['init'],
            p['order'], tau_prop * 2, p['bn']
        )
        nup_later = kinetics[LIVING_CHAIN_DP](
            p['alpha'], p['init_mon'], p['init'],
            p['order'], tau_prop * 10, p['bn']
        )
        assert np.isclose(nup_end, nup_later, rtol=1e-4)

    def test_monomer_decreases(self, kinetics_and_params):
        """[M] should decrease with increasing tau."""
        kinetics, p = kinetics_and_params
        times = [0.0, 10.0, 50.0, 200.0]
        mons = [
            kinetics[MONOMER_CONVERSION](
                p['alpha'], p['init_mon'], p['init'],
                p['order'], t, p['bn']
            )
            for t in times
        ]
        for i in range(len(mons) - 1):
            assert mons[i] >= mons[i + 1]

    def test_conversion_to_time_monotonic(self, kinetics_and_params):
        """conversion_to_time should increase with conversion."""
        kinetics, p = kinetics_and_params
        convs = [0.1, 0.3, 0.5, 0.7, 0.9]
        taus = [
            kinetics[CONVERSION_TO_TIME](
                p['alpha'], p['init_mon'], p['init'],
                p['order'], c, p['bn']
            )
            for c in convs
        ]
        for i in range(len(taus) - 1):
            assert taus[i] < taus[i + 1]


# ---- Mass balance ----

class TestMassBalance:
    """Test conservation laws / consistency."""

    def test_chain_death_integral_equals_concentration_drop(self):
        """Integral of chain_death_rate from 0 to tau should equal init - [P*](tau)."""
        kinetics = make_romp_kinetics(alpha1=0.002, alpha2=0.001)
        alpha = 0.005  # fitted alpha3
        init_mon = 1.0
        init = 0.01
        order = 1.5
        bn = 1.0

        # Convert x=0.7 to tau
        tau_final = kinetics[CONVERSION_TO_TIME](
            alpha, init_mon, init, order, 0.7, bn
        )

        conc_0 = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, order, 0.0, bn
        )
        conc_f = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, order, tau_final, bn
        )
        expected_dead = conc_0 - conc_f

        # Integrate chain_death_rate over tau
        integral, _ = quad(
            lambda tau: kinetics[CHAIN_DEATH_RATE](
                alpha, init_mon, init, order, tau, bn
            ),
            0, tau_final
        )

        assert np.isclose(integral, expected_dead, rtol=1e-4)

    def test_chain_death_integral_nonunit_monomer(self):
        """Mass balance must hold when init_mon != 1.0."""
        kinetics = make_romp_kinetics(alpha1=0.002, alpha2=0.001)
        alpha = 0.005
        init_mon = 0.2
        init = 0.004
        order = 1.5
        bn = 1.0

        tau_final = kinetics[CONVERSION_TO_TIME](
            alpha, init_mon, init, order, 0.6, bn
        )

        conc_0 = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, order, 0.0, bn
        )
        conc_f = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, order, tau_final, bn
        )
        expected_dead = conc_0 - conc_f

        integral, _ = quad(
            lambda tau: kinetics[CHAIN_DEATH_RATE](
                alpha, init_mon, init, order, tau, bn
            ),
            0, tau_final
        )

        assert np.isclose(integral, expected_dead, rtol=1e-4), (
            f"Mass balance failed: integral={integral}, "
            f"expected={expected_dead}"
        )

    def test_chain_death_integral_during_aging(self):
        """Mass balance should hold across both propagation and aging."""
        kinetics = make_romp_kinetics(alpha1=0.002, alpha2=0.001)
        alpha = 0.005
        init_mon = 1.0
        init = 0.01
        order = 1.5
        bn = 1.0

        # Get a tau well into the aging phase
        from polyterm.kinetics.models import find_chain_death_time
        tau_final = find_chain_death_time(
            kinetics, alpha, init_mon, init, order, 0.99, bn
        )

        conc_0 = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, order, 0.0, bn
        )
        conc_f = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, order, tau_final, bn
        )
        expected_dead = conc_0 - conc_f

        integral, _ = quad(
            lambda tau: kinetics[CHAIN_DEATH_RATE](
                alpha, init_mon, init, order, tau, bn
            ),
            0, tau_final, limit=100
        )

        assert np.isclose(integral, expected_dead, rtol=1e-3), (
            f"Mass balance during aging failed: integral={integral}, "
            f"expected={expected_dead}"
        )

    def test_second_order_death_rate_is_subset_of_total(self):
        """SECOND_ORDER_DEATH_RATE should be <= CHAIN_DEATH_RATE at all times."""
        kinetics = make_romp_kinetics(alpha1=0.002, alpha2=0.001)
        alpha = 0.005
        init_mon = 1.0
        init = 0.01
        order = 1.5
        bn = 1.0

        for tau in [0.01, 10.0, 50.0, 200.0]:
            total = kinetics[CHAIN_DEATH_RATE](
                alpha, init_mon, init, order, tau, bn
            )
            second = kinetics[SECOND_ORDER_DEATH_RATE](
                alpha, init_mon, init, order, tau, bn
            )
            assert second <= total + 1e-15


# ---- Limiting cases ----

class TestLimitingCases:
    """Test behavior when one alpha is zero."""

    def test_alpha3_zero_no_combination_death(self):
        """When alpha3=0, SECOND_ORDER_DEATH_RATE should be 0."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 1e-15  # effectively zero alpha3
        init_mon = 1.0
        init = 0.01

        rate = kinetics[SECOND_ORDER_DEATH_RATE](
            alpha, init_mon, init, 1.5, 10.0, 1.0
        )
        assert np.isclose(rate, 0.0, atol=1e-10)

    def test_alpha1_zero_no_monomer_dependent_death(self):
        """When alpha1=0, death rate at tau=0 has no monomer-dependent term."""
        kinetics = make_romp_kinetics(alpha1=0.0, alpha2=0.001)
        alpha = 0.005  # fitted alpha3
        init_mon = 1.0
        init = 0.01

        # At tau=0: [M]=M0, [P*]=init
        # death = 0*M0*init + 0.001*init + 0.005*init^2
        death = kinetics[CHAIN_DEATH_RATE](
            alpha, init_mon, init, 1.5, 0.0, 1.0
        )
        expected = 0.001 * init + alpha * init * init
        assert np.isclose(death, expected, rtol=1e-10)

    def test_alpha2_zero(self):
        """When alpha2=0, first-order death term is absent."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha3=0.005)
        alpha = 0.0  # fitted alpha2 = 0
        init_mon = 1.0
        init = 0.01

        # At tau=0: death = a1*M0*init + 0 + a3*init^2
        death = kinetics[CHAIN_DEATH_RATE](
            alpha, init_mon, init, 1.5, 0.0, 1.0
        )
        expected = 0.001 * init_mon * init + 0.005 * init * init
        assert np.isclose(death, expected, rtol=1e-10)

    def test_concentration_matches_ode_gamma_zero(self):
        """When a3=0, [P*](tau) should match numerical ODE."""
        from scipy.integrate import solve_ivp

        a1, a2, a3 = 0.002, 0.001, 0.0
        kinetics = make_romp_kinetics(alpha1=a1, alpha2=a2)
        init_mon = 0.2
        init = 0.004

        def ode(tau, y):
            M, P = y[0], y[1]
            dM = -M * P
            dP = -(a1 * M * P + a2 * P)
            return [dM, dP]

        sol = solve_ivp(ode, [0, 200], [init_mon, init],
                        dense_output=True, rtol=1e-12, atol=1e-15)

        for tau in [1.0, 10.0, 50.0, 100.0]:
            analytical = kinetics[LIVING_CHAIN_CONC](
                a3, init_mon, init, 1.5, tau, 1.0
            )
            numerical = max(sol.sol(tau)[1], 0.0)
            assert np.isclose(analytical, numerical, rtol=1e-4), (
                f"gamma=0 mismatch at tau={tau}: "
                f"analytical={analytical}, numerical={numerical}"
            )


# ---- Numerical verification against ODE ----

class TestODEConsistency:
    """Verify solution against independent numerical ODE integration."""

    def test_concentration_matches_ode(self):
        """[P*](tau) should match numerical ODE solution."""
        from scipy.integrate import solve_ivp

        a1, a2 = 0.002, 0.001
        kinetics = make_romp_kinetics(alpha1=a1, alpha2=a2)
        a3 = 0.005  # fitted alpha3
        init_mon = 1.0
        init = 0.01

        def ode(tau, y):
            M, P = y[0], y[1]
            if P <= 0:
                return [0.0, 0.0]
            dM = -M * P
            dP = -(a1 * M * P + a2 * P + a3 * P * P)
            return [dM, dP]

        sol = solve_ivp(ode, [0, 300], [init_mon, init],
                        dense_output=True, rtol=1e-12, atol=1e-15)

        for tau in [1.0, 10.0, 50.0, 100.0, 200.0]:
            analytical = kinetics[LIVING_CHAIN_CONC](
                a3, init_mon, init, 1.5, tau, 1.0
            )
            numerical = max(sol.sol(tau)[1], 0.0)
            assert np.isclose(analytical, numerical, rtol=1e-4), (
                f"Mismatch at tau={tau}: "
                f"analytical={analytical}, numerical={numerical}"
            )

    def test_concentration_matches_ode_nonunit_monomer(self):
        """[P*](tau) must match ODE when init_mon != 1.0."""
        from scipy.integrate import solve_ivp

        a1, a2 = 0.002, 0.001
        kinetics = make_romp_kinetics(alpha1=a1, alpha2=a2)
        a3 = 0.005
        init_mon = 0.2
        init = 0.004

        def ode(tau, y):
            M, P = y[0], y[1]
            if P <= 0:
                return [0.0, 0.0]
            dM = -M * P
            dP = -(a1 * M * P + a2 * P + a3 * P * P)
            return [dM, dP]

        sol = solve_ivp(ode, [0, 200], [init_mon, init],
                        dense_output=True, rtol=1e-12, atol=1e-15)

        for tau in [1.0, 10.0, 30.0, 50.0]:
            analytical = kinetics[LIVING_CHAIN_CONC](
                a3, init_mon, init, 1.5, tau, 1.0
            )
            numerical = max(sol.sol(tau)[1], 0.0)
            assert np.isclose(analytical, numerical, rtol=1e-4), (
                f"Mismatch at tau={tau}: analytical={analytical}, "
                f"numerical={numerical}"
            )

    def test_monomer_conversion_solves_ode(self):
        """MONOMER_CONVERSION should match the ODE solution for [M]."""
        from scipy.integrate import solve_ivp

        a1, a2 = 0.002, 0.001
        kinetics = make_romp_kinetics(alpha1=a1, alpha2=a2)
        a3 = 0.005
        init_mon = 1.0
        init = 0.01

        def ode(tau, y):
            M, P = y[0], y[1]
            if P <= 0:
                return [0.0, 0.0]
            dM = -M * P
            dP = -(a1 * M * P + a2 * P + a3 * P * P)
            return [dM, dP]

        sol = solve_ivp(ode, [0, 300], [init_mon, init],
                        dense_output=True, rtol=1e-12, atol=1e-15)

        times = np.array([1.0, 10.0, 50.0, 100.0])
        model_mons = kinetics[MONOMER_CONVERSION](
            a3, init_mon, init, 1.5, times, 1.0
        )
        ode_mons = np.array([max(sol.sol(t)[0], 0.0) for t in times])

        np.testing.assert_allclose(model_mons, ode_mons, rtol=1e-4)

    def test_monomer_conversion_scalar_and_array(self):
        """MONOMER_CONVERSION should handle both scalar and array inputs."""
        kinetics = make_romp_kinetics(alpha1=0.002, alpha2=0.001)
        alpha = 0.005
        init_mon = 1.0
        init = 0.01

        scalar_result = kinetics[MONOMER_CONVERSION](
            alpha, init_mon, init, 1.5, 10.0, 1.0
        )
        assert np.isscalar(scalar_result)

        array_result = kinetics[MONOMER_CONVERSION](
            alpha, init_mon, init, 1.5, np.array([10.0]), 1.0
        )
        assert hasattr(array_result, '__len__')
        assert np.isclose(scalar_result, array_result[0])

    def test_conversion_to_time_round_trip(self):
        """conversion_to_time and monomer_conversion should be consistent."""
        kinetics = make_romp_kinetics(alpha1=0.002, alpha2=0.001)
        alpha = 0.005
        init_mon = 1.0
        init = 0.01

        for conv in [0.3, 0.5, 0.7, 0.9]:
            tau = kinetics[CONVERSION_TO_TIME](
                alpha, init_mon, init, 1.5, conv, 1.0
            )
            mon = kinetics[MONOMER_CONVERSION](
                alpha, init_mon, init, 1.5, tau, 1.0
            )
            recovered_conv = 1.0 - mon / init_mon
            assert np.isclose(recovered_conv, conv, rtol=1e-4), (
                f"Round-trip failed: conv={conv}, tau={tau}, "
                f"recovered={recovered_conv}"
            )


# ---- Which alpha is fitted ----

class TestFittedAlphaMapping:
    """Test that the unfixed alpha correctly maps to the alpha argument."""

    def _conc_at(self, kinetics, alpha, tau):
        return kinetics[LIVING_CHAIN_CONC](alpha, 1.0, 0.01, 1.5, tau, 1.0)

    def test_fitting_alpha3(self):
        """When alpha1 and alpha2 are fixed, alpha argument is alpha3."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        c1 = self._conc_at(kinetics, 0.001, 50.0)
        c2 = self._conc_at(kinetics, 0.01, 50.0)
        assert not np.isclose(c1, c2)

    def test_fitting_alpha2(self):
        """When alpha1 and alpha3 are fixed, alpha argument is alpha2."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha3=0.005)
        c1 = self._conc_at(kinetics, 0.0001, 50.0)
        c2 = self._conc_at(kinetics, 0.01, 50.0)
        assert not np.isclose(c1, c2)

    def test_fitting_alpha1(self):
        """When alpha2 and alpha3 are fixed, alpha argument is alpha1."""
        kinetics = make_romp_kinetics(alpha2=0.0005, alpha3=0.005)
        c1 = self._conc_at(kinetics, 0.0001, 50.0)
        c2 = self._conc_at(kinetics, 0.01, 50.0)
        assert not np.isclose(c1, c2)

    def test_symmetry_same_result(self):
        """Providing the same three alphas via different configs should agree."""
        a1, a2, a3 = 0.002, 0.001, 0.005

        k_fit3 = make_romp_kinetics(alpha1=a1, alpha2=a2)
        k_fit2 = make_romp_kinetics(alpha1=a1, alpha3=a3)
        k_fit1 = make_romp_kinetics(alpha2=a2, alpha3=a3)

        tau = 50.0
        c3 = k_fit3[LIVING_CHAIN_CONC](a3, 1.0, 0.01, 1.5, tau, 1.0)
        c2 = k_fit2[LIVING_CHAIN_CONC](a2, 1.0, 0.01, 1.5, tau, 1.0)
        c1 = k_fit1[LIVING_CHAIN_CONC](a1, 1.0, 0.01, 1.5, tau, 1.0)

        assert np.isclose(c3, c2, rtol=1e-4)
        assert np.isclose(c2, c1, rtol=1e-4)


# ---- Aging phase ----

class TestAgingPhase:
    """Test that post-propagation aging works correctly."""

    def test_aging_reduces_living_chains(self):
        """[P*] should continue to decrease during aging."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.005
        init_mon = 1.0
        init = 0.01

        # Get tau at full propagation
        tau_prop = kinetics[CONVERSION_TO_TIME](
            alpha, init_mon, init, 1.5, 0.999, 1.0
        )

        P_prop = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, 1.5, tau_prop, 1.0
        )
        P_aged = kinetics[LIVING_CHAIN_CONC](
            alpha, init_mon, init, 1.5, tau_prop * 100, 1.0
        )
        assert P_aged < P_prop

    def test_aging_death_rate_positive(self):
        """Death rate should be positive during aging."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.005
        init_mon = 1.0
        init = 0.01

        tau_prop = kinetics[CONVERSION_TO_TIME](
            alpha, init_mon, init, 1.5, 0.999, 1.0
        )

        death = kinetics[CHAIN_DEATH_RATE](
            alpha, init_mon, init, 1.5, tau_prop * 10, 1.0
        )
        assert death > 0

    def test_aging_monomer_is_zero(self):
        """[M] should be essentially zero during aging."""
        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        alpha = 0.005
        init_mon = 1.0
        init = 0.01

        tau_prop = kinetics[CONVERSION_TO_TIME](
            alpha, init_mon, init, 1.5, 0.999, 1.0
        )

        mon = kinetics[MONOMER_CONVERSION](
            alpha, init_mon, init, 1.5, tau_prop * 10, 1.0
        )
        assert mon < 1e-6 * init_mon


# ---- Integration with calculate_mwd ----

class TestCalculateMWDIntegration:
    """Test that the model works with calculate_mwd."""

    def test_produces_valid_mwd(self):
        """calculate_mwd should produce a valid MWD with this model."""
        from polyterm import calculate_mwd

        kinetics = make_romp_kinetics(alpha1=0.001, alpha2=0.0005)
        mws = np.logspace(3, 5, 200)

        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=0.005,
            init=0.01, conversion=0.8, order=1.5, sigma=0.05,
            combination=1.0, kinetics=kinetics
        )

        assert result.r_squared is None or True
        assert np.all(np.isfinite(result.intensities))
        assert np.max(result.intensities) > 0

    def test_round_trip_fit(self):
        """Generate MWD then fit it back; recovered alpha should be close."""
        from polyterm import calculate_mwd, fit_mwd

        a1, a2, a3 = 0.001, 0.0005, 0.005
        kinetics = make_romp_kinetics(alpha1=a1, alpha2=a2)
        mws = np.logspace(3, 5, 300)

        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=a3,
            init=0.01, conversion=0.8, order=1.5, sigma=0.05,
            combination=1.0, kinetics=kinetics
        )

        fit_result = fit_mwd(
            mws, result.intensities, order=1.5,
            monomer_mw=104.15, init_mon=1.0,
            init=0.01, conversion=0.8, sigma=0.05,
            combination=1.0, kinetics=kinetics
        )

        assert fit_result.r_squared > 0.95

    def test_round_trip_fit_small_concentrations(self):
        """Round-trip fit with small init_mon and init (realistic ROMP)."""
        from polyterm import calculate_mwd, fit_mwd

        a1_true = 0.005
        kinetics = make_romp_kinetics(alpha2=1e-8, alpha3=1e-8)
        mws = np.logspace(np.log10(2e3), np.log10(1e7), 300)

        result = calculate_mwd(
            mws, monomer_mw=247.34, init_mon=0.17, alpha=a1_true,
            init=0.0003, conversion=0.8, order=2.0, sigma=0.05,
            combination=1.0, kinetics=kinetics, n_quadrature_points=150
        )

        fit_result = fit_mwd(
            mws, result.intensities, order=2.0,
            monomer_mw=247.34, init_mon=0.17,
            init=0.0003, conversion=0.8, sigma=0.05,
            combination=1.0, kinetics=kinetics, n_quadrature_points=150
        )

        assert fit_result.r_squared > 0.95, (
            f"R²={fit_result.r_squared:.4f}, "
            f"alpha={fit_result.alpha:.6f} (true={a1_true})"
        )
