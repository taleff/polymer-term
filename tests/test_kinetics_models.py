"""Tests for kinetics_models module."""

import numpy as np
import pytest

from polyterm.core.kinetics_models import (
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    MONOMER_CONVERSION,
    CHAIN_DEATH_RATE,
    REQUIRED_KEYS,
    STANDARD_KINETICS,
    ROMP_FIRST_ORDER_KINETICS,
    ROMP_SECOND_ORDER_KINETICS,
    validate_kinetics,
    find_chain_death_time,
)

from polyterm.core.kinetics import (
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
)


class TestKeyConstants:
    """Test that key constants are correctly defined."""

    def test_key_constants_are_strings(self):
        """Key constants should be strings."""
        assert isinstance(LIVING_CHAIN_CONC, str)
        assert isinstance(LIVING_CHAIN_DP, str)
        assert isinstance(CONVERSION_TO_TIME, str)
        assert isinstance(MONOMER_CONVERSION, str)

    def test_required_keys_contains_all_constants(self):
        """REQUIRED_KEYS should contain all key constants."""
        assert LIVING_CHAIN_CONC in REQUIRED_KEYS
        assert LIVING_CHAIN_DP in REQUIRED_KEYS
        assert CONVERSION_TO_TIME in REQUIRED_KEYS
        assert MONOMER_CONVERSION in REQUIRED_KEYS
        assert CHAIN_DEATH_RATE in REQUIRED_KEYS
        assert len(REQUIRED_KEYS) == 5


class TestStandardKinetics:
    """Test that STANDARD_KINETICS is correctly configured."""

    def test_has_all_required_keys(self):
        """STANDARD_KINETICS should have all required keys."""
        for key in REQUIRED_KEYS:
            assert key in STANDARD_KINETICS

    def test_all_values_are_callable(self):
        """All values in STANDARD_KINETICS should be callable."""
        for key, func in STANDARD_KINETICS.items():
            assert callable(func), f"{key} is not callable"

    def test_living_chain_conc_unified_signature(self):
        """living_chain_concentration wrapper accepts unified signature."""
        # Unified signature: (alpha, init_mon, init, order, time, bn)
        func = STANDARD_KINETICS[LIVING_CHAIN_CONC]
        result = func(0.001, 1.0, 0.01, 1.0, 0.5, 1.0)

        # Compare to direct call (ignores alpha, init_mon, bn)
        expected = living_chain_concentration(0.01, 1.0, 0.5)
        assert np.isclose(result, expected)

    def test_living_chain_dp_unified_signature(self):
        """living_chain_dp wrapper accepts unified signature."""
        func = STANDARD_KINETICS[LIVING_CHAIN_DP]
        result = func(0.001, 1.0, 0.01, 1.5, 0.5, 1.0)

        # Compare to direct call
        expected = living_chain_dp(0.001, 1.0, 0.01, 1.5, 0.5, 1.0)
        assert np.isclose(result, expected)

    def test_conversion_to_time_unified_signature(self):
        """conversion_to_time wrapper accepts unified signature."""
        func = STANDARD_KINETICS[CONVERSION_TO_TIME]
        # Signature: (alpha, init_mon, init, order, conversion, bn)
        result = func(0.001, 1.0, 0.01, 1.5, 0.8, 1.0)

        # Compare to direct call (ignores init_mon)
        expected = conversion_to_time(0.001, 0.01, 1.5, 0.8, 1.0)
        assert np.isclose(result, expected)

    def test_monomer_conversion_unified_signature(self):
        """monomer_conversion wrapper accepts unified signature."""
        func = STANDARD_KINETICS[MONOMER_CONVERSION]
        # Signature: (alpha, init_mon, init, order, time, bn)
        result = func(0.001, 1.0, 0.01, 1.5, 0.5, 1.0)

        # Should return monomer concentration (fraction of init_mon remaining)
        assert 0 < result <= 1.0  # Should be less than init_mon


class TestValidateKinetics:
    """Test validate_kinetics function."""

    def test_none_returns_standard_kinetics(self):
        """validate_kinetics(None) should return STANDARD_KINETICS."""
        result = validate_kinetics(None)
        assert result is STANDARD_KINETICS

    def test_valid_kinetics_returned_unchanged(self):
        """Valid kinetics dict should be returned unchanged."""
        custom = {
            LIVING_CHAIN_CONC: lambda *args: 0.01,
            LIVING_CHAIN_DP: lambda *args: 100.0,
            CONVERSION_TO_TIME: lambda *args: 0.5,
            MONOMER_CONVERSION: lambda *args: 0.5,
            CHAIN_DEATH_RATE: lambda *args: 0.01,
        }
        result = validate_kinetics(custom)
        assert result is custom

    def test_missing_key_raises_value_error(self):
        """Missing required key should raise ValueError."""
        incomplete = {
            LIVING_CHAIN_CONC: lambda *args: 0.01,
            LIVING_CHAIN_DP: lambda *args: 100.0,
            # Missing CONVERSION_TO_TIME and MONOMER_CONVERSION
        }
        with pytest.raises(ValueError) as excinfo:
            validate_kinetics(incomplete)
        assert "missing required keys" in str(excinfo.value).lower()

    def test_typo_key_gives_helpful_message(self):
        """Typo in key should produce helpful error message."""
        with_typo = {
            'living_chain_conc': lambda *args: 0.01,  # Missing 'entration'
            LIVING_CHAIN_DP: lambda *args: 100.0,
            CONVERSION_TO_TIME: lambda *args: 0.5,
            MONOMER_CONVERSION: lambda *args: 0.5,
            CHAIN_DEATH_RATE: lambda *args: 0.01,
        }
        with pytest.raises(ValueError) as excinfo:
            validate_kinetics(with_typo)
        error_msg = str(excinfo.value).lower()
        assert "missing" in error_msg
        assert "typo" in error_msg or "unexpected" in error_msg


class TestFindChainDeathTime:
    """Test find_chain_death_time function."""

    # For STANDARD_KINETICS, alpha and init_mon are not used in
    # living_chain_concentration, so we use placeholder values
    alpha = 0.001
    init_mon = 1.0

    def test_first_order_kinetics(self):
        """Test chain death time for first-order termination."""
        init = 0.01
        order = 1.0
        death_fraction = 0.9999

        time = find_chain_death_time(
            STANDARD_KINETICS, self.alpha, self.init_mon, init, order, death_fraction
        )

        # Verify: at this time, only (1-death_fraction) of chains are living
        remaining = living_chain_concentration(init, order, time)
        expected_remaining = init * (1 - death_fraction)
        assert np.isclose(remaining, expected_remaining, rtol=1e-6)

    def test_second_order_kinetics(self):
        """Test chain death time for second-order termination."""
        init = 0.01
        order = 2.0
        death_fraction = 0.9999

        time = find_chain_death_time(
            STANDARD_KINETICS, self.alpha, self.init_mon, init, order, death_fraction
        )

        remaining = living_chain_concentration(init, order, time)
        expected_remaining = init * (1 - death_fraction)
        assert np.isclose(remaining, expected_remaining, rtol=1e-6)

    def test_fractional_order_kinetics(self):
        """Test chain death time for fractional-order termination."""
        init = 0.01
        order = 1.5
        death_fraction = 0.9999

        time = find_chain_death_time(
            STANDARD_KINETICS, self.alpha, self.init_mon, init, order, death_fraction
        )

        remaining = living_chain_concentration(init, order, time)
        expected_remaining = init * (1 - death_fraction)
        assert np.isclose(remaining, expected_remaining, rtol=1e-6)

    def test_custom_death_fraction(self):
        """Test with different death fractions."""
        init = 0.01
        order = 1.5

        for death_fraction in [0.5, 0.9, 0.99, 0.999]:
            time = find_chain_death_time(
                STANDARD_KINETICS, self.alpha, self.init_mon, init, order, death_fraction
            )
            remaining = living_chain_concentration(init, order, time)
            expected_remaining = init * (1 - death_fraction)
            assert np.isclose(remaining, expected_remaining, rtol=1e-6)


class TestCustomKineticsIntegration:
    """Test that custom kinetics can override standard behavior."""

    def test_override_single_function(self):
        """Custom kinetics can override a single function."""
        call_count = [0]
        original = STANDARD_KINETICS[LIVING_CHAIN_DP]

        def counting_dp(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        custom = {**STANDARD_KINETICS, LIVING_CHAIN_DP: counting_dp}

        # Validate it
        validated = validate_kinetics(custom)
        assert validated is custom

        # Call the function
        validated[LIVING_CHAIN_DP](0.001, 1.0, 0.01, 1.5, 0.5, 1.0)
        assert call_count[0] == 1

    def test_custom_kinetics_with_different_behavior(self):
        """Custom kinetics can implement different physics."""
        # Simple mock that returns constant values
        custom = {
            LIVING_CHAIN_CONC: lambda *args: 0.005,  # 50% of init=0.01
            LIVING_CHAIN_DP: lambda *args: 200.0,    # Fixed DP
            CONVERSION_TO_TIME: lambda *args: 1.0,   # Fixed time
            MONOMER_CONVERSION: lambda *args: 0.2,   # 80% conversion
            CHAIN_DEATH_RATE: lambda *args: 0.01,    # Constant death rate
        }

        validated = validate_kinetics(custom)

        # Verify custom functions are used
        assert validated[LIVING_CHAIN_CONC](0, 0, 0.01, 0, 0, 0) == 0.005
        assert validated[LIVING_CHAIN_DP](0, 0, 0, 0, 0, 0) == 200.0
        assert validated[CONVERSION_TO_TIME](0, 0, 0, 0, 0, 0) == 1.0
        assert validated[MONOMER_CONVERSION](0, 0, 0, 0, 0, 0) == 0.2
        assert validated[CHAIN_DEATH_RATE](0, 0, 0, 0, 0, 0) == 0.01


class TestKineticsWithFitMwd:
    """Test custom kinetics integration with fit_mwd."""

    def test_custom_kinetics_called_during_fit(self):
        """Verify custom kinetics functions are used during fitting."""
        from polyterm import fit_mwd, calculate_mwd

        # Generate test data using standard kinetics
        mws = np.logspace(3, 5, 200)
        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=0.002,
            init=0.01, conversion=0.8, order=1.5, sigma=0.05
        )

        # Track calls to custom DP function
        call_count = [0]
        original = STANDARD_KINETICS[LIVING_CHAIN_DP]

        def counting_dp(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        custom = {**STANDARD_KINETICS, LIVING_CHAIN_DP: counting_dp}

        # Fit with custom kinetics
        fit_result = fit_mwd(
            mws, result.intensities, order=1.5,
            monomer_mw=104.15, init_mon=1.0,
            sigma=0.05, kinetics=custom
        )

        # Verify custom function was called
        assert call_count[0] > 0
        # Verify fit still works
        assert fit_result.r_squared > 0.9


class TestROMPFirstOrderKinetics:
    """Test ROMP first-order kinetics model."""

    def test_has_all_required_keys(self):
        """ROMP_FIRST_ORDER_KINETICS should have all required keys."""
        for key in REQUIRED_KEYS:
            assert key in ROMP_FIRST_ORDER_KINETICS

    def test_all_values_are_callable(self):
        """All values should be callable."""
        for key, func in ROMP_FIRST_ORDER_KINETICS.items():
            assert callable(func), f"{key} is not callable"

    def test_living_chain_conc_decreases_with_time(self):
        """Living chain concentration should decrease with time."""
        # For first-order ROMP: [P*] = [I]_0 + alpha*[M]_0*(exp(-t/alpha) - 1)
        # This decreases as t increases (exp term goes to 0)
        alpha = 0.001
        init_mon = 1.0
        init = 0.01  # Must have alpha*init_mon < init
        order = 1.0
        bn = 1.0

        func = ROMP_FIRST_ORDER_KINETICS[LIVING_CHAIN_CONC]
        conc_t0 = func(alpha, init_mon, init, order, 0, bn)
        conc_t1 = func(alpha, init_mon, init, order, 0.01, bn)
        conc_t2 = func(alpha, init_mon, init, order, 0.1, bn)

        assert np.isclose(conc_t0, init)  # At t=0, conc = init
        assert conc_t1 < conc_t0
        assert conc_t2 < conc_t1

    def test_living_chain_dp_increases_with_time(self):
        """Living chain DP should increase with time."""
        alpha = 0.001
        init_mon = 1.0
        init = 0.01
        order = 1.0
        bn = 1.0

        func = ROMP_FIRST_ORDER_KINETICS[LIVING_CHAIN_DP]
        dp_t0 = func(alpha, init_mon, init, order, 0, bn)
        dp_t1 = func(alpha, init_mon, init, order, 0.01, bn)
        dp_t2 = func(alpha, init_mon, init, order, 0.1, bn)

        assert np.isclose(dp_t0, 0)  # At t=0, DP = 0
        assert dp_t1 > dp_t0
        assert dp_t2 > dp_t1

    def test_conversion_to_time_inverse_relationship(self):
        """conversion_to_time should be consistent with monomer_conversion."""
        alpha = 0.001
        init_mon = 1.0
        init = 0.01
        order = 1.0
        bn = 1.0

        conv_to_time = ROMP_FIRST_ORDER_KINETICS[CONVERSION_TO_TIME]
        mon_conv = ROMP_FIRST_ORDER_KINETICS[MONOMER_CONVERSION]

        # Get time for 50% conversion
        time = conv_to_time(alpha, init_mon, init, order, 0.5, bn)

        # Verify that at this time, monomer is 50% of initial
        mon_remaining = mon_conv(alpha, init_mon, init, order, time, bn)
        conversion = 1 - mon_remaining / init_mon

        assert np.isclose(conversion, 0.5, rtol=1e-6)

    def test_physical_constraint_alpha_init_mon_less_than_init(self):
        """For first-order ROMP, alpha*init_mon must be < init for valid kinetics."""
        # When alpha*init_mon >= init, living chain conc goes negative
        alpha = 0.02  # alpha*init_mon = 0.02 > init=0.01
        init_mon = 1.0
        init = 0.01
        order = 1.0
        bn = 1.0

        func = ROMP_FIRST_ORDER_KINETICS[LIVING_CHAIN_CONC]

        # At long times, exp(-t/alpha) -> 0
        # So conc -> init + alpha*init_mon*(0 - 1) = init - alpha*init_mon
        # This becomes negative when alpha*init_mon > init
        conc_long_time = func(alpha, init_mon, init, order, 100, bn)

        # The concentration becomes negative (unphysical)
        assert conc_long_time < 0


class TestROMPSecondOrderKinetics:
    """Test ROMP second-order kinetics model."""

    def test_has_all_required_keys(self):
        """ROMP_SECOND_ORDER_KINETICS should have all required keys."""
        for key in REQUIRED_KEYS:
            assert key in ROMP_SECOND_ORDER_KINETICS

    def test_all_values_are_callable(self):
        """All values should be callable."""
        for key, func in ROMP_SECOND_ORDER_KINETICS.items():
            assert callable(func), f"{key} is not callable"

    def test_living_chain_conc_decreases_with_time(self):
        """Living chain concentration should decrease with time."""
        alpha = 0.0001
        init_mon = 1.0
        init = 0.01
        order = 1.0
        bn = 1.0

        func = ROMP_SECOND_ORDER_KINETICS[LIVING_CHAIN_CONC]
        conc_t0 = func(alpha, init_mon, init, order, 0, bn)
        conc_t1 = func(alpha, init_mon, init, order, 0.01, bn)
        conc_t2 = func(alpha, init_mon, init, order, 0.1, bn)

        # At t=0, should be close to init
        assert np.isclose(conc_t0, init, rtol=1e-3)
        assert conc_t1 < conc_t0
        assert conc_t2 < conc_t1

    def test_living_chain_dp_increases_with_time(self):
        """Living chain DP should increase with time."""
        alpha = 0.0001
        init_mon = 1.0
        init = 0.01
        order = 1.0
        bn = 1.0

        func = ROMP_SECOND_ORDER_KINETICS[LIVING_CHAIN_DP]
        dp_t0 = func(alpha, init_mon, init, order, 0, bn)
        dp_t1 = func(alpha, init_mon, init, order, 0.01, bn)
        dp_t2 = func(alpha, init_mon, init, order, 0.1, bn)

        # At t=0, DP should be 0 (or close to it)
        assert np.isclose(dp_t0, 0, atol=1e-6)
        assert dp_t1 > dp_t0
        assert dp_t2 > dp_t1


class TestROMPKineticsWithFitMwd:
    """Test ROMP kinetics integration with fit_mwd."""

    def test_romp_first_order_fit(self):
        """Fit with ROMP first-order kinetics should work."""
        from polyterm import fit_mwd, calculate_mwd

        # Generate test data using ROMP first-order kinetics
        # Important: alpha*init_mon < init for valid kinetics
        # Use parameters that produce significant termination for a good fit
        mws = np.logspace(3, 5, 200)
        alpha = 0.005  # alpha*init_mon = 0.005 < init=0.01
        init_mon = 1.0
        init = 0.01
        conversion = 0.9

        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=init_mon, alpha=alpha,
            init=init, conversion=conversion, order=1.0, sigma=0.05,
            kinetics=ROMP_FIRST_ORDER_KINETICS
        )

        # Fit the generated data with fixed init and conversion
        fit_result = fit_mwd(
            mws, result.intensities, order=1.0,
            monomer_mw=104.15, init_mon=init_mon,
            init=init, conversion=conversion, sigma=0.05,
            kinetics=ROMP_FIRST_ORDER_KINETICS
        )

        # Verify fit quality (MWD fitting with ROMP kinetics is challenging,
        # so we use a lower threshold than for standard kinetics)
        assert fit_result.r_squared > 0.85

    def test_romp_second_order_fit(self):
        """Fit with ROMP second-order kinetics should work."""
        from polyterm import fit_mwd, calculate_mwd

        # Generate test data using ROMP second-order kinetics
        # Important: alpha*init_mon != init (singularity)
        # Use alpha << init/init_mon to stay far from singularity
        mws = np.logspace(3, 5, 200)
        alpha = 0.0005  # alpha*init_mon = 0.0005 << init=0.01
        init_mon = 1.0
        init = 0.01
        conversion = 0.8

        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=init_mon, alpha=alpha,
            init=init, conversion=conversion, order=1.0, sigma=0.05,
            kinetics=ROMP_SECOND_ORDER_KINETICS
        )

        # Fit the generated data with fixed init and conversion
        fit_result = fit_mwd(
            mws, result.intensities, order=1.0,
            monomer_mw=104.15, init_mon=init_mon,
            init=init, conversion=conversion, sigma=0.05,
            kinetics=ROMP_SECOND_ORDER_KINETICS
        )

        # Verify fit quality
        assert fit_result.r_squared > 0.95
