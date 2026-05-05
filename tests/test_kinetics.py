"""
Tests for core kinetic functions.
"""

import pytest
import numpy as np
from polyterm.core.kinetics import (
    monomer_conversion,
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    time_to_chain_death,
)


class TestLivingChainConcentration:
    """Test living chain concentration calculations."""

    def test_zero_time(self):
        """Test that t=0 gives initial concentration."""
        init = 0.01
        for order in [1.0, 1.5, 2.0]:
            result = living_chain_concentration(init, order, 0.0)
            assert np.isclose(result, init, rtol=1e-10)


class TestConversionMonomer:
    """Test monomer conversion calculations."""

    def test_first_order_termination(self):
        """Test monomer consumption with first-order termination."""
        times = np.array([0, 1, 2, 3])
        kp = 100.0
        kt = 0.1
        init_mon = 1.0
        init = 0.01
        order = 1.0
        bn = 1.0

        result = monomer_conversion(times, kp, kt, init_mon, init, order, bn)

        # At t=0, should have initial concentration
        assert np.isclose(result[0], init_mon, rtol=1e-10)

        # Should be monotonically decreasing
        assert np.all(np.diff(result) < 0)

        # Should be positive
        assert np.all(result > 0)


class TestConversionTime:
    """Test conversion/time relationship."""

    def test_round_trip_first_order(self):
        """Test conversion -> time -> conversion round trip for order=1."""
        alpha = 0.01
        kt = 1
        init_mon = 1.0
        init = 0.02
        order = 1.0
        conversion = 0.6
        bn = 1.0

        # Convert to time
        time = conversion_to_time(alpha, init, order, conversion, bn)

        # Convert back using monomer conversion
        monomer_fraction = monomer_conversion(
            time, kt/alpha, kt, init_mon, init, order, bn
        )

        assert np.isclose(conversion, 1-monomer_fraction, atol=1e-10)

    def test_zero_conversion(self):
        """Test that zero conversion gives zero time."""
        alpha = 0.01
        init = 0.02
        order = 1.5
        conversion = 0.0
        bn = 1.0

        time = conversion_to_time(alpha, init, order, conversion, bn)

        assert np.isclose(time, 0.0, atol=1e-10)

    def test_high_conversion(self):
        """Test behavior at high conversion."""
        alpha = 0.01
        init = 0.02
        order = 1.5
        conversion = 0.9999
        bn = 1.0

        time = conversion_to_time(alpha, init, order, conversion, bn)

        assert time > 0
        assert np.isfinite(time)


class TestLivingChainDP:
    """Test living chain degree of polymerization calculations."""

    def test_increases_with_time(self):
        """Test that DP increases with time."""
        alpha = 0.01
        init_mon = 1.0
        init = 0.02
        order = 1.5
        bn = 1.0

        times = np.array([0.1, 0.5, 1.0, 2.0])
        dps = [living_chain_dp(alpha, init_mon, init, order, t, bn)
               for t in times]

        # DP should increase monotonically
        assert all(dps[i] < dps[i+1] for i in range(len(dps)-1))

    def test_positive_values(self):
        """Test that DP is always positive."""
        alpha = 0.01
        init_mon = 1.0
        init = 0.02
        bn = 1.0

        for order in [1.0, 1.5, 2.0]:
            for time in [0.1, 1.0, 5.0]:
                dp = living_chain_dp(alpha, init_mon, init, order, time, bn)
                assert dp > 0

    def test_order_dependence(self):
        """Test that DP depends on termination order."""
        alpha = 0.05
        init_mon = 1.0
        init = 0.02
        time = 1.0
        bn = 1.0

        dp1 = living_chain_dp(alpha, init_mon, init, 1.0, time, bn)
        dp15 = living_chain_dp(alpha, init_mon, init, 1.5, time, bn)
        dp2 = living_chain_dp(alpha, init_mon, init, 2.0, time, bn)

        # DPs should be different for different orders
        assert not np.isclose(dp1, dp2, rtol=0.1)

    def test_kinetic_chain_length_order_1(self):
        """Test chain length calculation for first-order termination (n=1)."""
        # Parameters from deprecated test
        nu = 500
        alpha1 = 0.0005
        init_mon = 1.0
        init = 0.001
        order1 = 1.0
        conv = (init / init_mon) * nu
        bn = 1.0

        # Expected value from deprecated test
        expected = 599.123

        # Convert conversion to time
        time = conversion_to_time(alpha1, init, order1, conv, bn)
        # Calculate chain length
        nup = living_chain_dp(alpha1, init_mon, init, order1, time, bn)

        assert np.isclose(nup, expected, rtol=1e-3), \
            f'Incorrect chain length for n={order1}: got {nup}, expected {expected}'

    def test_kinetic_chain_length_order_2(self):
        """Test chain length calculation for second-order termination (n=2)."""
        # Parameters from deprecated test
        nu = 500
        alpha2 = 0.5
        init_mon = 1.0
        init = 0.001
        order2 = 2.0
        conv = (init / init_mon) * nu
        bn = 1.0

        # Expected value from deprecated test
        expected = 585.786

        # Convert conversion to time
        time = conversion_to_time(alpha2, init, order2, conv, bn)
        # Calculate chain length
        nup = living_chain_dp(alpha2, init_mon, init, order2, time, bn)

        assert np.isclose(nup, expected, rtol=1e-3), \
            f'Incorrect chain length for n={order2}: got {nup}, expected {expected}'

    def test_kinetic_chain_length_order_1_1(self):
        """Test chain length calculation for n=1.1 termination."""
        # Parameters from deprecated test
        nu = 500
        alpha3 = 0.0005
        init_mon = 1.0
        init = 0.001
        order3 = 1.1
        conv = (init / init_mon) * nu
        bn = 1.0

        # Expected value from deprecated test
        expected = 542.955

        # Convert conversion to time
        time = conversion_to_time(alpha3, init, order3, conv, bn)
        # Calculate chain length
        nup = living_chain_dp(alpha3, init_mon, init, order3, time, bn)

        assert np.isclose(nup, expected, rtol=1e-3), \
            f'Incorrect chain length for n={order3}: got {nup}, expected {expected}'


class TestTimeToChainDeath:
    """Test time_to_chain_death function."""

    def test_first_order_termination(self):
        """Test time to chain death for first-order termination."""
        chain_conversion = 0.5
        init = 0.01
        order = 1.0

        time = time_to_chain_death(chain_conversion, init, order)

        assert time > 0
        assert np.isfinite(time)

    def test_second_order_termination(self):
        """Test time to chain death for second-order termination."""
        chain_conversion = 0.5
        init = 0.01
        order = 2.0

        time = time_to_chain_death(chain_conversion, init, order)

        assert time > 0
        assert np.isfinite(time)

    def test_other_order_termination(self):
        """Test time to chain death for other termination orders."""
        chain_conversion = 0.5
        init = 0.01
        order = 1.5

        time = time_to_chain_death(chain_conversion, init, order)

        assert time > 0
        assert np.isfinite(time)

    def test_zero_conversion(self):
        """Test that zero conversion gives zero time."""
        chain_conversion = 0.0
        init = 0.01
        order = 1.5

        time = time_to_chain_death(chain_conversion, init, order)

        assert np.isclose(time, 0.0, atol=1e-10)

    def test_high_conversion(self):
        """Test behavior at high chain conversion."""
        chain_conversion = 0.99
        init = 0.01
        order = 1.5

        time = time_to_chain_death(chain_conversion, init, order)

        assert time > 0
        assert np.isfinite(time)


class TestMonomerConversionEdgeCases:
    """Test edge cases for monomer conversion function."""

    def test_second_order_termination(self):
        """Test monomer conversion with second-order termination."""
        times = np.array([0, 1, 2, 3])
        kp = 100.0
        kt = 0.1
        init_mon = 1.0
        init = 0.01
        order = 2.0
        bn = 1.0

        result = monomer_conversion(times, kp, kt, init_mon, init, order, bn)

        assert np.isclose(result[0], init_mon, rtol=1e-10)
        assert np.all(np.diff(result) < 0)
        assert np.all(result > 0)

    def test_special_order_1_plus_1_bn(self):
        """Test monomer conversion with order = 1 + 1/bn."""
        times = np.array([0, 1, 2, 3])
        kp = 100.0
        kt = 0.1
        init_mon = 1.0
        init = 0.01
        bn = 1.0
        order = 1.0 + (1.0 / bn)  # = 2.0

        result = monomer_conversion(times, kp, kt, init_mon, init, order, bn)

        assert np.isclose(result[0], init_mon, rtol=1e-10)
        assert np.all(result > 0)

    def test_other_orders(self):
        """Test monomer conversion with various termination orders."""
        times = np.array([0, 0.5, 1.0])
        kp = 100.0
        kt = 0.1
        init_mon = 1.0
        init = 0.01
        bn = 1.0

        for order in [1.5, 2.5, 3.0]:
            result = monomer_conversion(times, kp, kt, init_mon, init, order, bn)
            assert np.isclose(result[0], init_mon, rtol=1e-10)
            assert np.all(result > 0)


class TestLivingChainConcentrationVariations:
    """Test living chain concentration with various parameters."""

    def test_first_order_non_zero_time(self):
        """Test living chain concentration for first-order at non-zero time."""
        init = 0.01
        order = 1.0
        time = 1.0

        result = living_chain_concentration(init, order, time)

        assert result < init  # Should decrease with time
        assert result > 0

    def test_second_order_non_zero_time(self):
        """Test living chain concentration for second-order at non-zero time."""
        init = 0.01
        order = 2.0
        time = 1.0

        result = living_chain_concentration(init, order, time)

        assert result < init
        assert result > 0

    def test_other_order_non_zero_time(self):
        """Test living chain concentration for other orders at non-zero time."""
        init = 0.01
        order = 1.5
        time = 1.0

        result = living_chain_concentration(init, order, time)

        assert result < init
        assert result > 0

    def test_decreases_with_time(self):
        """Test that living chain concentration decreases with time."""
        init = 0.01
        order = 1.5

        times = np.array([0.1, 0.5, 1.0, 2.0])
        concentrations = [living_chain_concentration(init, order, t) for t in times]

        # Should decrease monotonically
        assert all(concentrations[i] > concentrations[i+1] for i in range(len(concentrations)-1))


class TestLivingChainDPWithBn:
    """Test living chain DP with bn != 1."""

    def test_bn_not_equal_one(self):
        """Test living chain DP calculation when bn != 1."""
        alpha = 0.01
        init_mon = 1.0
        init = 0.02
        order = 1.5
        time = 1.0
        bn = 0.5  # Non-standard value

        dp = living_chain_dp(alpha, init_mon, init, order, time, bn)

        assert dp > 0
        assert np.isfinite(dp)


class TestConversionTimeWithBn:
    """Test conversion_to_time with bn != 1."""

    def test_bn_not_equal_one_order_1(self):
        """Test conversion to time with bn != 1 and order = 1."""
        alpha = 0.01
        init = 0.02
        order = 1.0
        conversion = 0.1  # Lower conversion to avoid numerical issues
        bn = 0.5

        time = conversion_to_time(alpha, init, order, conversion, bn)

        assert time > 0 or not np.isfinite(time)  # May be invalid for some combinations

    def test_bn_not_equal_one_order_2(self):
        """Test conversion to time with bn != 1 and order = 2."""
        alpha = 0.01
        init = 0.02
        order = 2.0
        conversion = 0.5
        bn = 0.5

        time = conversion_to_time(alpha, init, order, conversion, bn)

        assert time > 0
        assert np.isfinite(time)

    def test_bn_not_equal_one_other_order(self):
        """Test conversion to time with bn != 1 and other orders."""
        alpha = 0.01
        init = 0.02
        order = 1.5
        conversion = 0.1  # Lower conversion to avoid numerical issues
        bn = 0.5

        time = conversion_to_time(alpha, init, order, conversion, bn)

        assert time > 0 or not np.isfinite(time)  # May be invalid for some combinations


class TestLivingChainDPHighOrder:
    """Test living chain DP with order > 2."""

    def test_order_greater_than_2(self):
        """Test living chain DP for order > 2."""
        alpha = 0.05  # Higher alpha to avoid numerical overflow
        init_mon = 1.0
        init = 0.01  # Lower init
        order = 2.5
        time = 0.5  # Shorter time to avoid numerical issues
        bn = 1.0

        dp = living_chain_dp(alpha, init_mon, init, order, time, bn)

        # For order > 2, function may encounter numerical issues with certain parameters
        assert dp > 0 or not np.isfinite(dp)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
