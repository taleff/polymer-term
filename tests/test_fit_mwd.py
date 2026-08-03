"""
Tests for the fit_mwd functional API.
"""

import pytest
import numpy as np
from functools import partial

from polyterm import fit_mwd, MWDResult, calculate_mwd
from polyterm.core.mwd_computation import (
    get_quadrature_points,
    compute_dead_chain_fracs,
)
from polyterm.kinetics.models import STANDARD_KINETICS


class TestQuadratureHelper:
    """Test the quadrature helper function."""

    def test_returns_correct_shapes(self):
        """Test that quadrature returns correct array shapes."""
        n_points = 50
        time_end = 10.0
        nodes, weights = get_quadrature_points(n_points, time_end)

        assert nodes.shape == (n_points,)
        assert weights.shape == (n_points,)

    def test_nodes_in_correct_range(self):
        """Test that nodes are scaled to [0, time_end]."""
        n_points = 50
        time_end = 10.0
        nodes, weights = get_quadrature_points(n_points, time_end)

        assert np.all(nodes >= 0)
        assert np.all(nodes <= time_end)

    def test_weights_sum_to_interval(self):
        """Test that weights sum to the interval length."""
        n_points = 50
        time_end = 10.0
        nodes, weights = get_quadrature_points(n_points, time_end)

        assert np.isclose(np.sum(weights), time_end, rtol=1e-10)

    def test_integrates_polynomial_exactly(self):
        """Test that quadrature integrates low-degree polynomials exactly."""
        n_points = 10
        time_end = 5.0
        nodes, weights = get_quadrature_points(n_points, time_end)

        f_vals = nodes ** 2
        integral = np.sum(weights * f_vals)
        expected = (time_end ** 3) / 3

        assert np.isclose(integral, expected, rtol=1e-10)


class TestDeadChainFracs:
    """Test the dead chain fraction computation."""

    def test_returns_correct_shape(self):
        """Test that dead fractions have correct shape."""
        time = 5.0
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5

        dead_fracs = compute_dead_chain_fracs(
            time, dps, alpha, init_mon, init, order,
            bn=1.0, combination=0.0, n_quadrature_points=40,
            kinetics=STANDARD_KINETICS
        )

        assert dead_fracs.shape == (len(dps),)

    def test_values_are_nonnegative(self):
        """Test that dead fractions are non-negative."""
        time = 5.0
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5

        dead_fracs = compute_dead_chain_fracs(
            time, dps, alpha, init_mon, init, order,
            bn=1.0, combination=0.0, n_quadrature_points=40,
            kinetics=STANDARD_KINETICS
        )

        assert np.all(dead_fracs >= 0)

    def test_sum_is_reasonable(self):
        """Test that total dead fraction is between 0 and init."""
        time = 5.0
        dps = np.arange(1, 500)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5

        dead_fracs = compute_dead_chain_fracs(
            time, dps, alpha, init_mon, init, order,
            bn=1.0, combination=0.0, n_quadrature_points=40,
            kinetics=STANDARD_KINETICS
        )

        total_dead = np.sum(dead_fracs)
        assert 0 < total_dead < init

    def test_quadrature_points_parameter(self):
        """Test that n_quadrature_points parameter affects accuracy."""
        time = 5.0
        dps = np.arange(1, 300)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5

        # Lower quadrature points
        dead_fracs_low = compute_dead_chain_fracs(
            time, dps, alpha, init_mon, init, order,
            bn=1.0, combination=0.0, n_quadrature_points=20,
            kinetics=STANDARD_KINETICS
        )

        # Higher quadrature points
        dead_fracs_high = compute_dead_chain_fracs(
            time, dps, alpha, init_mon, init, order,
            bn=1.0, combination=0.0, n_quadrature_points=100,
            kinetics=STANDARD_KINETICS
        )

        # Both should be similar in total
        assert np.isclose(
            np.sum(dead_fracs_low), np.sum(dead_fracs_high), rtol=0.05
        )


class TestFitMwdValidation:
    """Test input validation for fit_mwd."""

    def test_tau_requires_sigma(self, simple_mws):
        """Test that tau requires sigma to be specified."""
        intensities = np.ones_like(simple_mws)

        with pytest.raises(ValueError, match="tau requires sigma"):
            fit_mwd(
                simple_mws, intensities,
                order=1.5,
                monomer_mw=100.0,
                init_mon=1.0,
                tau=0.05  # tau without sigma
            )

    def test_order_must_be_positive(self, simple_mws):
        """Test that order must be positive."""
        intensities = np.ones_like(simple_mws)

        with pytest.raises(ValueError, match="order must be positive"):
            fit_mwd(
                simple_mws, intensities,
                order=-1.0,
                monomer_mw=100.0,
                init_mon=1.0
            )

    def test_conversion_bounds(self, simple_mws):
        """Test that conversion must be between 0 and 1."""
        intensities = np.ones_like(simple_mws)

        with pytest.raises(ValueError, match="conversion must be between"):
            fit_mwd(
                simple_mws, intensities,
                order=1.5,
                monomer_mw=100.0,
                init_mon=1.0,
                conversion=1.5
            )

    def test_init_must_be_positive(self, simple_mws):
        """Test that init must be positive if specified."""
        intensities = np.ones_like(simple_mws)

        with pytest.raises(ValueError, match="init must be positive"):
            fit_mwd(
                simple_mws, intensities,
                order=1.5,
                monomer_mw=100.0,
                init_mon=1.0,
                init=-0.01
            )

    def test_sigma_must_be_positive(self, simple_mws):
        """Test that sigma must be positive if specified."""
        intensities = np.ones_like(simple_mws)

        with pytest.raises(ValueError, match="sigma must be positive"):
            fit_mwd(
                simple_mws, intensities,
                order=1.5,
                monomer_mw=100.0,
                init_mon=1.0,
                sigma=-0.05
            )


class TestFitMwdBasic:
    """Basic functionality tests for fit_mwd."""

    def test_fit_returns_mwd_result(self, simple_mws, standard_params):
        """Test that fit_mwd returns an MWDResult object."""
        # Generate synthetic data
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            init=standard_params['init'],
            sigma=standard_params['sigma'],
        )

        assert isinstance(result, MWDResult)

    def test_fit_result_has_all_attributes(self, simple_mws, standard_params):
        """Test that MWDResult has all expected attributes."""
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            init=standard_params['init'],
            sigma=standard_params['sigma'],
        )

        # Check all required attributes
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'init')
        assert hasattr(result, 'order')
        assert hasattr(result, 'sigma')
        assert hasattr(result, 'tau')
        assert hasattr(result, 'conversion')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'molecular_weights')
        assert hasattr(result, 'intensities')
        assert hasattr(result, 'dead_chain_intensities')
        assert hasattr(result, 'dead_chain_fraction')


class TestRoundTripFitting:
    """Test fitting synthetic data recovers known parameters."""

    def test_fit_with_sigma_estimated(self, simple_mws, standard_params):
        """Test fitting with sigma=None (estimates sigma).

        Note: init must be provided because without it, multiple (alpha, init,
        conversion) combinations can produce similar MWD shapes, making the
        problem under-constrained.
        """
        true_alpha = standard_params['alpha']

        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            init=standard_params['init']  # Required for identifiability
        )

        # Should recover parameters within reasonable tolerance
        assert np.isclose(result.alpha, true_alpha, rtol=0.15)
        assert result.r_squared > 0.99

    def test_fit_with_fixed_sigma(self, simple_mws, standard_params):
        """Test fitting with fixed sigma (Gaussian broadening)."""
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma']  # Fixed sigma
        )

        # Should get good fit quality
        assert result.r_squared > 0.99
        assert result.sigma == standard_params['sigma']
        assert result.tau == 0.0

    def test_fit_with_fixed_sigma_and_tau(self, simple_mws, standard_params):
        """Test fitting with fixed sigma and tau (EGH broadening)."""
        # Generate data with EGH broadening
        tau = 0.03
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma'],
            tau=tau
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            tau=tau
        )

        assert result.r_squared > 0.99
        assert result.sigma == standard_params['sigma']
        assert result.tau == tau

    def test_fit_with_known_init(self, simple_mws, standard_params):
        """Test fitting with known initiator concentration."""
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            init=standard_params['init']  # Fixed init
        )

        assert np.isclose(result.conversion, standard_params['conversion'], rtol=0.05)
        assert result.r_squared > 0.99

    def test_fit_with_known_conversion(self, simple_mws, standard_params):
        """Test fitting with known conversion."""
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            conversion=standard_params['conversion']  # Fixed conversion
        )

        # Looser tolerance - fitting with only conversion known is under-constrained
        assert np.isclose(result.init, standard_params['init'], rtol=0.15)
        assert result.r_squared > 0.95


class TestDifferentOrders:
    """Test fitting with different termination orders."""

    def test_first_order(self, simple_mws, first_order_params):
        """Test fitting first order termination."""
        mwd_result = calculate_mwd(
            simple_mws,
            first_order_params['monomer_mw'],
            first_order_params['init_mon'],
            first_order_params['alpha'],
            first_order_params['init'],
            first_order_params['conversion'],
            first_order_params['order'],
            first_order_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=first_order_params['order'],
            monomer_mw=first_order_params['monomer_mw'],
            init_mon=first_order_params['init_mon'],
            sigma=first_order_params['sigma']
        )

        assert result.r_squared > 0.95
        assert result.order == 1.0

    def test_second_order(self, simple_mws, second_order_params):
        """Test fitting second order termination."""
        mwd_result = calculate_mwd(
            simple_mws,
            second_order_params['monomer_mw'],
            second_order_params['init_mon'],
            second_order_params['alpha'],
            second_order_params['init'],
            second_order_params['conversion'],
            second_order_params['order'],
            second_order_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=second_order_params['order'],
            monomer_mw=second_order_params['monomer_mw'],
            init_mon=second_order_params['init_mon'],
            sigma=second_order_params['sigma']
        )

        assert result.r_squared > 0.95
        assert result.order == 2.0

    def test_fractional_order(self, simple_mws, other_order_params):
        """Test fitting fractional order termination."""
        mwd_result = calculate_mwd(
            simple_mws,
            other_order_params['monomer_mw'],
            other_order_params['init_mon'],
            other_order_params['alpha'],
            other_order_params['init'],
            other_order_params['conversion'],
            other_order_params['order'],
            other_order_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=other_order_params['order'],
            monomer_mw=other_order_params['monomer_mw'],
            init_mon=other_order_params['init_mon'],
            sigma=other_order_params['sigma']
        )

        assert result.r_squared > 0.95
        assert result.order == other_order_params['order']


class TestBatchProcessing:
    """Test batch processing with functools.partial."""

    def test_partial_application(self, simple_mws, standard_params):
        """Test that functools.partial works for batch processing."""
        # Create partially applied function
        fit_my_instrument = partial(
            fit_mwd,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma']
        )

        # Generate multiple synthetic datasets
        mwd_result1 = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        mwd_result2 = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'] * 0.8,  # Different conversion
            standard_params['order'],
            standard_params['sigma']
        )

        # Fit both using partial
        result1 = fit_my_instrument(simple_mws, mwd_result1.intensities)
        result2 = fit_my_instrument(simple_mws, mwd_result2.intensities)

        assert result1.r_squared > 0.99
        assert result2.r_squared > 0.99


class TestDownsampling:
    """Test that downsampling works correctly."""

    def test_fit_with_downsampling(self, simple_mws, standard_params):
        """Test that fitting works with different max_fit_points."""
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result1 = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            max_fit_points=200
        )

        result2 = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            max_fit_points=500
        )

        # Both should give reasonable fits
        assert result1.r_squared > 0.90
        assert result2.r_squared > 0.90


class TestHighDPFitting:
    """Test fitting with high degree of polymerization samples."""

    def test_fit_high_dp_sample(self):
        """Test that fit_mwd works correctly for high DP."""
        mws = np.logspace(3, 6, 500)  # 1k to 1M
        params = {
            "monomer_mw": 100.0,
            "conversion": 0.5,
            "alpha": 0.0005,
            "init_mon": 1.0,
            "init": 0.001,
            "order": 1.0,
            "sigma": 0.12
        }

        mwd_result = calculate_mwd(
            mws,
            params['monomer_mw'],
            params['init_mon'],
            params['alpha'],
            params['init'],
            params['conversion'],
            params['order'],
            params['sigma']
        )

        result = fit_mwd(
            mws, mwd_result.intensities,
            order=params['order'],
            monomer_mw=params['monomer_mw'],
            init_mon=params['init_mon'],
            sigma=params['sigma'],
            init=params['init']  # Provide init for identifiability at high DP
        )

        # Should achieve good fit
        assert result.r_squared > 0.95
        # Should recover alpha within 40% (high DP is more numerically challenging)
        assert np.isclose(result.alpha, params['alpha'], rtol=0.4)

    def test_quadrature_points_parameter(self):
        """Test that n_quadrature_points parameter works."""
        mws = np.logspace(3, 5, 200)
        params = {
            "monomer_mw": 100.0,
            "conversion": 0.5,
            "alpha": 0.002,
            "init_mon": 1.0,
            "init": 0.005,
            "order": 1.0,  # First order is more stable for quadrature comparison
            "sigma": 0.12
        }

        mwd_result = calculate_mwd(
            mws,
            params['monomer_mw'],
            params['init_mon'],
            params['alpha'],
            params['init'],
            params['conversion'],
            params['order'],
            params['sigma']
        )

        # Fit with different quadrature points (provide init for identifiability)
        result_50 = fit_mwd(
            mws, mwd_result.intensities,
            order=params['order'],
            monomer_mw=params['monomer_mw'],
            init_mon=params['init_mon'],
            init=params['init'],
            sigma=params['sigma'],
            n_quadrature_points=50
        )

        result_150 = fit_mwd(
            mws, mwd_result.intensities,
            order=params['order'],
            monomer_mw=params['monomer_mw'],
            init_mon=params['init_mon'],
            init=params['init'],
            sigma=params['sigma'],
            n_quadrature_points=150
        )

        # Both should give good fits
        assert result_50.r_squared > 0.95
        assert result_150.r_squared > 0.95
        # Results should be similar (both converge to same solution)
        assert np.isclose(result_50.alpha, result_150.alpha, rtol=0.15)


class TestDistributionParameter:
    """Test the distribution parameter for custom chain length distributions."""

    def test_fit_mwd_accepts_distribution(self, simple_mws, standard_params):
        """Test that fit_mwd accepts a distribution parameter."""
        from scipy.stats import poisson

        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            distribution=poisson.pmf,
        )

        assert result.r_squared > 0.99

    def test_fit_mwd_with_custom_distribution_round_trip(
        self, simple_mws, standard_params
    ):
        """Test round-trip: generate with custom dist, fit with same dist."""
        def gaussian_dist(dps, nup):
            sigma = np.sqrt(nup)
            return np.exp(-0.5 * ((dps - nup) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )

        # Generate data with custom distribution
        mwd_result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma'],
            distribution=gaussian_dist,
        )

        # Fit with same custom distribution
        result = fit_mwd(
            simple_mws, mwd_result.intensities,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            distribution=gaussian_dist,
        )

        assert result.r_squared > 0.99


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
