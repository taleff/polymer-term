"""
Tests for the fit_mwd functional API.
"""

import pytest
import numpy as np
from functools import partial

from scipy.integrate import quad_vec

from polyterm import fit_mwd, FitResult
from polyterm.core.distributions import calculate_mwd, living_distribution_integrand
from polyterm.models.fitting import (
    _get_quadrature_points,
    _precompute_poisson_matrix,
    _compute_dead_fracs_quadrature,
)


class TestQuadratureHelper:
    """Test the quadrature helper function."""

    def test_returns_correct_shapes(self):
        """Test that quadrature returns correct array shapes."""
        n_points = 50
        time_end = 10.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        assert nodes.shape == (n_points,)
        assert weights.shape == (n_points,)

    def test_nodes_in_correct_range(self):
        """Test that nodes are scaled to [0, time_end]."""
        n_points = 50
        time_end = 10.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        assert np.all(nodes >= 0)
        assert np.all(nodes <= time_end)

    def test_weights_sum_to_interval(self):
        """Test that weights sum to the interval length."""
        n_points = 50
        time_end = 10.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        assert np.isclose(np.sum(weights), time_end, rtol=1e-10)

    def test_integrates_polynomial_exactly(self):
        """Test that quadrature integrates low-degree polynomials exactly."""
        n_points = 10
        time_end = 5.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        f_vals = nodes ** 2
        integral = np.sum(weights * f_vals)
        expected = (time_end ** 3) / 3

        assert np.isclose(integral, expected, rtol=1e-10)


class TestPoissonPrecomputation:
    """Test the Poisson precomputation function."""

    def test_returns_correct_shape(self):
        """Test that precomputed matrix has correct shape."""
        times = np.array([0.1, 0.5, 1.0, 2.0])
        dps = np.arange(1, 100)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0

        matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        assert matrix.shape == (len(times), len(dps))

    def test_rows_sum_approximately_to_one(self):
        """Test that each row sums to approximately 1 (Poisson normalization)."""
        times = np.array([0.1, 0.5, 1.0])
        dps = np.arange(1, 500)  # Wide range to capture most probability mass
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0

        matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        row_sums = np.sum(matrix, axis=1)
        assert np.all(row_sums > 0.99)  # Allow some truncation

    def test_values_are_nonnegative(self):
        """Test that all Poisson probabilities are non-negative."""
        times = np.array([0.1, 1.0, 5.0])
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0

        matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        assert np.all(matrix >= 0)


class TestDeadFracsQuadrature:
    """Test the fixed quadrature dead fraction computation."""

    def test_returns_correct_shape(self):
        """Test that dead fractions have correct shape."""
        n_points = 50
        time_end = 5.0
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        combination = False

        times, weights = _get_quadrature_points(n_points, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        assert dead_fracs.shape == (len(dps),)

    def test_values_are_nonnegative(self):
        """Test that dead fractions are non-negative."""
        n_points = 50
        time_end = 5.0
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        combination = False

        times, weights = _get_quadrature_points(n_points, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        assert np.all(dead_fracs >= 0)

    def test_sum_is_reasonable(self):
        """Test that total dead fraction is between 0 and init."""
        n_points = 50
        time_end = 5.0
        dps = np.arange(1, 500)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        combination = False

        times, weights = _get_quadrature_points(n_points, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        total_dead = np.sum(dead_fracs)
        assert 0 < total_dead < init


class TestQuadratureAccuracy:
    """Test that fixed quadrature matches quad_vec accuracy.

    These tests verify that the fixed Gauss-Legendre quadrature produces
    results equivalent to scipy's adaptive quad_vec within 1% relative error.
    Different termination orders require different numbers of quadrature points
    due to varying integrand smoothness.
    """

    def test_matches_quad_vec_order_1(self):
        """Test fixed quadrature matches quad_vec for order=1."""
        dps = np.arange(1, 300)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.0
        bn = 1.0
        time_end = 5.0
        combination = False

        # Reference: quad_vec
        args = (dps, alpha, init_mon, init, order, combination, bn)
        ref_dead_fracs, _ = quad_vec(
            living_distribution_integrand, 0, time_end, args=args
        )

        # Fixed quadrature with 100 points for 1% accuracy (sufficient for order=1)
        times, weights = _get_quadrature_points(100, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )
        test_dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        # Should match within 1%
        relative_error = np.abs(test_dead_fracs - ref_dead_fracs) / (ref_dead_fracs + 1e-15)
        significant = ref_dead_fracs > 1e-10
        assert np.all(relative_error[significant] < 0.01)

    def test_matches_quad_vec_order_1_5(self):
        """Test fixed quadrature matches quad_vec for order=1.5."""
        dps = np.arange(1, 300)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        time_end = 5.0
        combination = False

        # Reference: quad_vec
        args = (dps, alpha, init_mon, init, order, combination, bn)
        ref_dead_fracs, _ = quad_vec(
            living_distribution_integrand, 0, time_end, args=args
        )

        # Fixed quadrature with 400 points for 1% accuracy
        # Fractional orders require more points due to integrand complexity
        times, weights = _get_quadrature_points(400, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )
        test_dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        # Should match within 1%
        relative_error = np.abs(test_dead_fracs - ref_dead_fracs) / (ref_dead_fracs + 1e-15)
        significant = ref_dead_fracs > 1e-10
        assert np.all(relative_error[significant] < 0.01)

    def test_matches_quad_vec_order_2(self):
        """Test fixed quadrature matches quad_vec for order=2."""
        dps = np.arange(1, 300)
        alpha = 0.5
        init_mon = 1.0
        init = 0.005
        order = 2.0
        bn = 1.0
        time_end = 5.0
        combination = False

        # Reference: quad_vec
        args = (dps, alpha, init_mon, init, order, combination, bn)
        ref_dead_fracs, _ = quad_vec(
            living_distribution_integrand, 0, time_end, args=args
        )

        # Fixed quadrature with 100 points for 1% accuracy
        times, weights = _get_quadrature_points(100, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )
        test_dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        # Should match within 1%
        relative_error = np.abs(test_dead_fracs - ref_dead_fracs) / (ref_dead_fracs + 1e-15)
        significant = ref_dead_fracs > 1e-10
        assert np.all(relative_error[significant] < 0.01)


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

    def test_fit_returns_fit_result(self, simple_mws, standard_params):
        """Test that fit_mwd returns a FitResult object."""
        # Generate synthetic data
        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon']
        )

        assert isinstance(result, FitResult)

    def test_fit_result_has_all_attributes(self, simple_mws, standard_params):
        """Test that FitResult has all expected attributes."""
        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon']
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
        assert hasattr(result, 'predicted_intensities')
        assert hasattr(result, 'dead_chain_intensities')
        assert hasattr(result, 'dead_chain_fraction')
        assert hasattr(result, 'fit_message')


class TestRoundTripFitting:
    """Test fitting synthetic data recovers known parameters."""

    def test_fit_with_sigma_estimated(self, simple_mws, standard_params):
        """Test fitting with sigma=None (estimates sigma)."""
        true_alpha = standard_params['alpha']
        true_init = standard_params['init']

        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon']
        )

        # Should recover parameters within reasonable tolerance
        assert np.isclose(result.alpha, true_alpha, rtol=0.15)
        assert np.isclose(result.init, true_init, rtol=0.15)
        assert result.r_squared > 0.99

    def test_fit_with_fixed_sigma(self, simple_mws, standard_params):
        """Test fitting with fixed sigma (Gaussian broadening)."""
        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
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
        """Test fitting with fixed sigma and tau (EMG broadening)."""
        # Generate data with EMG broadening
        tau = 0.03
        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma'],
            tau=tau
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
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
        true_conv = (standard_params['init'] * standard_params['nu'] /
                     standard_params['init_mon'])

        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            init=standard_params['init']  # Fixed init
        )

        assert np.isclose(result.conversion, true_conv, rtol=0.01)
        assert result.r_squared > 0.99

    def test_fit_with_known_conversion(self, simple_mws, standard_params):
        """Test fitting with known conversion."""
        true_conv = (standard_params['init'] * standard_params['nu'] /
                     standard_params['init_mon'])

        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            conversion=true_conv  # Fixed conversion
        )

        assert np.isclose(result.init, standard_params['init'], rtol=0.01)
        assert result.r_squared > 0.99


class TestDifferentOrders:
    """Test fitting with different termination orders."""

    def test_first_order(self, simple_mws, first_order_params):
        """Test fitting first order termination."""
        mwd_ints = calculate_mwd(
            simple_mws,
            first_order_params['monomer_mw'],
            first_order_params['nu'],
            first_order_params['alpha'],
            first_order_params['init_mon'],
            first_order_params['init'],
            first_order_params['order'],
            first_order_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=first_order_params['order'],
            monomer_mw=first_order_params['monomer_mw'],
            init_mon=first_order_params['init_mon'],
            sigma=first_order_params['sigma']
        )

        assert result.r_squared > 0.95
        assert result.order == 1.0

    def test_second_order(self, simple_mws, second_order_params):
        """Test fitting second order termination."""
        mwd_ints = calculate_mwd(
            simple_mws,
            second_order_params['monomer_mw'],
            second_order_params['nu'],
            second_order_params['alpha'],
            second_order_params['init_mon'],
            second_order_params['init'],
            second_order_params['order'],
            second_order_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=second_order_params['order'],
            monomer_mw=second_order_params['monomer_mw'],
            init_mon=second_order_params['init_mon'],
            sigma=second_order_params['sigma']
        )

        assert result.r_squared > 0.95
        assert result.order == 2.0

    def test_fractional_order(self, simple_mws, other_order_params):
        """Test fitting fractional order termination."""
        mwd_ints = calculate_mwd(
            simple_mws,
            other_order_params['monomer_mw'],
            other_order_params['nu'],
            other_order_params['alpha'],
            other_order_params['init_mon'],
            other_order_params['init'],
            other_order_params['order'],
            other_order_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
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
        mwd_ints1 = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        mwd_ints2 = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'] * 1.5,  # Different nu
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        # Fit both using partial
        result1 = fit_my_instrument(simple_mws, mwd_ints1)
        result2 = fit_my_instrument(simple_mws, mwd_ints2)

        assert result1.r_squared > 0.99
        assert result2.r_squared > 0.99


class TestDownsampling:
    """Test that downsampling works correctly."""

    def test_fit_with_downsampling(self, simple_mws, standard_params):
        """Test that fitting works with different max_fit_points."""
        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result1 = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            max_fit_points=200
        )

        result2 = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            max_fit_points=500
        )

        # Both should give reasonable fits
        assert result1.r_squared > 0.90
        assert result2.r_squared > 0.90


class TestFitResultRepr:
    """Test FitResult string representation."""

    def test_repr_without_tau(self, simple_mws, standard_params):
        """Test repr for Gaussian broadening (no tau)."""
        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma']
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma']
        )

        repr_str = repr(result)
        assert 'FitResult' in repr_str
        assert 'alpha' in repr_str
        assert 'sigma' in repr_str
        assert 'tau' not in repr_str  # tau=0 should not be shown

    def test_repr_with_tau(self, simple_mws, standard_params):
        """Test repr for EMG broadening (with tau)."""
        tau = 0.03
        mwd_ints = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['nu'],
            standard_params['alpha'],
            standard_params['init_mon'],
            standard_params['init'],
            standard_params['order'],
            standard_params['sigma'],
            tau=tau
        )

        result = fit_mwd(
            simple_mws, mwd_ints,
            order=standard_params['order'],
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            sigma=standard_params['sigma'],
            tau=tau
        )

        repr_str = repr(result)
        assert 'tau' in repr_str  # tau > 0 should be shown


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
