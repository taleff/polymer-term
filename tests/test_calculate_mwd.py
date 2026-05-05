"""
Tests for calculate_mwd module.

This module tests the accuracy of the Gauss-Legendre quadrature
implementation by comparing it against scipy's adaptive quad_vec method.
"""

import numpy as np
import pytest
from scipy.stats import poisson
from scipy.integrate import quad_vec

from polyterm import calculate_mwd
from polyterm.core.kinetics import (
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    time_to_chain_death,
)
from polyterm.core.broadening import egh_broadening


# Reference implementation: the old quad_vec approach
def living_distribution_integrand(
    time: float,
    dps: np.ndarray,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    combination: float = 0.0,
    bn: float = 1.0
) -> np.ndarray:
    """
    Reference implementation of the living chain distribution integrand.

    This is the original quad_vec-based integrand used before the
    Gauss-Legendre refactoring.
    """
    b = living_chain_concentration(init, order, time)
    nup = living_chain_dp(alpha, init_mon, init, order, time, bn)
    death_rate = (b ** order) * (init ** (1 - order))

    disp = death_rate * poisson.pmf(dps, nup)
    comb = death_rate * poisson.pmf(dps, 2 * nup) / 2
    return (1 - combination) * disp + combination * comb


def calculate_distribution_quad_vec(
    dps: np.ndarray,
    conversion: float,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    combination: float = 0.0,
    bn: float = 1.0
):
    """
    Reference implementation using quad_vec for dead chain integration.

    This replicates the old approach before Gauss-Legendre optimization.
    """
    CHAIN_DEATH_FRACTION = 0.9999

    if np.isclose(conversion, 1):
        red_time = time_to_chain_death(CHAIN_DEATH_FRACTION, init, order)
    else:
        red_time = conversion_to_time(alpha, init, order, conversion, bn)

    b = living_chain_concentration(init, order, red_time)
    nup = living_chain_dp(alpha, init_mon, init, order, red_time, bn)

    # Living chains follow a Poisson distribution
    alive_fracs = b * poisson.pmf(dps, nup)

    # Dead chains: integrate over all termination times using quad_vec
    args = (dps, alpha, init_mon, init, order, combination, bn)
    dead_fracs, _ = quad_vec(living_distribution_integrand, 0, red_time,
                             args=args)

    return np.array(alive_fracs), np.array(dead_fracs)


def calculate_mwd_quad_vec(
    molecular_weights: np.ndarray,
    monomer_mw: float,
    init_mon: float,
    alpha: float,
    init: float,
    conversion: float,
    order: float,
    sigma: float,
    tau: float = 0,
    combination: float = 0.0,
    bn: float = 1.0
) -> np.ndarray:
    """
    Reference MWD calculation using quad_vec integration.

    This is used as the reference to compare against the new
    Gauss-Legendre implementation.
    """
    max_dp = int(np.max(molecular_weights) / monomer_mw)
    dps = np.arange(1, max_dp, dtype=int)

    alive_fracs, dead_fracs = calculate_distribution_quad_vec(
        dps, conversion, alpha, init_mon, init, order, combination, bn
    )

    # Create meshgrid for broadening calculation
    dps_mesh, mws_mesh = np.meshgrid(dps, molecular_weights)
    broadenings = egh_broadening(mws_mesh, dps_mesh * monomer_mw, sigma, tau)

    # Convert to weight distribution
    tot_dist = (alive_fracs + dead_fracs) * dps
    raw_mwd = np.matmul(broadenings, tot_dist)

    # Normalize by peak
    mwd = raw_mwd / np.max(raw_mwd)

    return mwd, alive_fracs, dead_fracs


class TestQuadratureAccuracy:
    """Test that Gauss-Legendre quadrature matches quad_vec accuracy.

    These tests verify that the fixed Gauss-Legendre quadrature produces
    results equivalent to scipy's adaptive quad_vec within acceptable
    tolerances. Different termination orders are tested.
    """

    @pytest.fixture
    def standard_mws(self):
        """Standard molecular weight array for testing."""
        return np.logspace(2, 5, 500)

    @pytest.fixture
    def standard_params(self):
        """Standard kinetic parameters."""
        return {
            'monomer_mw': 104.15,
            'init_mon': 1.0,
            'init': 0.01,
            'sigma': 0.128,
            'tau': 0.0456,
        }

    def test_order_1_distribution_matches_quad_vec(
        self, standard_mws, standard_params
    ):
        """Test that first-order termination matches quad_vec reference."""
        # For order=1, max conversion = 1 - exp(-init/alpha)
        # With alpha=0.001, init=0.01: max_conv = 1 - exp(-10) ≈ 0.99995
        alpha = 0.001
        conversion = 0.7
        order = 1.0

        # Reference: quad_vec implementation
        ref_mwd, _, _ = calculate_mwd_quad_vec(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # New Gauss-Legendre implementation
        result = calculate_mwd(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # Compare distributions
        # Focus on region with significant intensity (>1% of max)
        significant = result.intensities > 0.01 * np.max(result.intensities)
        relative_error = np.abs(result.intensities - ref_mwd) / (ref_mwd + 1e-15)

        # Should match within 1% where intensity is significant
        assert np.all(relative_error[significant] < 0.01), \
            f"Max relative error: {np.max(relative_error[significant]):.4f}"

    def test_order_1_5_distribution_matches_quad_vec(
        self, standard_mws, standard_params
    ):
        """Test that order=1.5 termination matches quad_vec reference."""
        alpha = 0.05
        conversion = 0.6
        order = 1.5

        # Reference: quad_vec implementation
        ref_mwd, _, _ = calculate_mwd_quad_vec(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # New Gauss-Legendre implementation
        result = calculate_mwd(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # Compare distributions
        significant = result.intensities > 0.01 * np.max(result.intensities)
        relative_error = np.abs(result.intensities - ref_mwd) / (ref_mwd + 1e-15)

        # Order 1.5 may have slightly higher tolerance due to integrand complexity
        assert np.all(relative_error[significant] < 0.02), \
            f"Max relative error: {np.max(relative_error[significant]):.4f}"

    def test_order_2_distribution_matches_quad_vec(
        self, standard_mws, standard_params
    ):
        """Test that second-order termination matches quad_vec reference."""
        # Note: alpha=1.0 causes singularity in living_chain_dp for order=2
        alpha = 2.0
        conversion = 0.5
        order = 2.0

        # Reference: quad_vec implementation
        ref_mwd, _, _ = calculate_mwd_quad_vec(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # New Gauss-Legendre implementation
        result = calculate_mwd(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # Compare distributions
        significant = result.intensities > 0.01 * np.max(result.intensities)
        relative_error = np.abs(result.intensities - ref_mwd) / (ref_mwd + 1e-15)

        assert np.all(relative_error[significant] < 0.01), \
            f"Max relative error: {np.max(relative_error[significant]):.4f}"

    def test_dead_chain_fraction_matches_quad_vec(
        self, standard_mws, standard_params
    ):
        """Test that dead chain fraction matches between implementations."""
        # Parameters chosen to be physically achievable for each order
        # Note: For order=2, alpha=1.0 causes singularity in living_chain_dp
        test_cases = [
            {'alpha': 0.001, 'conversion': 0.7, 'order': 1.0},
            {'alpha': 0.05, 'conversion': 0.6, 'order': 1.5},
            {'alpha': 2.0, 'conversion': 0.5, 'order': 2.0},
        ]

        for case in test_cases:
            # Reference: quad_vec
            _, ref_alive, ref_dead = calculate_mwd_quad_vec(
                standard_mws,
                standard_params['monomer_mw'],
                standard_params['init_mon'],
                case['alpha'],
                standard_params['init'],
                case['conversion'],
                case['order'],
                standard_params['sigma'],
                standard_params['tau']
            )
            ref_dead_frac = np.sum(ref_dead) / np.sum(ref_alive + ref_dead)

            # New implementation
            result = calculate_mwd(
                standard_mws,
                standard_params['monomer_mw'],
                standard_params['init_mon'],
                case['alpha'],
                standard_params['init'],
                case['conversion'],
                case['order'],
                standard_params['sigma'],
                standard_params['tau']
            )

            # Dead chain fraction should match within 1%
            relative_error = abs(result.dead_chain_fraction - ref_dead_frac) / (ref_dead_frac + 1e-15)
            assert relative_error < 0.01, \
                f"Order {case['order']}: dead frac error {relative_error:.4f}"

    def test_high_conversion_matches_quad_vec(
        self, standard_mws, standard_params
    ):
        """Test accuracy at high conversion (near 100%)."""
        # For order=1, max conversion = 1 - exp(-init/alpha)
        # With alpha=0.0005, init=0.01: max_conv = 1 - exp(-20) ≈ 0.9999999
        alpha = 0.0005
        conversion = 0.95
        order = 1.0

        # Reference: quad_vec
        ref_mwd, _, _ = calculate_mwd_quad_vec(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # New implementation
        result = calculate_mwd(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # Compare
        significant = result.intensities > 0.01 * np.max(result.intensities)
        relative_error = np.abs(result.intensities - ref_mwd) / (ref_mwd + 1e-15)

        assert np.all(relative_error[significant] < 0.02), \
            f"Max relative error at high conversion: {np.max(relative_error[significant]):.4f}"

    def test_low_alpha_matches_quad_vec(
        self, standard_mws, standard_params
    ):
        """Test accuracy with very low alpha (minimal termination)."""
        alpha = 0.001
        conversion = 0.5
        order = 1.0

        # Reference: quad_vec
        ref_mwd, _, _ = calculate_mwd_quad_vec(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # New implementation
        result = calculate_mwd(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # Compare
        significant = result.intensities > 0.01 * np.max(result.intensities)
        relative_error = np.abs(result.intensities - ref_mwd) / (ref_mwd + 1e-15)

        assert np.all(relative_error[significant] < 0.01), \
            f"Max relative error with low alpha: {np.max(relative_error[significant]):.4f}"

    def test_high_alpha_matches_quad_vec(
        self, standard_mws, standard_params
    ):
        """Test accuracy with high alpha (significant termination)."""
        alpha = 5.0
        conversion = 0.3
        order = 2.0

        # Reference: quad_vec
        ref_mwd, _, _ = calculate_mwd_quad_vec(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # New implementation
        result = calculate_mwd(
            standard_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            alpha,
            standard_params['init'],
            conversion,
            order,
            standard_params['sigma'],
            standard_params['tau']
        )

        # Compare
        significant = result.intensities > 0.01 * np.max(result.intensities)
        relative_error = np.abs(result.intensities - ref_mwd) / (ref_mwd + 1e-15)

        assert np.all(relative_error[significant] < 0.01), \
            f"Max relative error with high alpha: {np.max(relative_error[significant]):.4f}"


class TestCalculateMwdBasic:
    """Basic functionality tests for calculate_mwd."""

    def test_returns_mwd_result(self):
        """Test that function returns an MWDResult."""
        from polyterm.mwd import MWDResult

        mws = np.logspace(2, 5, 100)
        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=0.01,
            init=0.01, conversion=0.5, order=1.0, sigma=0.13
        )

        assert isinstance(result, MWDResult)

    def test_intensities_normalized(self):
        """Test that intensities peak is normalized to 1."""
        mws = np.logspace(2, 5, 500)
        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=0.01,
            init=0.01, conversion=0.5, order=1.0, sigma=0.13
        )

        assert np.isclose(np.max(result.intensities), 1.0, rtol=0.01)

    def test_no_negative_intensities(self):
        """Test that all intensities are non-negative."""
        mws = np.logspace(2, 5, 500)
        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=0.01,
            init=0.01, conversion=0.5, order=1.0, sigma=0.13
        )

        assert np.all(result.intensities >= 0)
        assert np.all(result.live_chain_intensities >= 0)
        assert np.all(result.dead_chain_intensities >= 0)

    def test_live_plus_dead_equals_total(self):
        """Test that live + dead intensities equal total."""
        mws = np.logspace(2, 5, 500)
        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=0.01,
            init=0.01, conversion=0.5, order=1.0, sigma=0.13
        )

        total = result.live_chain_intensities + result.dead_chain_intensities
        np.testing.assert_allclose(
            result.intensities, total, rtol=1e-10
        )

    def test_dead_fraction_in_valid_range(self):
        """Test that dead chain fraction is between 0 and 1."""
        mws = np.logspace(2, 5, 500)
        result = calculate_mwd(
            mws, monomer_mw=104.15, init_mon=1.0, alpha=0.01,
            init=0.01, conversion=0.5, order=1.0, sigma=0.13
        )

        assert 0 <= result.dead_chain_fraction <= 1


class TestDistributionParameter:
    """Test the distribution parameter for custom chain length distributions."""

    @pytest.fixture
    def base_params(self):
        """Common parameters for distribution tests."""
        return {
            'molecular_weights': np.logspace(2, 5, 500),
            'monomer_mw': 104.15,
            'init_mon': 1.0,
            'alpha': 0.01,
            'init': 0.01,
            'conversion': 0.5,
            'order': 1.0,
            'sigma': 0.13,
        }

    def test_explicit_poisson_matches_default(self, base_params):
        """Passing poisson.pmf explicitly gives same result as default."""
        result_default = calculate_mwd(**base_params)
        result_poisson = calculate_mwd(**base_params, distribution=poisson.pmf)

        np.testing.assert_allclose(
            result_default.intensities, result_poisson.intensities, rtol=1e-10
        )

    def test_custom_distribution_changes_output(self, base_params):
        """A different distribution function produces different intensities."""
        result_default = calculate_mwd(**base_params)

        # Use a Gaussian-like distribution instead of Poisson
        def gaussian_dist(dps, nup):
            sigma = np.sqrt(nup)
            return np.exp(-0.5 * ((dps - nup) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )

        result_custom = calculate_mwd(**base_params, distribution=gaussian_dist)

        # Results should differ (Gaussian != Poisson, especially in tails)
        assert not np.allclose(
            result_default.intensities, result_custom.intensities, rtol=0.01
        )

    def test_custom_distribution_produces_valid_mwd(self, base_params):
        """A custom distribution still produces normalized, non-negative output."""
        def gaussian_dist(dps, nup):
            sigma = np.sqrt(nup)
            return np.exp(-0.5 * ((dps - nup) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )

        result = calculate_mwd(**base_params, distribution=gaussian_dist)

        assert np.all(result.intensities >= 0)
        assert np.isclose(np.max(result.intensities), 1.0, rtol=0.01)

    def test_custom_distribution_used_in_dead_chains(self, base_params):
        """Custom distribution affects dead chain intensities."""
        result_default = calculate_mwd(**base_params)

        # Narrow distribution: only populate chains near the mean DP
        def narrow_dist(dps, nup):
            sigma = np.maximum(np.sqrt(nup) * 0.1, 1.0)
            return np.exp(-0.5 * ((dps - nup) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )

        result_custom = calculate_mwd(**base_params, distribution=narrow_dist)

        # Dead chain intensities should differ
        assert not np.allclose(
            result_default.dead_chain_intensities,
            result_custom.dead_chain_intensities,
            rtol=0.01
        )

    def test_custom_distribution_used_in_live_chains(self, base_params):
        """Custom distribution affects live chain intensities."""
        result_default = calculate_mwd(**base_params)

        def narrow_dist(dps, nup):
            sigma = np.maximum(np.sqrt(nup) * 0.1, 1.0)
            return np.exp(-0.5 * ((dps - nup) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )

        result_custom = calculate_mwd(**base_params, distribution=narrow_dist)

        assert not np.allclose(
            result_default.live_chain_intensities,
            result_custom.live_chain_intensities,
            rtol=0.01
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
