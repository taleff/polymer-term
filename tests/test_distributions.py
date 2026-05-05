"""
Tests for distribution calculation functions.
"""

import pytest
import numpy as np
from polyterm.core.broadening import gaussian_broadening
from polyterm import calculate_mwd


class TestGaussianBroadening:
    """Test Gaussian broadening function."""

    def test_gaussian_properties(self, gaussian_test_data):
        """Test basic properties of Gaussian broadening."""
        result = gaussian_broadening(
            gaussian_test_data['mws'],
            gaussian_test_data['center'],
            gaussian_test_data['sigma']
        )

        # All values should be positive
        assert np.all(result > 0)
        # Peak should be near center
        peak_idx = np.argmax(result)
        assert np.isclose(gaussian_test_data['mws'][peak_idx], gaussian_test_data['center'], rtol=0.1)

    def test_peak_at_center(self, gaussian_test_data):
        """Test that peak is at the center."""
        result = gaussian_broadening(
            gaussian_test_data['mws'],
            gaussian_test_data['center'],
            gaussian_test_data['sigma']
        )

        peak_idx = np.argmax(result)
        peak_mw = gaussian_test_data['mws'][peak_idx]
        assert np.isclose(peak_mw, gaussian_test_data['center'], rtol=0.05)

    def test_positive_values(self, gaussian_test_data):
        """Test that all values are positive."""
        result = gaussian_broadening(
            gaussian_test_data['mws'],
            gaussian_test_data['center'],
            gaussian_test_data['sigma']
        )

        assert np.all(result > 0)

    def test_multiple_centers_sequentially(self):
        """Test with multiple centers applied sequentially."""
        mws = np.logspace(3, 5, 100)
        centers = [5000.0, 10000.0, 20000.0]
        sigma = 0.1

        for center in centers:
            result = gaussian_broadening(mws, center, sigma)
            assert result.shape == mws.shape
            assert np.all(result > 0)


class TestCalculateMWD:
    """Test calculate_mwd function."""

    def test_basic_mwd(self, simple_mws, standard_params):
        """Test basic MWD calculation."""
        result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        assert result.intensities.shape == simple_mws.shape
        assert np.all(result.intensities >= 0)

    def test_normalized(self, simple_mws, standard_params):
        """Test that MWD peak is normalized to 1."""
        result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        assert np.isclose(np.max(result.intensities), 1.0, rtol=0.01)

    def test_first_order_mwd(self, simple_mws, first_order_params):
        """Test MWD with first-order termination."""
        result = calculate_mwd(
            simple_mws,
            first_order_params['monomer_mw'],
            first_order_params['init_mon'],
            first_order_params['alpha'],
            first_order_params['init'],
            first_order_params['conversion'],
            first_order_params['order'],
            first_order_params['sigma']
        )

        assert np.all(np.isfinite(result.intensities))
        assert np.any(result.intensities > 0)

    def test_second_order_mwd(self, simple_mws, second_order_params):
        """Test MWD with second-order termination."""
        result = calculate_mwd(
            simple_mws,
            second_order_params['monomer_mw'],
            second_order_params['init_mon'],
            second_order_params['alpha'],
            second_order_params['init'],
            second_order_params['conversion'],
            second_order_params['order'],
            second_order_params['sigma']
        )

        assert np.all(np.isfinite(result.intensities))
        assert np.any(result.intensities > 0)

    def test_returns_mwd_result(self, simple_mws, standard_params):
        """Test that calculate_mwd returns an MWDResult."""
        from polyterm import MWDResult

        result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        assert isinstance(result, MWDResult)

    def test_has_live_and_dead_fractions(self, simple_mws, standard_params):
        """Test that result has live and dead chain intensities."""
        result = calculate_mwd(
            simple_mws,
            standard_params['monomer_mw'],
            standard_params['init_mon'],
            standard_params['alpha'],
            standard_params['init'],
            standard_params['conversion'],
            standard_params['order'],
            standard_params['sigma']
        )

        assert hasattr(result, 'live_chain_intensities')
        assert hasattr(result, 'dead_chain_intensities')
        assert hasattr(result, 'dead_chain_fraction')
        assert 0 <= result.dead_chain_fraction <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
