"""
Tests for egh_broadening function.
"""

import pytest
import numpy as np
from polyterm.core.broadening import egh_broadening, gaussian_broadening


class TestEghBroadening:
    """Test egh_broadening function."""

    @pytest.fixture
    def mws(self):
        """Standard molecular weight array for testing."""
        return np.logspace(3, 5, 500)

    def test_returns_array(self, mws):
        """egh_broadening should return a numpy array."""
        result = egh_broadening(mws, center=10000.0, sigma=0.10, tau=0.05)
        assert isinstance(result, np.ndarray)
        assert result.shape == mws.shape

    def test_reduces_to_gaussian_when_tau_zero(self, mws):
        """When tau=0, EGH should equal Gaussian broadening."""
        center = 10000.0
        sigma = 0.10

        egh_result = egh_broadening(mws, center, sigma, tau=0.0)
        gaussian_result = gaussian_broadening(mws, center, sigma)

        np.testing.assert_allclose(egh_result, gaussian_result, rtol=1e-10)

    def test_reduces_to_gaussian_when_tau_small(self, mws):
        """When tau is very small, EGH should approximate Gaussian."""
        center = 10000.0
        sigma = 0.10

        egh_result = egh_broadening(mws, center, sigma, tau=1e-12)
        gaussian_result = gaussian_broadening(mws, center, sigma)

        np.testing.assert_allclose(egh_result, gaussian_result, rtol=1e-10)

    def test_normalized_to_unity(self, mws):
        """EGH should integrate to 1 over log(MW) space."""
        result = egh_broadening(mws, center=10000.0, sigma=0.10, tau=0.05)
        log_mws = np.log(mws)
        integral = np.trapezoid(result, log_mws)

        assert np.isclose(integral, 1.0, rtol=0.01)

    def test_peak_near_center(self, mws):
        """Peak maximum should be near the center."""
        center = 10000.0
        result = egh_broadening(mws, center, sigma=0.10, tau=0.05)
        peak_mw = mws[np.argmax(result)]

        # Peak should be within 20% of center (tau shifts the peak slightly)
        assert np.isclose(peak_mw, center, rtol=0.2)

    def test_asymmetric_tailing_toward_lower_mw(self, mws):
        """With tau > 0, should show tailing toward lower MW."""
        center = 10000.0
        sigma = 0.10
        tau = 0.08

        result = egh_broadening(mws, center, sigma, tau)
        log_mws = np.log(mws)
        log_center = np.log(center)

        # Find indices for left (lower MW) and right (higher MW) of center
        peak_idx = np.argmax(result)
        left_mask = np.arange(len(mws)) < peak_idx
        right_mask = np.arange(len(mws)) > peak_idx

        # Calculate areas on each side
        if np.any(left_mask) and np.any(right_mask):
            left_area = np.trapezoid(result[left_mask], log_mws[left_mask])
            right_area = np.trapezoid(result[right_mask], log_mws[right_mask])

            # Tailing toward lower MW means more area on the left side
            assert left_area > right_area, "Expected more area on low-MW side (tailing)"

    def test_more_asymmetric_with_larger_tau(self, mws):
        """Larger tau should produce more asymmetry."""
        center = 10000.0
        sigma = 0.10

        result_small_tau = egh_broadening(mws, center, sigma, tau=0.02)
        result_large_tau = egh_broadening(mws, center, sigma, tau=0.10)

        # Calculate asymmetry as ratio of left/right areas
        log_mws = np.log(mws)

        peak_idx_small = np.argmax(result_small_tau)
        peak_idx_large = np.argmax(result_large_tau)

        left_area_small = np.trapezoid(result_small_tau[:peak_idx_small],
                                        log_mws[:peak_idx_small])
        right_area_small = np.trapezoid(result_small_tau[peak_idx_small:],
                                         log_mws[peak_idx_small:])

        left_area_large = np.trapezoid(result_large_tau[:peak_idx_large],
                                        log_mws[:peak_idx_large])
        right_area_large = np.trapezoid(result_large_tau[peak_idx_large:],
                                         log_mws[peak_idx_large:])

        asymmetry_small = left_area_small / (right_area_small + 1e-10)
        asymmetry_large = left_area_large / (right_area_large + 1e-10)

        assert asymmetry_large > asymmetry_small

    def test_cutoff_at_high_mw(self, mws):
        """EGH should cut off (go to zero) at high MW for tau > 0."""
        center = 10000.0
        sigma = 0.10
        tau = 0.10

        result = egh_broadening(mws, center, sigma, tau)

        # At very high MW, the function should be exactly zero
        # (where denominator becomes <= 0)
        high_mw_mask = mws > center * np.exp(2 * sigma**2 / tau)

        if np.any(high_mw_mask):
            assert np.all(result[high_mw_mask] == 0), \
                "EGH should be zero where denominator <= 0"

    def test_non_negative(self, mws):
        """EGH values should always be non-negative."""
        result = egh_broadening(mws, center=10000.0, sigma=0.10, tau=0.08)
        assert np.all(result >= 0)

    def test_no_nan_or_inf(self, mws):
        """EGH should not produce NaN or Inf values."""
        result = egh_broadening(mws, center=10000.0, sigma=0.10, tau=0.08)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_different_centers(self, mws):
        """EGH should work with different center values."""
        for center in [5000.0, 10000.0, 50000.0]:
            result = egh_broadening(mws, center, sigma=0.10, tau=0.05)
            assert result.shape == mws.shape
            assert np.all(result >= 0)
            # Check normalization
            integral = np.trapezoid(result, np.log(mws))
            assert np.isclose(integral, 1.0, rtol=0.02)

    def test_different_sigma_values(self, mws):
        """EGH should work with different sigma values."""
        for sigma in [0.05, 0.10, 0.20]:
            result = egh_broadening(mws, center=10000.0, sigma=sigma, tau=0.05)
            assert result.shape == mws.shape
            assert np.all(result >= 0)

    def test_wider_peak_with_larger_sigma(self, mws):
        """Larger sigma should produce wider peaks."""
        center = 10000.0
        tau = 0.05

        result_narrow = egh_broadening(mws, center, sigma=0.05, tau=tau)
        result_wide = egh_broadening(mws, center, sigma=0.15, tau=tau)

        # Peak height should be lower for wider peak (same total area)
        assert np.max(result_wide) < np.max(result_narrow)


class TestEghVsEmg:
    """Compare EGH and EMG behavior."""

    @pytest.fixture
    def mws(self):
        """Standard molecular weight array for testing."""
        return np.logspace(3, 5, 500)

    def test_similar_at_low_asymmetry(self, mws):
        """EGH and EMG should be similar at low asymmetries."""
        from polyterm.core.broadening import emg_broadening

        center = 10000.0
        sigma = 0.10
        tau = 0.02  # Low asymmetry

        egh_result = egh_broadening(mws, center, sigma, tau)
        emg_result = emg_broadening(mws, center, sigma, tau)

        # They should be reasonably similar (not identical, but close)
        correlation = np.corrcoef(egh_result, emg_result)[0, 1]
        assert correlation > 0.99, f"EGH and EMG should be similar at low tau, got r={correlation}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
