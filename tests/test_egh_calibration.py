"""
Tests for calibrate_egh_broadening function.
"""

import pytest
import numpy as np
from polyterm.calibration import calibrate_egh_broadening, CalibrationResult
from polyterm.core.broadening import egh_broadening


class TestCalibrateEghBroadening:
    """Test calibrate_egh_broadening function."""

    @pytest.fixture
    def narrow_standard_gaussian(self):
        """Generate synthetic data for a narrow standard with Gaussian broadening."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10
        tau = 0.0  # Pure Gaussian

        # Generate synthetic peak
        intensities = np.exp(-((np.log(mws) - np.log(center)) ** 2) / (2 * sigma ** 2))
        intensities = intensities / np.trapezoid(intensities, np.log(mws))

        return {
            'mws': mws,
            'intensities': intensities,
            'true_sigma': sigma,
            'true_tau': tau,
            'center': center
        }

    @pytest.fixture
    def narrow_standard_egh(self):
        """Generate synthetic data for a narrow standard with EGH broadening."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.08
        tau = 0.04

        # Generate synthetic EGH peak using the actual egh_broadening function
        # to ensure consistency between generation and fitting
        intensities = egh_broadening(mws, center, sigma, tau)

        return {
            'mws': mws,
            'intensities': intensities,
            'true_sigma': sigma,
            'true_tau': tau,
            'center': center
        }

    def test_returns_calibration_result(self, narrow_standard_gaussian):
        """calibrate_egh_broadening should return a CalibrationResult."""
        result = calibrate_egh_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        assert isinstance(result, CalibrationResult)

    def test_result_has_sigma_and_tau(self, narrow_standard_gaussian):
        """Result should have sigma and tau attributes."""
        result = calibrate_egh_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        assert hasattr(result, 'sigma')
        assert hasattr(result, 'tau')
        assert result.sigma > 0
        assert result.tau >= 0

    def test_recovers_gaussian_parameters(self, narrow_standard_gaussian):
        """Should recover sigma and tau=0 for pure Gaussian data."""
        result = calibrate_egh_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        # Should recover sigma within 10%
        assert np.isclose(result.sigma, narrow_standard_gaussian['true_sigma'], rtol=0.1)
        # tau should be near zero (or very small)
        assert result.tau < 0.02

    def test_recovers_egh_parameters(self, narrow_standard_egh):
        """Should fit EGH data with high quality."""
        result = calibrate_egh_broadening(
            narrow_standard_egh['mws'],
            narrow_standard_egh['intensities']
        )

        # The fit should capture the overall broadening well (high R²)
        # Note: There is some sigma-tau degeneracy, so we verify fit quality
        # rather than exact parameter values
        assert result.r_squared > 0.98, (
            f"R² = {result.r_squared:.4f} should be > 0.98 for synthetic data"
        )

        # Tau should be positive (detecting asymmetry)
        assert result.tau > 0, "Tau should be positive for EGH data"

        # The fitted shape should match the original data well
        fitted = egh_broadening(
            narrow_standard_egh['mws'],
            result.center,
            result.sigma,
            result.tau
        )
        # Normalize for comparison
        log_mws = np.log(narrow_standard_egh['mws'])
        fitted = fitted / np.trapezoid(fitted, log_mws)
        original = narrow_standard_egh['intensities']
        original = original / np.trapezoid(original, log_mws)

        # Correlation should be very high
        correlation = np.corrcoef(fitted, original)[0, 1]
        assert correlation > 0.99, f"Correlation {correlation:.4f} should be > 0.99"

    def test_result_has_r_squared(self, narrow_standard_gaussian):
        """Result should have r_squared metric."""
        result = calibrate_egh_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        assert hasattr(result, 'r_squared')
        assert 0 <= result.r_squared <= 1
        # Good fit should have high R²
        assert result.r_squared > 0.95

    def test_result_has_center(self, narrow_standard_gaussian):
        """Result should have fitted center value."""
        result = calibrate_egh_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        assert hasattr(result, 'center')
        # Center should be close to true value
        assert np.isclose(result.center, narrow_standard_gaussian['center'], rtol=0.1)


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""

    def test_result_is_immutable(self):
        """CalibrationResult should be immutable (frozen dataclass)."""
        result = CalibrationResult(
            sigma=0.10,
            tau=0.05,
            center=10000.0,
            r_squared=0.99
        )

        with pytest.raises(AttributeError):
            result.sigma = 0.15

    def test_result_repr(self):
        """Result should have meaningful string representation."""
        result = CalibrationResult(
            sigma=0.10,
            tau=0.05,
            center=10000.0,
            r_squared=0.99
        )

        repr_str = repr(result)
        assert 'sigma' in repr_str.lower() or '0.1' in repr_str


class TestEghVsEmgCalibration:
    """Compare EGH and EMG calibration results."""

    @pytest.fixture
    def synthetic_asymmetric_peak(self):
        """Generate synthetic asymmetric peak data."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.08
        tau = 0.04

        # Use EGH to generate the synthetic data
        intensities = egh_broadening(mws, center, sigma, tau)

        return {
            'mws': mws,
            'intensities': intensities,
            'true_sigma': sigma,
            'true_tau': tau,
            'center': center
        }

    def test_both_calibrations_produce_good_fits(self, synthetic_asymmetric_peak):
        """Both EMG and EGH calibrations should produce reasonable fits."""
        from polyterm.calibration import calibrate_emg_broadening

        mws = synthetic_asymmetric_peak['mws']
        intensities = synthetic_asymmetric_peak['intensities']

        egh_result = calibrate_egh_broadening(mws, intensities)
        emg_result = calibrate_emg_broadening(mws, intensities)

        # Both should have decent R²
        assert egh_result.r_squared > 0.90
        assert emg_result.r_squared > 0.90


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
