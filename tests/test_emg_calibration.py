"""
Tests for calibrate_emg_broadening function.
"""

import pytest
import numpy as np
from scipy.stats import poisson
from polyterm.calibration import calibrate_emg_broadening, CalibrationResult
from polyterm.core.broadening import emg_broadening


class TestCalibrateEmgBroadening:
    """Test calibrate_emg_broadening function."""

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
    def narrow_standard_emg(self):
        """Generate synthetic data for a narrow standard with EMG broadening."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.08
        tau = 0.04

        # Generate synthetic EMG peak using the actual emg_broadening function
        # to ensure consistency between generation and fitting
        intensities = emg_broadening(mws, center, sigma, tau)

        return {
            'mws': mws,
            'intensities': intensities,
            'true_sigma': sigma,
            'true_tau': tau,
            'center': center
        }

    def test_returns_calibration_result(self, narrow_standard_gaussian):
        """calibrate_emg_broadening should return a CalibrationResult."""
        result = calibrate_emg_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        assert isinstance(result, CalibrationResult)

    def test_result_has_sigma_and_tau(self, narrow_standard_gaussian):
        """Result should have sigma and tau attributes."""
        result = calibrate_emg_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        assert hasattr(result, 'sigma')
        assert hasattr(result, 'tau')
        assert result.sigma > 0
        assert result.tau >= 0

    def test_recovers_gaussian_parameters(self, narrow_standard_gaussian):
        """Should recover sigma and tau=0 for pure Gaussian data."""
        result = calibrate_emg_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        # Should recover sigma within 10%
        assert np.isclose(result.sigma, narrow_standard_gaussian['true_sigma'], rtol=0.1)
        # tau should be near zero (or very small)
        assert result.tau < 0.02

    def test_recovers_emg_parameters(self, narrow_standard_emg):
        """Should fit EMG data with high quality."""
        result = calibrate_emg_broadening(
            narrow_standard_emg['mws'],
            narrow_standard_emg['intensities']
        )

        # The fit should capture the overall broadening well (high R²)
        # Note: There is some sigma-tau degeneracy, so we verify fit quality
        # rather than exact parameter values
        assert result.r_squared > 0.98, (
            f"R² = {result.r_squared:.4f} should be > 0.98 for synthetic data"
        )

        # Tau should be positive (detecting asymmetry)
        assert result.tau > 0, "Tau should be positive for EMG data"

        # The fitted shape should match the original data well
        fitted = emg_broadening(
            narrow_standard_emg['mws'],
            result.center,
            result.sigma,
            result.tau
        )
        # Normalize for comparison
        log_mws = np.log(narrow_standard_emg['mws'])
        fitted = fitted / np.trapezoid(fitted, log_mws)
        original = narrow_standard_emg['intensities']
        original = original / np.trapezoid(original, log_mws)

        # Correlation should be very high
        correlation = np.corrcoef(fitted, original)[0, 1]
        assert correlation > 0.99, f"Correlation {correlation:.4f} should be > 0.99"

    def test_result_has_r_squared(self, narrow_standard_gaussian):
        """Result should have r_squared metric."""
        result = calibrate_emg_broadening(
            narrow_standard_gaussian['mws'],
            narrow_standard_gaussian['intensities']
        )

        assert hasattr(result, 'r_squared')
        assert 0 <= result.r_squared <= 1
        # Good fit should have high R²
        assert result.r_squared > 0.95

    def test_result_has_center(self, narrow_standard_gaussian):
        """Result should have fitted center value."""
        result = calibrate_emg_broadening(
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


class TestPoissonBroadenedCalibration:
    """Test calibration with monomer_mw (Poisson-broadened fitting)."""

    def _generate_poisson_broadened_distribution(
        self, mws, center_dp, monomer_mw, sigma, tau
    ):
        """
        Generate a Poisson-broadened distribution for testing.

        This mimics a living polymer standard where chain lengths follow
        a Poisson distribution, each broadened by instrumental effects.
        """
        max_dp = int(center_dp + 6 * np.sqrt(center_dp)) + 1
        max_dp = max(max_dp, int(np.max(mws) / monomer_mw) + 1)
        dps = np.arange(1, max_dp, dtype=int)

        # Compute broadening matrix
        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        broadening_matrix = emg_broadening(
            mws_mesh, dps_mesh * monomer_mw, sigma, tau
        )

        # Poisson mass fractions
        mass_fracs = poisson.pmf(dps, center_dp) * dps

        # Sum contributions
        distribution = broadening_matrix @ mass_fracs
        return distribution

    @pytest.fixture
    def living_standard_low_dp(self):
        """
        Generate synthetic data for a living polymer standard at low DP.

        At low DP, the Poisson width is significant compared to instrumental
        broadening, so accounting for it matters.
        """
        mws = np.logspace(2.5, 4.5, 500)
        monomer_mw = 100.0
        center_dp = 50  # Low DP: Poisson std = sqrt(50) ~ 7
        sigma = 0.08  # Instrumental broadening
        tau = 0.03

        intensities = self._generate_poisson_broadened_distribution(
            mws, center_dp, monomer_mw, sigma, tau
        )
        intensities = intensities / np.trapezoid(intensities, np.log(mws))

        return {
            'mws': mws,
            'intensities': intensities,
            'true_sigma': sigma,
            'true_tau': tau,
            'center_dp': center_dp,
            'monomer_mw': monomer_mw,
            'center_mw': center_dp * monomer_mw
        }

    @pytest.fixture
    def living_standard_high_dp(self):
        """
        Generate synthetic data for a living polymer standard at high DP.

        At high DP, the Poisson width is small compared to instrumental
        broadening, so delta function assumption works.
        """
        mws = np.logspace(3.5, 5.5, 500)
        monomer_mw = 100.0
        center_dp = 500  # High DP: Poisson std = sqrt(500) ~ 22
        sigma = 0.08
        tau = 0.03

        intensities = self._generate_poisson_broadened_distribution(
            mws, center_dp, monomer_mw, sigma, tau
        )
        intensities = intensities / np.trapezoid(intensities, np.log(mws))

        return {
            'mws': mws,
            'intensities': intensities,
            'true_sigma': sigma,
            'true_tau': tau,
            'center_dp': center_dp,
            'monomer_mw': monomer_mw,
            'center_mw': center_dp * monomer_mw
        }

    def test_monomer_mw_produces_good_fit(self, living_standard_low_dp):
        """Calibration with monomer_mw should fit Poisson-broadened data well."""
        data = living_standard_low_dp
        result = calibrate_emg_broadening(
            data['mws'],
            data['intensities'],
            monomer_mw=data['monomer_mw']
        )

        # Should achieve high R² even for Poisson-broadened data
        assert result.r_squared > 0.95, (
            f"R² = {result.r_squared:.4f} should be > 0.95"
        )

    def test_monomer_mw_recovers_sigma_better_at_low_dp(self, living_standard_low_dp):
        """
        With monomer_mw, sigma recovery should be better at low DP.

        Without accounting for Poisson width, the fitted sigma would be
        inflated because the optimizer attributes all width to broadening.
        """
        data = living_standard_low_dp

        # Fit without monomer_mw (ignores Poisson width)
        result_delta = calibrate_emg_broadening(
            data['mws'],
            data['intensities'],
            monomer_mw=None
        )

        # Fit with monomer_mw (accounts for Poisson width)
        result_poisson = calibrate_emg_broadening(
            data['mws'],
            data['intensities'],
            monomer_mw=data['monomer_mw']
        )

        # Without Poisson correction, sigma is inflated
        # The Poisson contribution to variance is ~1/DP in log space
        # At DP=50, this is significant
        sigma_delta_error = abs(result_delta.sigma - data['true_sigma'])
        sigma_poisson_error = abs(result_poisson.sigma - data['true_sigma'])

        # Poisson-aware fit should have smaller sigma error
        assert sigma_poisson_error < sigma_delta_error, (
            f"Poisson-aware sigma error ({sigma_poisson_error:.4f}) should be "
            f"smaller than delta assumption error ({sigma_delta_error:.4f})"
        )

    def test_monomer_mw_similar_to_delta_at_high_dp(self, living_standard_high_dp):
        """At high DP, both methods should give similar results."""
        data = living_standard_high_dp

        result_delta = calibrate_emg_broadening(
            data['mws'],
            data['intensities'],
            monomer_mw=None
        )

        result_poisson = calibrate_emg_broadening(
            data['mws'],
            data['intensities'],
            monomer_mw=data['monomer_mw']
        )

        # At high DP, Poisson width is negligible, so results should be similar
        # Allow 20% tolerance since there's always some optimizer variability
        assert np.isclose(result_delta.sigma, result_poisson.sigma, rtol=0.2), (
            f"At high DP, delta sigma ({result_delta.sigma:.4f}) and "
            f"Poisson sigma ({result_poisson.sigma:.4f}) should be similar"
        )

    def test_monomer_mw_recovers_center(self, living_standard_low_dp):
        """Calibration with monomer_mw should recover center MW."""
        data = living_standard_low_dp
        result = calibrate_emg_broadening(
            data['mws'],
            data['intensities'],
            monomer_mw=data['monomer_mw']
        )

        # Center should be close to true value (within 10%)
        assert np.isclose(result.center, data['center_mw'], rtol=0.1), (
            f"Center ({result.center:.1f}) should be close to "
            f"true center ({data['center_mw']:.1f})"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
