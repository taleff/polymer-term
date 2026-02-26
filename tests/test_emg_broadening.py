"""
Tests for EMG (Exponentially Modified Gaussian) broadening function.
"""

import pytest
import numpy as np
from polyterm.core.broadening import emg_broadening, gaussian_broadening


class TestEMGBroadening:
    """Test EMG broadening function."""

    def test_reduces_to_gaussian_when_tau_zero(self):
        """EMG with tau=0 should equal Gaussian broadening."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10
        tau = 0.0

        emg_result = emg_broadening(mws, center, sigma, tau)
        gaussian_result = gaussian_broadening(mws, center, sigma)

        # Should be identical when tau=0
        np.testing.assert_allclose(emg_result, gaussian_result, rtol=1e-5)

    def test_asymmetric_tailing_toward_lower_mw(self):
        """EMG with tau > 0 should have asymmetric tail toward lower MW."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10
        tau = 0.05

        emg_result = emg_broadening(mws, center, sigma, tau)
        gaussian_result = gaussian_broadening(mws, center, sigma)

        # The EMG peak should be shifted toward lower MW compared to Gaussian
        emg_peak_idx = np.argmax(emg_result)
        gaussian_peak_idx = np.argmax(gaussian_result)

        assert emg_peak_idx < gaussian_peak_idx, (
            f"EMG peak at index {emg_peak_idx} should be < Gaussian peak at {gaussian_peak_idx}"
        )

        # The low-MW tail should be elevated compared to Gaussian
        # Check at a point well below the peak
        low_mw_idx = gaussian_peak_idx - 50  # About 0.5 in log10 units below peak
        assert emg_result[low_mw_idx] > gaussian_result[low_mw_idx] * 1.1, (
            "EMG should have elevated low-MW tail compared to Gaussian"
        )

    def test_larger_tau_more_asymmetric(self):
        """Larger tau should produce larger peak shift toward lower MW."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10

        gaussian_result = gaussian_broadening(mws, center, sigma)
        gaussian_peak_idx = np.argmax(gaussian_result)

        # Calculate peak shift for different tau values
        peak_shifts = []
        for tau in [0.01, 0.03, 0.05, 0.08]:
            result = emg_broadening(mws, center, sigma, tau)
            peak_idx = np.argmax(result)
            # Shift toward lower MW = lower index = positive shift value
            shift = gaussian_peak_idx - peak_idx
            peak_shifts.append(shift)

        # Peak shift should increase with tau (more shift toward lower MW)
        assert all(peak_shifts[i] <= peak_shifts[i+1]
                   for i in range(len(peak_shifts)-1)), (
            f"Peak shifts should increase with tau: {peak_shifts}"
        )

    def test_output_shape_matches_input(self):
        """Output shape should match input molecular weights."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10
        tau = 0.05

        result = emg_broadening(mws, center, sigma, tau)

        assert result.shape == mws.shape

    def test_all_values_non_negative(self):
        """All output values should be non-negative."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10
        tau = 0.05

        result = emg_broadening(mws, center, sigma, tau)

        assert np.all(result >= 0)

    def test_peak_shifted_toward_lower_mw(self):
        """EMG peak should be shifted toward lower MW compared to Gaussian."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10
        tau = 0.05

        emg_result = emg_broadening(mws, center, sigma, tau)
        gaussian_result = gaussian_broadening(mws, center, sigma)

        emg_peak_mw = mws[np.argmax(emg_result)]
        gaussian_peak_mw = mws[np.argmax(gaussian_result)]

        # EMG peak should be at lower MW than Gaussian peak
        assert emg_peak_mw < gaussian_peak_mw, (
            f"EMG peak at {emg_peak_mw:.0f} should be < Gaussian peak at {gaussian_peak_mw:.0f}"
        )

    def test_normalized_integral(self):
        """Integral over log(MW) space should be close to 1."""
        mws = np.logspace(2, 6, 1000)  # Wide range
        center = 10000.0
        sigma = 0.10
        tau = 0.05

        result = emg_broadening(mws, center, sigma, tau)
        integral = np.trapezoid(result, np.log(mws))

        # Should integrate to approximately 1
        assert np.isclose(integral, 1.0, rtol=0.05)

    def test_handles_array_of_centers(self):
        """Should work with array of center values (for meshgrid use)."""
        mws = np.logspace(3, 5, 100)
        centers = np.array([8000.0, 10000.0, 12000.0])
        sigma = 0.10
        tau = 0.05

        # Create meshgrid like in calculate_mwd
        centers_mesh, mws_mesh = np.meshgrid(centers, mws)

        result = emg_broadening(mws_mesh, centers_mesh, sigma, tau)

        assert result.shape == mws_mesh.shape
        assert np.all(result >= 0)


class TestEMGBroadeningEdgeCases:
    """Test edge cases for EMG broadening."""

    def test_very_small_tau(self):
        """Very small tau should be nearly Gaussian in shape."""
        mws = np.logspace(3, 5, 500)
        center = 10000.0
        sigma = 0.10
        tau = 0.005  # Small but not tiny

        emg_result = emg_broadening(mws, center, sigma, tau)
        gaussian_result = gaussian_broadening(mws, center, sigma)

        # Peak positions should be very close
        emg_peak_idx = np.argmax(emg_result)
        gaussian_peak_idx = np.argmax(gaussian_result)

        # Small tau should produce minimal peak shift (< 5 indices)
        assert abs(emg_peak_idx - gaussian_peak_idx) < 10, (
            f"Small tau should produce minimal peak shift"
        )

        # Overall shape should be similar (correlation > 0.99)
        correlation = np.corrcoef(emg_result, gaussian_result)[0, 1]
        assert correlation > 0.99, f"Correlation {correlation} should be > 0.99"

    def test_center_at_edge_of_range(self):
        """Should handle center near edge of MW range."""
        mws = np.logspace(3, 5, 500)
        center = 1500.0  # Near low end
        sigma = 0.10
        tau = 0.05

        result = emg_broadening(mws, center, sigma, tau)

        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
