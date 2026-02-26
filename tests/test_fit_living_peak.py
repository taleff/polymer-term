"""
Tests for fit_living_peak function.
"""

import pytest
import numpy as np
from polyterm import fit_living_peak, LivingPeakResult
from polyterm.core.broadening import gaussian_broadening, emg_broadening, egh_broadening


class TestFitLivingPeakValidation:
    """Test input validation for fit_living_peak."""

    def test_invalid_broadening_type(self):
        """Test that invalid broadening type raises ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones_like(mws)

        with pytest.raises(ValueError, match="broadening_type must be one of"):
            fit_living_peak(mws, ints, 'invalid', sigma=0.1)

    def test_negative_sigma_raises_error(self):
        """Test that negative sigma raises ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones_like(mws)

        with pytest.raises(ValueError, match="sigma must be positive"):
            fit_living_peak(mws, ints, 'gaussian', sigma=-0.1)

    def test_zero_sigma_raises_error(self):
        """Test that zero sigma raises ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones_like(mws)

        with pytest.raises(ValueError, match="sigma must be positive"):
            fit_living_peak(mws, ints, 'gaussian', sigma=0.0)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones(50)  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            fit_living_peak(mws, ints, 'gaussian', sigma=0.1)


class TestFitLivingPeakBasic:
    """Basic functionality tests for fit_living_peak."""

    @pytest.fixture
    def synthetic_living_peak(self):
        """Generate synthetic data with known living peak."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10
        tau = 0.0

        # Pure Gaussian living distribution
        intensities = gaussian_broadening(mws, center, sigma)

        return {
            'mws': mws,
            'intensities': intensities,
            'center': center,
            'sigma': sigma,
            'tau': tau
        }

    @pytest.fixture
    def synthetic_mixed_distribution(self):
        """Generate synthetic data with living and dead chains."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10

        # Living distribution (Gaussian at high MW)
        living = gaussian_broadening(mws, center, sigma)

        # Dead distribution (lower MW shoulder)
        dead_center = 20000.0
        dead = 0.3 * gaussian_broadening(mws, dead_center, sigma * 1.5)

        # Combined (weight fractions)
        total = living + dead
        total = total / np.trapezoid(total, mws)  # Normalize

        return {
            'mws': mws,
            'intensities': total,
            'living_center': center,
            'dead_center': dead_center,
            'sigma': sigma
        }

    def test_returns_living_peak_result(self, synthetic_living_peak):
        """Test that fit_living_peak returns a LivingPeakResult."""
        result = fit_living_peak(
            synthetic_living_peak['mws'],
            synthetic_living_peak['intensities'],
            'gaussian',
            sigma=synthetic_living_peak['sigma']
        )

        assert isinstance(result, LivingPeakResult)

    def test_result_has_all_attributes(self, synthetic_living_peak):
        """Test that result has all expected attributes."""
        result = fit_living_peak(
            synthetic_living_peak['mws'],
            synthetic_living_peak['intensities'],
            'gaussian',
            sigma=synthetic_living_peak['sigma']
        )

        assert hasattr(result, 'living_intensities')
        assert hasattr(result, 'dead_intensities')
        assert hasattr(result, 'dead_chain_fraction')
        assert hasattr(result, 'living_peak_mw')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'molecular_weights')
        assert hasattr(result, 'broadening_type')
        assert hasattr(result, 'sigma')
        assert hasattr(result, 'tau')
        assert hasattr(result, 'coefficient')
        assert hasattr(result, 'fit_message')

    def test_recovers_known_peak(self, synthetic_living_peak):
        """Test that fitting recovers the known peak position."""
        result = fit_living_peak(
            synthetic_living_peak['mws'],
            synthetic_living_peak['intensities'],
            'gaussian',
            sigma=synthetic_living_peak['sigma']
        )

        # Should recover center within 10%
        assert np.isclose(
            result.living_peak_mw,
            synthetic_living_peak['center'],
            rtol=0.1
        )

    def test_pure_living_has_low_dead_fraction(self, synthetic_living_peak):
        """Test that pure living distribution has low dead fraction."""
        result = fit_living_peak(
            synthetic_living_peak['mws'],
            synthetic_living_peak['intensities'],
            'gaussian',
            sigma=synthetic_living_peak['sigma']
        )

        # Pure living should have near-zero dead fraction
        assert result.dead_chain_fraction < 0.1

    def test_good_fit_quality(self, synthetic_living_peak):
        """Test that fit quality is high for synthetic data."""
        result = fit_living_peak(
            synthetic_living_peak['mws'],
            synthetic_living_peak['intensities'],
            'gaussian',
            sigma=synthetic_living_peak['sigma']
        )

        # Should have high R-squared
        assert result.r_squared > 0.95

    def test_mixed_distribution_detects_dead(self, synthetic_mixed_distribution):
        """Test that mixed distribution shows non-zero dead fraction."""
        result = fit_living_peak(
            synthetic_mixed_distribution['mws'],
            synthetic_mixed_distribution['intensities'],
            'gaussian',
            sigma=synthetic_mixed_distribution['sigma']
        )

        # Should detect dead chains
        assert result.dead_chain_fraction > 0.05

    def test_intensities_arrays_match_input_length(self, synthetic_living_peak):
        """Test that output arrays match input length."""
        result = fit_living_peak(
            synthetic_living_peak['mws'],
            synthetic_living_peak['intensities'],
            'gaussian',
            sigma=synthetic_living_peak['sigma']
        )

        n = len(synthetic_living_peak['mws'])
        assert len(result.living_intensities) == n
        assert len(result.dead_intensities) == n
        assert len(result.molecular_weights) == n


class TestFitLivingPeakBroadeningTypes:
    """Test fit_living_peak with different broadening types."""

    @pytest.fixture
    def egh_peak_data(self):
        """Generate synthetic data with EGH broadening."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10
        tau = 0.05

        intensities = egh_broadening(mws, center, sigma, tau)

        return {
            'mws': mws,
            'intensities': intensities,
            'center': center,
            'sigma': sigma,
            'tau': tau
        }

    @pytest.fixture
    def emg_peak_data(self):
        """Generate synthetic data with EMG broadening."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10
        tau = 0.05

        intensities = emg_broadening(mws, center, sigma, tau)

        return {
            'mws': mws,
            'intensities': intensities,
            'center': center,
            'sigma': sigma,
            'tau': tau
        }

    def test_gaussian_broadening_type(self):
        """Test fitting with Gaussian broadening."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10

        intensities = gaussian_broadening(mws, center, sigma)

        result = fit_living_peak(mws, intensities, 'gaussian', sigma=sigma)

        assert result.broadening_type == 'gaussian'
        assert result.tau == 0.0
        assert np.isclose(result.living_peak_mw, center, rtol=0.1)

    def test_egh_broadening_type(self, egh_peak_data):
        """Test fitting with EGH broadening."""
        result = fit_living_peak(
            egh_peak_data['mws'],
            egh_peak_data['intensities'],
            'egh',
            sigma=egh_peak_data['sigma'],
            tau=egh_peak_data['tau']
        )

        assert result.broadening_type == 'egh'
        assert result.tau == egh_peak_data['tau']
        assert result.r_squared > 0.90

    def test_emg_broadening_type(self, emg_peak_data):
        """Test fitting with EMG broadening."""
        result = fit_living_peak(
            emg_peak_data['mws'],
            emg_peak_data['intensities'],
            'emg',
            sigma=emg_peak_data['sigma'],
            tau=emg_peak_data['tau']
        )

        assert result.broadening_type == 'emg'
        assert result.tau == emg_peak_data['tau']
        assert result.r_squared > 0.90

    def test_gaussian_ignores_tau(self):
        """Test that Gaussian type ignores tau parameter."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10

        intensities = gaussian_broadening(mws, center, sigma)

        result = fit_living_peak(
            mws, intensities, 'gaussian',
            sigma=sigma, tau=0.05  # This should be ignored
        )

        assert result.tau == 0.0


class TestFitLivingPeakEdgeCases:
    """Test edge cases for fit_living_peak."""

    def test_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10

        intensities = gaussian_broadening(mws, center, sigma)

        # Shuffle the data
        shuffle_idx = np.random.permutation(len(mws))
        mws_shuffled = mws[shuffle_idx]
        ints_shuffled = intensities[shuffle_idx]

        result = fit_living_peak(
            mws_shuffled, ints_shuffled, 'gaussian', sigma=sigma
        )

        # Should still work and recover center
        assert np.isclose(result.living_peak_mw, center, rtol=0.15)

    def test_narrow_peak(self):
        """Test fitting a very narrow peak."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.03  # Very narrow

        intensities = gaussian_broadening(mws, center, sigma)

        result = fit_living_peak(mws, intensities, 'gaussian', sigma=sigma)

        assert result.r_squared > 0.90

    def test_wide_peak(self):
        """Test fitting a wide peak."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.25  # Wide

        intensities = gaussian_broadening(mws, center, sigma)

        result = fit_living_peak(mws, intensities, 'gaussian', sigma=sigma)

        assert result.r_squared > 0.90


class TestLivingPeakResultRepr:
    """Test LivingPeakResult string representation."""

    def test_repr_without_tau(self):
        """Test repr for Gaussian broadening (no tau)."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10

        intensities = gaussian_broadening(mws, center, sigma)
        result = fit_living_peak(mws, intensities, 'gaussian', sigma=sigma)

        repr_str = repr(result)
        assert 'LivingPeakResult' in repr_str
        assert 'living_peak_mw' in repr_str
        assert 'dead_chain_fraction' in repr_str
        assert 'sigma' in repr_str
        assert 'tau' not in repr_str  # tau=0 should not be shown

    def test_repr_with_tau(self):
        """Test repr for EGH broadening (with tau)."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10
        tau = 0.05

        intensities = egh_broadening(mws, center, sigma, tau)
        result = fit_living_peak(
            mws, intensities, 'egh', sigma=sigma, tau=tau
        )

        repr_str = repr(result)
        assert 'tau' in repr_str  # tau > 0 should be shown


class TestLivingPeakResultImmutability:
    """Test that LivingPeakResult is immutable."""

    def test_cannot_modify_attributes(self):
        """Test that attributes cannot be modified."""
        mws = np.logspace(3, 6, 500)
        center = 50000.0
        sigma = 0.10

        intensities = gaussian_broadening(mws, center, sigma)
        result = fit_living_peak(mws, intensities, 'gaussian', sigma=sigma)

        with pytest.raises(Exception):  # FrozenInstanceError
            result.living_peak_mw = 100000.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
