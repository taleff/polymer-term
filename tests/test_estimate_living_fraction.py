"""
Tests for estimate_living_fraction function.

This function estimates living chain fraction using a Poisson-broadened
distribution model, which accounts for the intrinsic width of living
chain distributions at low DP.
"""

import pytest
import numpy as np
from scipy.stats import poisson
from polyterm import estimate_living_fraction, LivingFractionResult
from polyterm.core.broadening import egh_broadening, gaussian_broadening


class TestEstimateLivingFractionValidation:
    """Test input validation for estimate_living_fraction."""

    def test_negative_sigma_raises_error(self):
        """Test that negative sigma raises ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones_like(mws)

        with pytest.raises(ValueError, match="sigma must be positive"):
            estimate_living_fraction(mws, ints, sigma=-0.1, tau=0.0, monomer_mw=100)

    def test_zero_sigma_raises_error(self):
        """Test that zero sigma raises ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones_like(mws)

        with pytest.raises(ValueError, match="sigma must be positive"):
            estimate_living_fraction(mws, ints, sigma=0.0, tau=0.0, monomer_mw=100)

    def test_negative_monomer_mw_raises_error(self):
        """Test that negative monomer_mw raises ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones_like(mws)

        with pytest.raises(ValueError, match="monomer_mw must be positive"):
            estimate_living_fraction(mws, ints, sigma=0.1, tau=0.0, monomer_mw=-100)

    def test_zero_monomer_mw_raises_error(self):
        """Test that zero monomer_mw raises ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones_like(mws)

        with pytest.raises(ValueError, match="monomer_mw must be positive"):
            estimate_living_fraction(mws, ints, sigma=0.1, tau=0.0, monomer_mw=0)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        mws = np.logspace(3, 5, 100)
        ints = np.ones(50)  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            estimate_living_fraction(mws, ints, sigma=0.1, tau=0.0, monomer_mw=100)


class TestEstimateLivingFractionBasic:
    """Basic functionality tests for estimate_living_fraction."""

    @pytest.fixture
    def synthetic_poisson_peak(self):
        """Generate synthetic data with known Poisson living peak."""
        mws = np.logspace(3, 6, 500)
        monomer_mw = 100.0
        center_dp = 500.0  # DP = 500, MW = 50000
        sigma = 0.10
        tau = 0.0

        # Generate Poisson distribution of DPs
        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)
        mole_fracs = poisson.pmf(dps, center_dp)
        mass_fracs = mole_fracs * dps

        # Apply broadening
        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = egh_broadening(mws_mesh, true_mws_mesh, sigma, tau)
        intensities = broadening_matrix @ mass_fracs

        # Normalize
        intensities = intensities / np.trapezoid(intensities, mws)

        return {
            'mws': mws,
            'intensities': intensities,
            'center_dp': center_dp,
            'center_mw': center_dp * monomer_mw,
            'sigma': sigma,
            'tau': tau,
            'monomer_mw': monomer_mw
        }

    @pytest.fixture
    def synthetic_mixed_distribution(self):
        """Generate synthetic data with living and dead chains."""
        mws = np.logspace(3, 6, 500)
        monomer_mw = 100.0
        living_dp = 500.0  # Living chains at DP = 500
        dead_dp = 200.0    # Dead chains at lower DP
        sigma = 0.10
        tau = 0.05

        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)

        # Living chains (Poisson)
        living_mole_fracs = poisson.pmf(dps, living_dp)
        living_mass_fracs = living_mole_fracs * dps

        # Dead chains (broader, lower MW)
        dead_mole_fracs = poisson.pmf(dps, dead_dp) * 0.3
        dead_mass_fracs = dead_mole_fracs * dps

        # Apply broadening
        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = egh_broadening(mws_mesh, true_mws_mesh, sigma, tau)

        living_dist = broadening_matrix @ living_mass_fracs
        dead_dist = broadening_matrix @ dead_mass_fracs
        total = living_dist + dead_dist

        # Normalize
        total = total / np.trapezoid(total, mws)

        return {
            'mws': mws,
            'intensities': total,
            'living_dp': living_dp,
            'dead_dp': dead_dp,
            'sigma': sigma,
            'tau': tau,
            'monomer_mw': monomer_mw
        }

    def test_returns_living_fraction_result(self, synthetic_poisson_peak):
        """Test that estimate_living_fraction returns a LivingFractionResult."""
        result = estimate_living_fraction(
            synthetic_poisson_peak['mws'],
            synthetic_poisson_peak['intensities'],
            sigma=synthetic_poisson_peak['sigma'],
            tau=synthetic_poisson_peak['tau'],
            monomer_mw=synthetic_poisson_peak['monomer_mw']
        )

        assert isinstance(result, LivingFractionResult)

    def test_result_has_all_attributes(self, synthetic_poisson_peak):
        """Test that result has all expected attributes."""
        result = estimate_living_fraction(
            synthetic_poisson_peak['mws'],
            synthetic_poisson_peak['intensities'],
            sigma=synthetic_poisson_peak['sigma'],
            tau=synthetic_poisson_peak['tau'],
            monomer_mw=synthetic_poisson_peak['monomer_mw']
        )

        assert hasattr(result, 'living_peak_mw')
        assert hasattr(result, 'dead_chain_fraction')
        assert hasattr(result, 'living_distribution')
        assert hasattr(result, 'dead_distribution')
        assert hasattr(result, 'molecular_weights')
        assert hasattr(result, 'sigma')
        assert hasattr(result, 'tau')
        assert hasattr(result, 'monomer_mw')
        assert hasattr(result, 'living_peak_dp')
        assert hasattr(result, 'coefficient')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'fit_message')

    def test_recovers_known_dp(self, synthetic_poisson_peak):
        """Test that fitting recovers the known DP position."""
        result = estimate_living_fraction(
            synthetic_poisson_peak['mws'],
            synthetic_poisson_peak['intensities'],
            sigma=synthetic_poisson_peak['sigma'],
            tau=synthetic_poisson_peak['tau'],
            monomer_mw=synthetic_poisson_peak['monomer_mw']
        )

        # Should recover center DP within 10%
        assert np.isclose(
            result.living_peak_dp,
            synthetic_poisson_peak['center_dp'],
            rtol=0.1
        )

    def test_recovers_known_mw(self, synthetic_poisson_peak):
        """Test that fitting recovers the known MW position."""
        result = estimate_living_fraction(
            synthetic_poisson_peak['mws'],
            synthetic_poisson_peak['intensities'],
            sigma=synthetic_poisson_peak['sigma'],
            tau=synthetic_poisson_peak['tau'],
            monomer_mw=synthetic_poisson_peak['monomer_mw']
        )

        # Should recover center MW within 10%
        assert np.isclose(
            result.living_peak_mw,
            synthetic_poisson_peak['center_mw'],
            rtol=0.1
        )

    def test_pure_living_has_low_dead_fraction(self, synthetic_poisson_peak):
        """Test that pure living distribution has low dead fraction."""
        result = estimate_living_fraction(
            synthetic_poisson_peak['mws'],
            synthetic_poisson_peak['intensities'],
            sigma=synthetic_poisson_peak['sigma'],
            tau=synthetic_poisson_peak['tau'],
            monomer_mw=synthetic_poisson_peak['monomer_mw']
        )

        # Pure living should have near-zero dead fraction
        assert result.dead_chain_fraction < 0.1

    def test_good_fit_quality(self, synthetic_poisson_peak):
        """Test that fit quality is high for synthetic data."""
        result = estimate_living_fraction(
            synthetic_poisson_peak['mws'],
            synthetic_poisson_peak['intensities'],
            sigma=synthetic_poisson_peak['sigma'],
            tau=synthetic_poisson_peak['tau'],
            monomer_mw=synthetic_poisson_peak['monomer_mw']
        )

        # Should have high R-squared
        assert result.r_squared > 0.95

    def test_mixed_distribution_detects_dead(self, synthetic_mixed_distribution):
        """Test that mixed distribution shows non-zero dead fraction."""
        result = estimate_living_fraction(
            synthetic_mixed_distribution['mws'],
            synthetic_mixed_distribution['intensities'],
            sigma=synthetic_mixed_distribution['sigma'],
            tau=synthetic_mixed_distribution['tau'],
            monomer_mw=synthetic_mixed_distribution['monomer_mw']
        )

        # Should detect dead chains
        assert result.dead_chain_fraction > 0.05

    def test_distribution_arrays_match_input_length(self, synthetic_poisson_peak):
        """Test that output arrays match input length."""
        result = estimate_living_fraction(
            synthetic_poisson_peak['mws'],
            synthetic_poisson_peak['intensities'],
            sigma=synthetic_poisson_peak['sigma'],
            tau=synthetic_poisson_peak['tau'],
            monomer_mw=synthetic_poisson_peak['monomer_mw']
        )

        n = len(synthetic_poisson_peak['mws'])
        assert len(result.living_distribution) == n
        assert len(result.dead_distribution) == n
        assert len(result.molecular_weights) == n


class TestEstimateLivingFractionLowDP:
    """Test estimate_living_fraction at low DP where Poisson width matters."""

    @pytest.fixture
    def low_dp_distribution(self):
        """Generate synthetic data at low DP where Poisson width is significant."""
        mws = np.logspace(2, 5, 500)
        monomer_mw = 100.0
        center_dp = 20.0  # Low DP, Poisson std dev ~ sqrt(20) ~ 4.5
        sigma = 0.10
        tau = 0.05

        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)
        mole_fracs = poisson.pmf(dps, center_dp)
        mass_fracs = mole_fracs * dps

        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = egh_broadening(mws_mesh, true_mws_mesh, sigma, tau)
        intensities = broadening_matrix @ mass_fracs
        intensities = intensities / np.trapezoid(intensities, mws)

        return {
            'mws': mws,
            'intensities': intensities,
            'center_dp': center_dp,
            'sigma': sigma,
            'tau': tau,
            'monomer_mw': monomer_mw
        }

    def test_recovers_low_dp(self, low_dp_distribution):
        """Test that fitting recovers low DP correctly."""
        result = estimate_living_fraction(
            low_dp_distribution['mws'],
            low_dp_distribution['intensities'],
            sigma=low_dp_distribution['sigma'],
            tau=low_dp_distribution['tau'],
            monomer_mw=low_dp_distribution['monomer_mw']
        )

        # Should recover center DP within 20% (harder at low DP)
        assert np.isclose(
            result.living_peak_dp,
            low_dp_distribution['center_dp'],
            rtol=0.2
        )

    def test_good_fit_at_low_dp(self, low_dp_distribution):
        """Test fit quality at low DP."""
        result = estimate_living_fraction(
            low_dp_distribution['mws'],
            low_dp_distribution['intensities'],
            sigma=low_dp_distribution['sigma'],
            tau=low_dp_distribution['tau'],
            monomer_mw=low_dp_distribution['monomer_mw']
        )

        # Should still have good R-squared
        assert result.r_squared > 0.90


class TestEstimateLivingFractionEGH:
    """Test estimate_living_fraction with EGH broadening."""

    @pytest.fixture
    def egh_peak_data(self):
        """Generate synthetic data with significant EGH tailing."""
        mws = np.logspace(3, 6, 500)
        monomer_mw = 100.0
        center_dp = 500.0
        sigma = 0.128
        tau = 0.0456  # Significant tailing

        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)
        mole_fracs = poisson.pmf(dps, center_dp)
        mass_fracs = mole_fracs * dps

        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = egh_broadening(mws_mesh, true_mws_mesh, sigma, tau)
        intensities = broadening_matrix @ mass_fracs
        intensities = intensities / np.trapezoid(intensities, mws)

        return {
            'mws': mws,
            'intensities': intensities,
            'center_dp': center_dp,
            'sigma': sigma,
            'tau': tau,
            'monomer_mw': monomer_mw
        }

    def test_egh_recovers_dp(self, egh_peak_data):
        """Test that EGH broadening recovers DP correctly."""
        result = estimate_living_fraction(
            egh_peak_data['mws'],
            egh_peak_data['intensities'],
            sigma=egh_peak_data['sigma'],
            tau=egh_peak_data['tau'],
            monomer_mw=egh_peak_data['monomer_mw']
        )

        assert np.isclose(
            result.living_peak_dp,
            egh_peak_data['center_dp'],
            rtol=0.1
        )

    def test_stores_tau(self, egh_peak_data):
        """Test that tau is stored in result."""
        result = estimate_living_fraction(
            egh_peak_data['mws'],
            egh_peak_data['intensities'],
            sigma=egh_peak_data['sigma'],
            tau=egh_peak_data['tau'],
            monomer_mw=egh_peak_data['monomer_mw']
        )

        assert result.tau == egh_peak_data['tau']


class TestEstimateLivingFractionEdgeCases:
    """Test edge cases for estimate_living_fraction."""

    def test_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        mws = np.logspace(3, 6, 500)
        monomer_mw = 100.0
        center_dp = 500.0
        sigma = 0.10
        tau = 0.0

        # Generate distribution
        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)
        mole_fracs = poisson.pmf(dps, center_dp)
        mass_fracs = mole_fracs * dps

        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = gaussian_broadening(mws_mesh, true_mws_mesh, sigma)
        intensities = broadening_matrix @ mass_fracs
        intensities = intensities / np.trapezoid(intensities, mws)

        # Shuffle the data
        shuffle_idx = np.random.permutation(len(mws))
        mws_shuffled = mws[shuffle_idx]
        ints_shuffled = intensities[shuffle_idx]

        result = estimate_living_fraction(
            mws_shuffled, ints_shuffled,
            sigma=sigma, tau=tau, monomer_mw=monomer_mw
        )

        # Should still recover center DP
        assert np.isclose(result.living_peak_dp, center_dp, rtol=0.15)

    def test_stores_monomer_mw(self):
        """Test that monomer_mw is stored in result."""
        mws = np.logspace(3, 6, 500)
        monomer_mw = 104.15
        center_dp = 500.0
        sigma = 0.10

        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)
        mole_fracs = poisson.pmf(dps, center_dp)
        mass_fracs = mole_fracs * dps

        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = gaussian_broadening(mws_mesh, true_mws_mesh, sigma)
        intensities = broadening_matrix @ mass_fracs
        intensities = intensities / np.trapezoid(intensities, mws)

        result = estimate_living_fraction(
            mws, intensities,
            sigma=sigma, tau=0.0, monomer_mw=monomer_mw
        )

        assert result.monomer_mw == monomer_mw


class TestLivingFractionResultRepr:
    """Test LivingFractionResult string representation."""

    @pytest.fixture
    def sample_result(self):
        """Generate a sample result for repr testing."""
        mws = np.logspace(3, 6, 500)
        monomer_mw = 100.0
        center_dp = 500.0
        sigma = 0.10
        tau = 0.05

        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)
        mole_fracs = poisson.pmf(dps, center_dp)
        mass_fracs = mole_fracs * dps

        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = egh_broadening(mws_mesh, true_mws_mesh, sigma, tau)
        intensities = broadening_matrix @ mass_fracs
        intensities = intensities / np.trapezoid(intensities, mws)

        return estimate_living_fraction(
            mws, intensities,
            sigma=sigma, tau=tau, monomer_mw=monomer_mw
        )

    def test_repr_contains_key_info(self, sample_result):
        """Test that repr contains key information."""
        repr_str = repr(sample_result)

        assert 'LivingFractionResult' in repr_str
        assert 'living_peak_mw' in repr_str
        assert 'living_peak_dp' in repr_str
        assert 'dead_chain_fraction' in repr_str
        assert 'R^2' in repr_str
        assert 'sigma' in repr_str

    def test_repr_shows_tau_when_nonzero(self, sample_result):
        """Test that tau is shown when non-zero."""
        repr_str = repr(sample_result)
        assert 'tau' in repr_str


class TestLivingFractionResultImmutability:
    """Test that LivingFractionResult is immutable."""

    def test_cannot_modify_attributes(self):
        """Test that attributes cannot be modified."""
        mws = np.logspace(3, 6, 500)
        monomer_mw = 100.0
        center_dp = 500.0
        sigma = 0.10

        max_dp = int(np.max(mws) / monomer_mw) + 1
        dps = np.arange(1, max_dp, dtype=int)
        mole_fracs = poisson.pmf(dps, center_dp)
        mass_fracs = mole_fracs * dps

        dps_mesh, mws_mesh = np.meshgrid(dps, mws)
        true_mws_mesh = dps_mesh * monomer_mw
        broadening_matrix = gaussian_broadening(mws_mesh, true_mws_mesh, sigma)
        intensities = broadening_matrix @ mass_fracs
        intensities = intensities / np.trapezoid(intensities, mws)

        result = estimate_living_fraction(
            mws, intensities,
            sigma=sigma, tau=0.0, monomer_mw=monomer_mw
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.living_peak_mw = 100000.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
