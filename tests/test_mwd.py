"""
Tests for MolecularWeightDistribution class.
"""

import pytest
import numpy as np
from polyterm import MolecularWeightDistribution


class TestMWDCreation:
    """Test MWD object creation."""

    def test_from_data(self, simple_mws):
        """Test creating MWD from data."""
        intensities = np.random.random(len(simple_mws))

        mwd = MolecularWeightDistribution.from_data(
            molecular_weights=simple_mws,
            intensities=intensities,
            monomer_mw=100.0,
            normalize=True
        )

        assert len(mwd.molecular_weights) == len(simple_mws)
        assert mwd.is_normalized
        assert mwd.monomer_mw == 100.0

    def test_from_kinetics(self, simple_mws, standard_params):
        """Test creating MWD from kinetic parameters."""
        mwd = MolecularWeightDistribution.from_kinetics(
            molecular_weights=simple_mws,
            **standard_params
        )

        assert len(mwd.molecular_weights) == len(simple_mws)
        assert mwd.is_normalized
        assert np.all(mwd.intensities >= 0)

    def test_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        mws = np.array([1000, 2000, 3000])
        ints = np.array([0.1, 0.5])  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            MolecularWeightDistribution.from_data(mws, ints, 100.0)

    def test_negative_mw_rejected(self):
        """Test that negative MWs are rejected."""
        mws = np.array([1000, -2000, 3000])
        ints = np.array([0.1, 0.5, 0.4])

        with pytest.raises(ValueError, match="must be positive"):
            MolecularWeightDistribution.from_data(mws, ints, 100.0)


class TestMWDProperties:
    """Test MWD property calculations."""

    def test_number_average_calculation(self, synthetic_mwd):
        """Test number average molecular weight calculation."""
        mn = synthetic_mwd.number_average_mw

        assert mn > 0
        assert np.isfinite(mn)

    def test_weight_average_calculation(self, synthetic_mwd):
        """Test weight average molecular weight calculation."""
        mw = synthetic_mwd.weight_average_mw

        assert mw > 0
        assert np.isfinite(mw)

    def test_dispersity_greater_than_one(self, synthetic_mwd):
        """Test that dispersity is >= 1 (Mw >= Mn)."""
        dispersity = synthetic_mwd.dispersity

        assert dispersity >= 1.0

    def test_peak_mw(self, synthetic_mwd):
        """Test peak molecular weight determination."""
        peak_mw = synthetic_mwd.peak_molecular_weight

        # Peak should be within the MW range
        assert (np.min(synthetic_mwd.molecular_weights) <=
                peak_mw <=
                np.max(synthetic_mwd.molecular_weights))

    def test_narrow_distribution_low_dispersity(self):
        """Test that narrow distributions have low dispersity."""
        mws = np.logspace(3, 6, 500)

        # Very narrow distribution (low alpha)
        mwd_narrow = MolecularWeightDistribution.from_kinetics(
            molecular_weights=mws,
            monomer_mw=100.0,
            nu=100.0,
            alpha=0.001,  # Very low termination
            init_mon=1.0,
            init=0.01,
            order=1.5,
            sigma=0.03
        )

        # Broader distribution (higher alpha)
        mwd_broad = MolecularWeightDistribution.from_kinetics(
            molecular_weights=mws,
            monomer_mw=100.0,
            nu=100.0,
            alpha=0.1,  # Higher termination
            init_mon=1.0,
            init=0.01,
            order=1.5,
            sigma=0.03
        )

        assert mwd_narrow.dispersity < mwd_broad.dispersity


class TestMWDMethods:
    """Test MWD manipulation methods."""

    def test_normalize(self, simple_mws):
        """Test normalization."""
        intensities = np.random.random(len(simple_mws)) + 1.0
        mwd = MolecularWeightDistribution.from_data(
            simple_mws, intensities, 100.0, normalize=False
        )

        mwd_norm = mwd.normalize()

        assert mwd_norm.is_normalized
        # Area should be approximately 1 (within numerical tolerance)
        area = np.trapezoid(mwd_norm.intensities, mwd_norm.molecular_weights)
        assert np.isclose(area, 1.0, rtol=0.01)

    def test_downsample(self, synthetic_mwd):
        """Test downsampling."""
        original_length = len(synthetic_mwd.molecular_weights)
        max_points = 100

        mwd_down = synthetic_mwd.downsample(max_points=max_points)

        assert len(mwd_down.molecular_weights) < original_length
        assert len(mwd_down.molecular_weights) <= max_points

    def test_downsample_no_change_if_small(self):
        """Test that downsampling doesn't change small datasets."""
        mws = np.logspace(3, 5, 50)  # Only 50 points
        ints = np.random.random(50)

        mwd = MolecularWeightDistribution.from_data(mws, ints, 100.0)
        mwd_down = mwd.downsample(max_points=100)

        assert len(mwd_down.molecular_weights) == len(mwd.molecular_weights)

    def test_normalize_on_log_scale(self, synthetic_mwd):
        """Test log-scale normalization."""
        mwd_log = synthetic_mwd.normalize_on_log_scale()

        # Area under curve on log scale should be 1
        log_mws = np.log(mwd_log.molecular_weights)
        area = np.trapezoid(mwd_log.intensities, log_mws)

        assert np.isclose(area, 1.0, rtol=0.01)
        assert mwd_log.is_normalized


class TestMWDImmutability:
    """Test that MWD objects are immutable."""

    def test_cannot_modify_attributes(self, synthetic_mwd):
        """Test that attributes cannot be modified."""
        with pytest.raises(Exception):  # dataclass frozen=True raises FrozenInstanceError
            synthetic_mwd.monomer_mw = 200.0



if __name__ == '__main__':
    pytest.main([__file__, '-v'])
