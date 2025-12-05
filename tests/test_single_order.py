"""
Tests for SingleOrderModel fitting.
"""

import pytest
import numpy as np
from polyterm import (
    MolecularWeightDistribution,
    SingleOrderModel,
)


class TestSingleOrderModel:
    """Test single order fitting model."""

    def test_model_creation(self):
        """Test creating a fitting model."""
        model = SingleOrderModel(
            monomer_mw=100.0,
            init_mon=1.0,
            order=1.5
        )

        assert model.monomer_mw == 100.0
        assert model.init_mon == 1.0
        assert model.order == 1.5

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            SingleOrderModel(monomer_mw=-100.0, init_mon=1.0, order=1.0)

        with pytest.raises(ValueError):
            SingleOrderModel(monomer_mw=100.0, init_mon=0.0, order=1.0)

        with pytest.raises(ValueError):
            SingleOrderModel(
                monomer_mw=100.0,
                init_mon=1.0,
                order=-1.0
            )

        with pytest.raises(ValueError):
            SingleOrderModel(
                monomer_mw=100.0,
                init_mon=1.0,
                order=1.0,
                conversion=1.5  # > 1
            )


class TestRoundTripFitting:
    """Test fitting synthetic data (round-trip tests)."""

    def test_fit_all_parameters(self, simple_mws, standard_params):
        """Test fitting with known order recovers parameters."""
        # Generate synthetic MWD with known parameters
        true_alpha = standard_params['alpha']
        true_init = standard_params['init']

        mwd_synthetic = MolecularWeightDistribution.from_kinetics(
            molecular_weights=simple_mws,
            **standard_params
        )

        # Fit with known order
        model = SingleOrderModel(
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            order=standard_params['order']
        )

        result = model.fit(mwd_synthetic)

        # Should recover parameters within reasonable tolerance
        # Note: When both init and conversion are unknown, the fitting problem
        # is more challenging due to parameter identifiability issues
        assert np.isclose(result.alpha, true_alpha, rtol=0.15)
        assert np.isclose(result.init, true_init, rtol=0.15)
        assert result.r_squared > 0.99

    def test_fit_with_known_initiator(self, simple_mws, standard_params):
        """Test fitting with known initiator."""
        # Calculate true conversion
        true_conv = (standard_params['init'] * standard_params['nu'] /
                     standard_params['init_mon'])

        mwd_synthetic = MolecularWeightDistribution.from_kinetics(
            molecular_weights=simple_mws,
            **standard_params
        )

        model = SingleOrderModel(
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            init=standard_params['init'],
            order=standard_params['order']
        )

        result = model.fit(mwd_synthetic)

        # Should recover parameters within reasonable tolerance
        assert np.isclose(result.conversion, true_conv, rtol=0.001)
        assert result.r_squared > 0.99

    def test_fit_with_known_conversion(self, simple_mws, standard_params):
        """Test fitting with known conversion."""
        # Calculate true conversion
        true_conv = (standard_params['init'] * standard_params['nu'] /
                     standard_params['init_mon'])

        mwd_synthetic = MolecularWeightDistribution.from_kinetics(
            molecular_weights=simple_mws,
            **standard_params
        )

        model = SingleOrderModel(
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            conversion=true_conv,
            order=standard_params['order']
        )

        result = model.fit(mwd_synthetic)

        # Should recover parameters within reasonable tolerance
        assert np.isclose(result.init, standard_params['init'], rtol=0.001)
        assert result.r_squared > 0.99


class TestFitResult:
    """Test FitResult object functionality."""

    def test_fit_result_attributes(self, synthetic_mwd):
        """Test that FitResult has all expected attributes."""
        model = SingleOrderModel(
            monomer_mw=104.15,
            init_mon=1.0,
            order=1.5
        )
        result = model.fit(synthetic_mwd)

        # Check all required attributes exist
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'init')
        assert hasattr(result, 'order')
        assert hasattr(result, 'sigma')
        assert hasattr(result, 'conversion')
        assert hasattr(result, 'r_squared')
        assert hasattr(result, 'molecular_weights')
        assert hasattr(result, 'predicted_intensities')
        assert hasattr(result, 'dead_chain_fraction')
        assert hasattr(result, 'fit_message')



class TestDownsampling:
    """Test that downsampling works correctly in fitting."""

    def test_fit_with_downsampling(self, simple_mws, standard_params):
        """Test that fitting works with downsampled data."""
        mwd = MolecularWeightDistribution.from_kinetics(
            molecular_weights=simple_mws,
            **standard_params
        )

        # Fit with different downsampling levels
        # Note: When both init and conversion are unknown, more points are needed
        # for stable fitting due to parameter identifiability issues
        model1 = SingleOrderModel(
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            order=standard_params['order'],
        )

        model2 = SingleOrderModel(
            monomer_mw=standard_params['monomer_mw'],
            init_mon=standard_params['init_mon'],
            order=standard_params['order'],
            max_fit_points=600
        )

        result1 = model1.fit(mwd)
        result2 = model2.fit(mwd)

        # Both should give reasonable fits
        assert result1.r_squared > 0.90
        assert result2.r_squared > 0.90

        # Parameters should be similar
        assert np.isclose(result1.alpha, result2.alpha, rtol=0.2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
