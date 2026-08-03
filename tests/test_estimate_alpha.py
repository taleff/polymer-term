"""
Tests for estimate_alpha module.

Tests the estimation of alpha (kt/kp) from conversion vs living Mn data.
"""

import pytest
import numpy as np

from polyterm.estimate_alpha import estimate_alpha, _predicted_living_dps
from polyterm.kinetics.models import (
    STANDARD_KINETICS,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
)
from polyterm.kinetics.romp import ROMP_FIRST_ORDER_KINETICS


# ====================== Fixtures ======================

@pytest.fixture
def standard_alpha_params():
    """Standard parameters for alpha estimation tests (order=1)."""
    return {
        "monomer_mw": 100.0,
        "init_mon": 1.0,
        "init": 0.01,
        "order": 1.0,
    }


@pytest.fixture
def synthetic_conversion_data(standard_alpha_params):
    """Generate synthetic living Mn vs conversion data from known alpha.

    Uses order=1 with alpha/init=0.1 so the kinetics stay well within
    the valid region across all tested conversions.
    """
    alpha_true = 0.001
    convs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    kinetics = STANDARD_KINETICS

    # Compute living Mn at each conversion using the kinetics model
    mns = np.empty(len(convs))
    for i, conv in enumerate(convs):
        time = kinetics[CONVERSION_TO_TIME](
            alpha_true, standard_alpha_params["init_mon"],
            standard_alpha_params["init"], standard_alpha_params["order"],
            conv, 1.0,
        )
        dp = kinetics[LIVING_CHAIN_DP](
            alpha_true, standard_alpha_params["init_mon"],
            standard_alpha_params["init"], standard_alpha_params["order"],
            time, 1.0,
        )
        mns[i] = dp * standard_alpha_params["monomer_mw"]

    return {
        "convs": convs,
        "living_mns": mns,
        "alpha_true": alpha_true,
        **standard_alpha_params,
    }


# ====================== Input Validation ======================

class TestEstimateAlphaValidation:
    """Test input validation for estimate_alpha."""

    def test_mismatched_lengths_raises(self, standard_alpha_params):
        """Raises ValueError when convs and living_mns have different lengths."""
        with pytest.raises(ValueError, match="Length mismatch"):
            estimate_alpha(
                convs=[0.1, 0.2, 0.3],
                living_mns=[100.0, 200.0],
                **standard_alpha_params,
            )

    def test_zero_monomer_mw_raises(self, standard_alpha_params):
        """Raises ValueError when monomer_mw is not positive."""
        params = {**standard_alpha_params, "monomer_mw": 0.0}
        with pytest.raises(ValueError, match="monomer_mw must be positive"):
            estimate_alpha(
                convs=[0.1, 0.2],
                living_mns=[100.0, 200.0],
                **params,
            )

    def test_negative_monomer_mw_raises(self, standard_alpha_params):
        """Raises ValueError when monomer_mw is negative."""
        params = {**standard_alpha_params, "monomer_mw": -10.0}
        with pytest.raises(ValueError, match="monomer_mw must be positive"):
            estimate_alpha(
                convs=[0.1, 0.2],
                living_mns=[100.0, 200.0],
                **params,
            )


# ====================== Return Structure ======================

class TestEstimateAlphaReturnStructure:
    """Test that estimate_alpha returns the expected structure."""

    def test_returns_dict_with_required_keys(self, synthetic_conversion_data):
        """Result dict contains alpha, predicted_mns, and r_squared."""
        result = estimate_alpha(
            convs=synthetic_conversion_data["convs"],
            living_mns=synthetic_conversion_data["living_mns"],
            monomer_mw=synthetic_conversion_data["monomer_mw"],
            init_mon=synthetic_conversion_data["init_mon"],
            init=synthetic_conversion_data["init"],
            order=synthetic_conversion_data["order"],
        )
        assert "alpha" in result
        assert "predicted_mns" in result
        assert "r_squared" in result

    def test_predicted_mns_same_length_as_input(self, synthetic_conversion_data):
        """predicted_mns array has same length as input conversions."""
        result = estimate_alpha(
            convs=synthetic_conversion_data["convs"],
            living_mns=synthetic_conversion_data["living_mns"],
            monomer_mw=synthetic_conversion_data["monomer_mw"],
            init_mon=synthetic_conversion_data["init_mon"],
            init=synthetic_conversion_data["init"],
            order=synthetic_conversion_data["order"],
        )
        assert len(result["predicted_mns"]) == len(synthetic_conversion_data["convs"])

    def test_alpha_is_positive(self, synthetic_conversion_data):
        """Fitted alpha must be positive."""
        result = estimate_alpha(
            convs=synthetic_conversion_data["convs"],
            living_mns=synthetic_conversion_data["living_mns"],
            monomer_mw=synthetic_conversion_data["monomer_mw"],
            init_mon=synthetic_conversion_data["init_mon"],
            init=synthetic_conversion_data["init"],
            order=synthetic_conversion_data["order"],
        )
        assert result["alpha"] > 0


# ====================== Recovery of Known Alpha ======================

class TestEstimateAlphaRecovery:
    """Test that estimate_alpha recovers known alpha values."""

    def test_recovers_alpha_from_exact_data(self, synthetic_conversion_data):
        """Recovers the true alpha from noiseless synthetic data."""
        result = estimate_alpha(
            convs=synthetic_conversion_data["convs"],
            living_mns=synthetic_conversion_data["living_mns"],
            monomer_mw=synthetic_conversion_data["monomer_mw"],
            init_mon=synthetic_conversion_data["init_mon"],
            init=synthetic_conversion_data["init"],
            order=synthetic_conversion_data["order"],
        )
        assert np.isclose(result["alpha"], synthetic_conversion_data["alpha_true"],
                          rtol=1e-3)

    def test_r_squared_near_one_for_exact_data(self, synthetic_conversion_data):
        """R-squared should be ~1.0 when fitting noiseless data."""
        result = estimate_alpha(
            convs=synthetic_conversion_data["convs"],
            living_mns=synthetic_conversion_data["living_mns"],
            monomer_mw=synthetic_conversion_data["monomer_mw"],
            init_mon=synthetic_conversion_data["init_mon"],
            init=synthetic_conversion_data["init"],
            order=synthetic_conversion_data["order"],
        )
        assert result["r_squared"] > 0.999

    def test_recovers_alpha_with_second_order(self):
        """Recovers alpha with second-order termination."""
        alpha_true = 0.5
        monomer_mw = 100.0
        init_mon = 1.0
        init = 0.01
        order = 2.0
        kinetics = STANDARD_KINETICS
        convs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        mns = np.empty(len(convs))
        for i, conv in enumerate(convs):
            time = kinetics[CONVERSION_TO_TIME](
                alpha_true, init_mon, init, order, conv, 1.0
            )
            dp = kinetics[LIVING_CHAIN_DP](
                alpha_true, init_mon, init, order, time, 1.0
            )
            mns[i] = dp * monomer_mw

        result = estimate_alpha(
            convs=convs, living_mns=mns,
            monomer_mw=monomer_mw, init_mon=init_mon,
            init=init, order=order,
        )
        assert np.isclose(result["alpha"], alpha_true, rtol=1e-3)

    def test_recovers_alpha_with_fractional_order(self):
        """Recovers alpha with 1.5 order termination."""
        alpha_true = 0.01
        monomer_mw = 100.0
        init_mon = 1.0
        init = 0.005
        order = 1.5
        kinetics = STANDARD_KINETICS
        convs = np.linspace(0.1, 0.7, 10)

        mns = np.empty(len(convs))
        for i, conv in enumerate(convs):
            time = kinetics[CONVERSION_TO_TIME](
                alpha_true, init_mon, init, order, conv, 1.0
            )
            dp = kinetics[LIVING_CHAIN_DP](
                alpha_true, init_mon, init, order, time, 1.0
            )
            mns[i] = dp * monomer_mw

        result = estimate_alpha(
            convs=convs, living_mns=mns,
            monomer_mw=monomer_mw, init_mon=init_mon,
            init=init, order=order,
        )
        assert np.isclose(result["alpha"], alpha_true, rtol=1e-3)



# ====================== Custom Kinetics ======================

class TestEstimateAlphaCustomKinetics:
    """Test estimate_alpha with non-default kinetics models."""

    def test_with_romp_first_order_kinetics(self):
        """Recovers alpha using ROMP first-order kinetics."""
        alpha_true = 0.01
        monomer_mw = 100.0
        init_mon = 1.0
        init = 0.1
        order = 1.0
        kinetics = ROMP_FIRST_ORDER_KINETICS
        convs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        mns = np.empty(len(convs))
        for i, conv in enumerate(convs):
            time = kinetics[CONVERSION_TO_TIME](
                alpha_true, init_mon, init, order, conv, 1.0
            )
            dp = kinetics[LIVING_CHAIN_DP](
                alpha_true, init_mon, init, order, time, 1.0
            )
            mns[i] = dp * monomer_mw

        result = estimate_alpha(
            convs=convs, living_mns=mns,
            monomer_mw=monomer_mw, init_mon=init_mon,
            init=init, order=order,
            kinetics=kinetics,
        )
        assert np.isclose(result["alpha"], alpha_true, rtol=1e-3)

    def test_default_kinetics_is_standard(self, synthetic_conversion_data):
        """Passing kinetics=None uses STANDARD_KINETICS (same as default)."""
        result_default = estimate_alpha(
            convs=synthetic_conversion_data["convs"],
            living_mns=synthetic_conversion_data["living_mns"],
            monomer_mw=synthetic_conversion_data["monomer_mw"],
            init_mon=synthetic_conversion_data["init_mon"],
            init=synthetic_conversion_data["init"],
            order=synthetic_conversion_data["order"],
        )
        result_explicit = estimate_alpha(
            convs=synthetic_conversion_data["convs"],
            living_mns=synthetic_conversion_data["living_mns"],
            monomer_mw=synthetic_conversion_data["monomer_mw"],
            init_mon=synthetic_conversion_data["init_mon"],
            init=synthetic_conversion_data["init"],
            order=synthetic_conversion_data["order"],
            kinetics=STANDARD_KINETICS,
        )
        assert np.isclose(result_default["alpha"], result_explicit["alpha"], rtol=1e-10)


# ====================== Helper Function ======================

class TestPredictedLivingDps:
    """Test the _predicted_living_dps helper."""

    def test_returns_correct_length(self):
        """Output array has same length as input conversions."""
        convs = np.array([0.1, 0.2, 0.3])
        kinetics = STANDARD_KINETICS
        dps = _predicted_living_dps(
            alpha=0.005, convs=convs,
            init_mon=1.0, init=0.01, order=1.0, bn=1.0,
            kinetics=kinetics,
        )
        assert len(dps) == len(convs)

    def test_dp_increases_with_conversion(self):
        """Living chain DP should increase with conversion."""
        convs = np.array([0.1, 0.3, 0.5, 0.7])
        kinetics = STANDARD_KINETICS
        dps = _predicted_living_dps(
            alpha=0.005, convs=convs,
            init_mon=1.0, init=0.01, order=1.0, bn=1.0,
            kinetics=kinetics,
        )
        assert np.all(np.diff(dps) > 0)
