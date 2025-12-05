"""
Shared pytest fixtures for all tests.
"""

import pytest
import numpy as np
from polyterm import MolecularWeightDistribution


@pytest.fixture
def simple_mws():
    """Simple array of molecular weights for basic testing."""
    return np.logspace(2, 5.3, 10000)  # 1000 to 100000


@pytest.fixture
def standard_params():
    """Standard kinetic parameters for MWD generation."""
    return {
        "monomer_mw": 100.0,
        "nu": 100.0,
        "alpha": 0.002,
        "init_mon": 1.0,
        "init": 0.005,
        "order": 1.0,
        "sigma": 0.12
    }


@pytest.fixture
def first_order_params():
    """First order kinetic parameters for MWD generation."""
    return {
        "monomer_mw": 100.15,
        "nu": 500.0,
        "alpha": 0.0005,
        "init_mon": 1.0,
        "init": 0.001,
        "order": 1.0,
        "sigma": 0.12
    }


@pytest.fixture
def second_order_params():
    """Second order kinetic parameters for MWD generation."""
    return {
        "monomer_mw": 100.15,
        "nu": 500.0,
        "alpha": 0.5,
        "init_mon": 1.0,
        "init": 0.001,
        "order": 2.0,
        "sigma": 0.12
    }



@pytest.fixture
def other_order_params():
    """Other order kinetic parameters for MWD generation."""
    return {
        "monomer_mw": 100.15,
        "nu": 500.0,
        "alpha": 0.0005,
        "init_mon": 1.0,
        "init": 0.001,
        "order": 1.1,
        "sigma": 0.12
    }


@pytest.fixture
def synthetic_mwd(simple_mws, standard_params):
    """A synthetic MWD object for testing properties and methods."""
    return MolecularWeightDistribution.from_kinetics(
        molecular_weights=simple_mws,
        **standard_params
    )


# Test data for utility function tests

@pytest.fixture
def simple_distribution_data():
    """Simple MW distribution for testing number average DP."""
    return {
        'mws': np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
        'intensities': np.array([0.1, 0.3, 0.4, 0.15, 0.05]),
        'monomer_mw': 100.0
    }


@pytest.fixture
def narrow_distribution_data():
    """Narrow distribution centered at known MW."""
    mws = np.linspace(9500, 10500, 100)
    center = 10000.0
    sigma = 100.0
    intensities = np.exp(-((mws - center) ** 2) / (2 * sigma ** 2))
    return {
        'mws': mws,
        'intensities': intensities,
        'monomer_mw': 100.0,
        'expected_nu': 100.0
    }


@pytest.fixture
def gaussian_peak_data():
    """Gaussian peak for testing edge fitting."""
    mws = np.logspace(3, 5, 1000)
    center = 10000.0
    width = 0.05
    intensities = np.exp(-((np.log(mws) - np.log(center)) ** 2) / (2 * width ** 2))
    return {
        'mws': mws,
        'intensities': intensities,
        'monomer_mw': 100.0,
        'expected_nup': 100.0,
        'center': center,
        'width': width
    }


@pytest.fixture
def r_squared_test_data():
    """Test arrays for R-squared calculations."""
    return {
        'perfect_observed': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'good_observed': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'good_predicted': np.array([1.1, 1.9, 3.1, 3.9, 5.0]),
        'poor_observed': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'poor_predicted': np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        'constant_observed': np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        'constant_perfect_predicted': np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        'constant_imperfect_predicted': np.array([4.0, 5.0, 6.0, 5.0, 5.0])
    }


@pytest.fixture
def gaussian_test_data():
    """Data for testing Gaussian broadening."""
    return {
        'mws': np.logspace(3, 5, 100),
        'center': 10000.0,
        'sigma': 0.1
    }
