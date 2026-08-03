"""
Tests for utility functions in polyterm.utils.
"""

import pytest
import numpy as np
from polyterm.core.utils import (
    calculate_number_average_dp,
    fit_right_edge,
    calculate_r_squared,
)


class TestCalculateNumberAverageDP:
  """Test number average degree of polymerization calculations."""

  def test_basic_calculation(self, simple_distribution_data):
    """Test basic number average DP calculation."""
    nu = calculate_number_average_dp(
      simple_distribution_data['mws'],
      simple_distribution_data['intensities'],
      simple_distribution_data['monomer_mw']
    )

    assert nu > 0
    assert 10 < nu < 50

  def test_narrow_distribution(self, narrow_distribution_data):
    """Test with a narrow distribution centered at known MW."""
    nu = calculate_number_average_dp(
      narrow_distribution_data['mws'],
      narrow_distribution_data['intensities'],
      narrow_distribution_data['monomer_mw']
    )

    assert np.isclose(nu, narrow_distribution_data['expected_nu'], rtol=0.05)

  def test_unnormalized_intensities(self):
    """Test that unnormalized intensities work correctly."""
    mws = np.array([1000, 2000, 3000])
    intensities = np.array([5, 10, 5])
    monomer_mw = 100.0

    nu = calculate_number_average_dp(mws, intensities, monomer_mw)

    assert nu > 0
    assert np.isfinite(nu)

  def test_single_peak(self):
    """Test with a single peak (delta-like function)."""
    mws = np.linspace(1000, 11000, 1000)
    intensities = np.zeros_like(mws)
    peak_idx = np.argmin(np.abs(mws - 5000))
    intensities[peak_idx-2:peak_idx+3] = [0.1, 0.2, 1.0, 0.2, 0.1]
    monomer_mw = 100.0

    nu = calculate_number_average_dp(mws, intensities, monomer_mw)

    assert np.isclose(nu, 50, rtol=0.1)


class TestFitRightEdge:
  """Test right edge fitting function."""

  def test_basic_fit(self, simple_mws, standard_params):
    """Test fitting a synthetic peak."""
    from polyterm import calculate_mwd

    mwd = calculate_mwd(
      simple_mws,
      standard_params['monomer_mw'],
      standard_params['init_mon'],
      standard_params['alpha'],
      standard_params['init'],
      standard_params['conversion'],
      standard_params['order'],
      standard_params['sigma']
    )

    nup, sigma = fit_right_edge(
      mwd.molecular_weights,
      mwd.intensities,
      standard_params['monomer_mw']
    )

    assert nup > 0
    assert sigma > 0
    assert 0.01 < sigma < 0.5

  def test_returns_two_values(self, gaussian_peak_data):
    """Test that function returns exactly two values."""
    result = fit_right_edge(
      gaussian_peak_data['mws'],
      gaussian_peak_data['intensities'],
      gaussian_peak_data['monomer_mw']
    )

    assert isinstance(result, tuple)
    assert len(result) == 2

  def test_with_narrow_peak(self, gaussian_peak_data):
    """Test with a very narrow peak."""
    nup, sigma = fit_right_edge(
      gaussian_peak_data['mws'],
      gaussian_peak_data['intensities'],
      gaussian_peak_data['monomer_mw']
    )

    assert 80 < nup < 120
    assert 0.01 < sigma < 0.2


class TestCalculateRSquared:
  """Test R-squared calculation."""

  def test_perfect_fit(self, r_squared_test_data):
    """Test R² for perfect fit."""
    observed = r_squared_test_data['perfect_observed']
    predicted = observed.copy()

    r_squared = calculate_r_squared(observed, predicted)

    assert np.isclose(r_squared, 1.0, rtol=1e-10)

  def test_good_fit(self, r_squared_test_data):
    """Test R² for a good but not perfect fit."""
    r_squared = calculate_r_squared(
      r_squared_test_data['good_observed'],
      r_squared_test_data['good_predicted']
    )

    assert 0.9 < r_squared < 1.0

  def test_poor_fit(self, r_squared_test_data):
    """Test R² for a poor fit."""
    r_squared = calculate_r_squared(
      r_squared_test_data['poor_observed'],
      r_squared_test_data['poor_predicted']
    )

    assert r_squared < 0

  def test_mean_prediction(self, r_squared_test_data):
    """Test R² when predicting the mean."""
    observed = r_squared_test_data['good_observed']
    predicted = np.full_like(observed, np.mean(observed))

    r_squared = calculate_r_squared(observed, predicted)

    assert np.isclose(r_squared, 0.0, atol=1e-10)

  def test_constant_observed_perfect_fit(self, r_squared_test_data):
    """Test edge case: constant observed with perfect prediction."""
    r_squared = calculate_r_squared(
      r_squared_test_data['constant_observed'],
      r_squared_test_data['constant_perfect_predicted']
    )

    assert r_squared == 1.0

  def test_constant_observed_imperfect_fit(self, r_squared_test_data):
    """Test edge case: constant observed with imperfect prediction."""
    r_squared = calculate_r_squared(
      r_squared_test_data['constant_observed'],
      r_squared_test_data['constant_imperfect_predicted']
    )

    assert r_squared == 0.0

  def test_with_realistic_data(self):
    """Test with realistic experimental-like data."""
    np.random.seed(42)
    observed = np.array([1.5, 2.3, 3.1, 4.2, 5.0, 5.8, 7.1, 8.2])
    predicted = observed + np.random.normal(0, 0.1, size=observed.shape)

    r_squared = calculate_r_squared(observed, predicted)

    assert 0.9 < r_squared < 1.0

  def test_large_arrays(self):
    """Test with large arrays (typical SEC data size)."""
    n_points = 10000
    np.random.seed(42)
    observed = np.random.randn(n_points) + 10
    predicted = observed + np.random.normal(0, 0.5, size=n_points)

    r_squared = calculate_r_squared(observed, predicted)

    assert 0 < r_squared < 1


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
