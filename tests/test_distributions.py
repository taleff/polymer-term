"""
Tests for distribution calculation functions.
"""

import pytest
import numpy as np
from polyterm.core.distributions import (
    living_distribution_integrand,
    calculate_distribution,
    calculate_mwd,
)
from polyterm.core.broadening import gaussian_broadening


class TestGaussianBroadening:
  """Test Gaussian broadening function."""

  def test_gaussian_properties(self, gaussian_test_data):
    """Test basic properties of Gaussian broadening."""
    result = gaussian_broadening(
      gaussian_test_data['mws'],
      gaussian_test_data['center'],
      gaussian_test_data['sigma']
    )

    # All values should be positive
    assert np.all(result > 0)
    # Peak should be near center
    peak_idx = np.argmax(result)
    assert np.isclose(gaussian_test_data['mws'][peak_idx], gaussian_test_data['center'], rtol=0.1)

  def test_peak_at_center(self, gaussian_test_data):
    """Test that peak is at the center."""
    result = gaussian_broadening(
      gaussian_test_data['mws'],
      gaussian_test_data['center'],
      gaussian_test_data['sigma']
    )

    peak_idx = np.argmax(result)
    peak_mw = gaussian_test_data['mws'][peak_idx]
    assert np.isclose(peak_mw, gaussian_test_data['center'], rtol=0.05)

  def test_positive_values(self, gaussian_test_data):
    """Test that all values are positive."""
    result = gaussian_broadening(
      gaussian_test_data['mws'],
      gaussian_test_data['center'],
      gaussian_test_data['sigma']
    )

    assert np.all(result > 0)

  def test_multiple_centers_sequentially(self):
    """Test with multiple centers applied sequentially."""
    mws = np.logspace(3, 5, 100)
    centers = [5000.0, 10000.0, 20000.0]
    sigma = 0.1

    for center in centers:
      result = gaussian_broadening(mws, center, sigma)
      assert result.shape == mws.shape
      assert np.all(result > 0)


class TestLivingDistributionIntegrand:
  """Test living chain distribution integrand function."""

  def test_basic_calculation(self, standard_params):
    """Test basic integrand calculation."""
    time = 1.0
    dps = np.arange(1, 200, dtype=int)

    result = living_distribution_integrand(
      time,
      dps,
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order']
    )

    assert result.shape == dps.shape
    assert np.all(result >= 0)

  def test_combination_termination(self, second_order_params):
    """Test with combination termination."""
    time = 1.0
    dps = np.arange(1, 200, dtype=int)

    result = living_distribution_integrand(
      time,
      dps,
      second_order_params['alpha'],
      second_order_params['init_mon'],
      second_order_params['init'],
      second_order_params['order'],
      combination=True
    )

    assert result.shape == dps.shape
    assert np.all(result >= 0)

  def test_decreases_with_time(self, standard_params):
    """Test that distribution decreases as chains terminate."""
    dps = np.arange(50, 150, dtype=int)
    time_early = 0.1
    time_late = 5.0

    result_early = living_distribution_integrand(
      time_early,
      dps,
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order']
    )

    result_late = living_distribution_integrand(
      time_late,
      dps,
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order']
    )

    # Total should decrease with time as chains terminate
    assert np.sum(result_late) < np.sum(result_early)


class TestCalculateDistribution:
  """Test calculate_distribution function."""

  def test_basic_distribution(self, standard_params):
    """Test basic distribution calculation."""
    dps = np.arange(1, 300, dtype=int)

    alive, dead = calculate_distribution(
      dps,
      standard_params['nu'],
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order']
    )

    assert alive.shape == dps.shape
    assert dead.shape == dps.shape
    assert np.all(alive >= 0)
    assert np.all(dead >= 0)

  def test_sum_equals_init(self, standard_params):
    """Test that distribution sums to initial initiator concentration."""
    # Use wide range to capture full distribution
    dps = np.arange(1, 800, dtype=int)

    alive, dead = calculate_distribution(
      dps,
      standard_params['nu'],
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order']
    )

    total = np.sum(alive) + np.sum(dead)
    # Distribution returns mole fractions that sum to init concentration
    assert np.isclose(total, standard_params['init'], rtol=0.05)

  def test_first_order_termination(self, first_order_params):
    """Test with first-order termination."""
    dps = np.arange(1, 1000, dtype=int)

    alive, dead = calculate_distribution(
      dps,
      first_order_params['nu'],
      first_order_params['alpha'],
      first_order_params['init_mon'],
      first_order_params['init'],
      first_order_params['order']
    )

    assert np.all(np.isfinite(alive))
    assert np.all(np.isfinite(dead))

  def test_second_order_termination(self, second_order_params):
    """Test with second-order termination."""
    dps = np.arange(1, 1000, dtype=int)

    alive, dead = calculate_distribution(
      dps,
      second_order_params['nu'],
      second_order_params['alpha'],
      second_order_params['init_mon'],
      second_order_params['init'],
      second_order_params['order']
    )

    assert np.all(np.isfinite(alive))
    assert np.all(np.isfinite(dead))

  def test_with_combination(self, second_order_params):
    """Test distribution with combination termination."""
    dps = np.arange(1, 1000, dtype=int)

    alive, dead = calculate_distribution(
      dps,
      second_order_params['nu'],
      second_order_params['alpha'],
      second_order_params['init_mon'],
      second_order_params['init'],
      second_order_params['order'],
      combination=True
    )

    assert np.all(np.isfinite(alive))
    assert np.all(np.isfinite(dead))


class TestCalculateMWD:
  """Test calculate_mwd function."""

  def test_basic_mwd(self, simple_mws, standard_params):
    """Test basic MWD calculation."""
    mwd = calculate_mwd(
      simple_mws,
      standard_params['monomer_mw'],
      standard_params['nu'],
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order'],
      standard_params['sigma']
    )

    assert mwd.shape == simple_mws.shape
    assert np.all(mwd >= 0)

  def test_normalized(self, simple_mws, standard_params):
    """Test that MWD is normalized."""
    mwd = calculate_mwd(
      simple_mws,
      standard_params['monomer_mw'],
      standard_params['nu'],
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order'],
      standard_params['sigma']
    )

    integral = np.trapezoid(mwd, simple_mws)
    assert np.isclose(integral, 1.0, rtol=0.01)

  def test_live_only_mode(self, simple_mws, standard_params):
    """Test MWD with live_only=True."""
    mwd_live = calculate_mwd(
      simple_mws,
      standard_params['monomer_mw'],
      standard_params['nu'],
      standard_params['alpha'],
      standard_params['init_mon'],
      standard_params['init'],
      standard_params['order'],
      standard_params['sigma'],
      live_only=True
    )

    assert mwd_live.shape == simple_mws.shape
    assert np.all(mwd_live >= 0)

  def test_first_order_mwd(self, simple_mws, first_order_params):
    """Test MWD with first-order termination."""
    mwd = calculate_mwd(
      simple_mws,
      first_order_params['monomer_mw'],
      first_order_params['nu'],
      first_order_params['alpha'],
      first_order_params['init_mon'],
      first_order_params['init'],
      first_order_params['order'],
      first_order_params['sigma']
    )

    assert np.all(np.isfinite(mwd))
    assert np.any(mwd > 0)

  def test_second_order_mwd(self, simple_mws, second_order_params):
    """Test MWD with second-order termination."""
    mwd = calculate_mwd(
      simple_mws,
      second_order_params['monomer_mw'],
      second_order_params['nu'],
      second_order_params['alpha'],
      second_order_params['init_mon'],
      second_order_params['init'],
      second_order_params['order'],
      second_order_params['sigma']
    )

    assert np.all(np.isfinite(mwd))
    assert np.any(mwd > 0)

  def test_combination_warning(self, simple_mws, standard_params):
    """Test that combination with order!=2 raises warning."""
    with pytest.warns(UserWarning, match="chemically unusual"):
      calculate_mwd(
        simple_mws,
        standard_params['monomer_mw'],
        standard_params['nu'],
        standard_params['alpha'],
        standard_params['init_mon'],
        standard_params['init'],
        standard_params['order'],  # order=1.0, not 2
        standard_params['sigma'],
        combination=True
      )


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
