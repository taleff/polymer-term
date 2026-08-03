"""
Shared MWD computation functions.

This module contains the core functions for computing molecular weight
distributions from kinetic chain fractions. These are used by both
calculate_mwd and fit_mwd.
"""

import numpy as np
from scipy.stats import poisson

from ..kinetics.models import (
    find_chain_death_time,
    LIVING_CHAIN_CONC,
    LIVING_CHAIN_DP,
    CONVERSION_TO_TIME,
    CHAIN_DEATH_RATE,
    SECOND_ORDER_DEATH_RATE,
)

from .distributions import get_poisson_dp_range

__all__ = [
    'poisson_distribution',
    'conversion_to_reduced_time',
    'compute_dead_chain_fracs',
    'compute_live_chain_fracs',
    'compute_mwd_from_fracs',
    'compute_dead_chain_fraction',
]

# At conversions close to one define a maximum conversion close enough
# to one to avoid computational problems
CHAIN_DEATH_FRACTION = 0.9999


def poisson_distribution(dps, nup):
    """Default Poisson distribution for living chain length."""
    return poisson.pmf(dps, nup)


def conversion_to_reduced_time(kinetics, alpha, init_mon, init, order,
                               conversion, bn=1.0):
    """Convert monomer conversion to reduced time via the kinetics model.

    Handles the special case of full conversion by finding the time at
    which nearly all chains have terminated.
    """
    if np.isclose(conversion, 1):
        return find_chain_death_time(
            kinetics, alpha, init_mon, init, order,
            CHAIN_DEATH_FRACTION, bn
        )
    return kinetics[CONVERSION_TO_TIME](
        alpha, init_mon, init, order, conversion, bn
    )


def get_quadrature_points(n_points, time_end):
    """Get Gauss-Legendre quadrature nodes and weights scaled to [0, time_end]."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    scaled_nodes = (nodes + 1) * time_end / 2
    scaled_weights = weights * time_end / 2
    return scaled_nodes, scaled_weights


def compute_dead_chain_fracs(time, dps, alpha, init_mon, init, order,
                             bn, combination, n_quadrature_points, kinetics,
                             distribution=poisson_distribution):
    """Compute mole fractions of dead chains at each DP using quadrature."""
    quad_t, quad_w = get_quadrature_points(n_quadrature_points, time)

    # Compute kinetic chain length of the live chain distribution at each time
    nups = np.array([
        kinetics[LIVING_CHAIN_DP](alpha, init_mon, init, order, t, bn)
        for t in quad_t
    ])

    # Compute chain death rate at each time point
    integrand_weights = np.array([
        kinetics[CHAIN_DEATH_RATE](alpha, init_mon, init, order, t, bn)
        for t in quad_t
    ])

    # Check for invalid kinetics parameters
    if not np.all(np.isfinite(nups)) or not np.all(np.isfinite(integrand_weights)):
        raise ValueError(
            "Invalid kinetics parameters: NaN or Inf encountered in "
            "chain DP or death rate calculations. This typically occurs when "
            "the fitted alpha value is outside the valid range for the kinetics model."
        )

    # Split death rate into components for proper dead chain DP assignment.
    # First-order + disproportionation produce dead chains at Poisson(nup).
    # Combination produces dead chains at Poisson(2*nup) with halved count
    # (two radicals combine into one chain).
    if SECOND_ORDER_DEATH_RATE in kinetics:
        second_order_weights = np.array([
            kinetics[SECOND_ORDER_DEATH_RATE](
                alpha, init_mon, init, order, t, bn)
            for t in quad_t
        ])
    else:
        second_order_weights = integrand_weights

    # Weights for chains produced at nup (first-order + disproportionation)
    disp_weights = integrand_weights - combination * second_order_weights
    # Weights for chains produced at 2*nup (combination, halved count)
    comb_weights = combination * second_order_weights / 2

    # Build disproportionation distribution matrix (chains at nup)
    disp_idx = get_poisson_dp_range(np.max(nups), dps)
    disp_matrix = np.zeros((len(nups), len(dps)), dtype=float)
    disp_matrix[:, :disp_idx] = distribution(
        dps[:disp_idx][np.newaxis, :], nups[:, np.newaxis]
    )
    disp_fracs = (quad_w * disp_weights) @ disp_matrix

    if combination > 0.0:
        # Build combination distribution matrix (chains at 2*nup)
        comb_idx = get_poisson_dp_range(np.max(2 * nups), dps)
        comb_matrix = np.zeros((len(nups), len(dps)), dtype=float)
        comb_matrix[:, :comb_idx] = distribution(
            dps[:comb_idx][np.newaxis, :], (2 * nups)[:, np.newaxis]
        )
        dead_fracs = disp_fracs + (quad_w * comb_weights) @ comb_matrix
    else:
        dead_fracs = disp_fracs

    return dead_fracs


def compute_live_chain_fracs(time, dps, alpha, init_mon, init, order, bn,
                             kinetics, distribution=poisson_distribution):
    """Compute mole fractions of living chains at each DP."""
    b = kinetics[LIVING_CHAIN_CONC](alpha, init_mon, init, order, time, bn)
    nup = kinetics[LIVING_CHAIN_DP](alpha, init_mon, init, order, time, bn)

    idx_end = get_poisson_dp_range(nup, dps)
    result = np.zeros(len(dps), dtype=float)
    result[:idx_end] = distribution(dps[:idx_end], nup)

    return b * result


def compute_mwd_from_fracs(dead_fracs, live_fracs, dps, broadenings):
    """Compute broadened MWD from mole fraction distributions.

    Converts mole fractions to weight fractions, applies broadening,
    and normalizes.
    """
    raw_mwd = np.matmul(broadenings, (live_fracs + dead_fracs) * dps)

    norm = np.max(raw_mwd)
    if norm <= 0:
        zeros = np.zeros_like(raw_mwd)
        return zeros, zeros, zeros
    intensities = raw_mwd / norm
    dead_intensities = np.matmul(broadenings, dead_fracs * dps) / norm
    live_intensities = np.matmul(broadenings, live_fracs * dps) / norm

    return intensities, dead_intensities, live_intensities


def compute_dead_chain_fraction(dead_fracs, live_fracs):
    """Compute fraction of chains that are dead."""
    total = np.sum(live_fracs + dead_fracs)
    if total <= 0:
        return 0.0
    return np.sum(dead_fracs) / total
