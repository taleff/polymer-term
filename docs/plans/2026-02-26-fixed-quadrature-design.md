# Fixed Quadrature Optimization for fit_mwd

## Problem

The `fit_mwd` function is slow for high molecular weight polymers. When the peak molecular weight corresponds to ~1000 DP, fitting takes ~5 minutes versus ~30 seconds for ~100 DP samples.

The bottleneck is repeated Poisson distribution calculations during numerical integration. The current implementation uses `scipy.integrate.quad_vec` which adaptively evaluates the integrand 50-100 times per optimization iteration, computing `poisson.pmf(dps, nup)` across thousands of DP values each time.

## Solution

Replace adaptive quadrature with fixed Gauss-Legendre quadrature and precompute all Poisson distributions at the start of each optimization iteration.

### Key Insight

Within a single optimization iteration, the kinetic parameters `(alpha, init, order, init_mon, bn)` are fixed. This means the trajectory `nup(t)` is deterministic. We can:

1. Choose fixed quadrature points in advance
2. Compute `nup` at each point
3. Precompute all Poisson PMFs in a single vectorized operation
4. Perform integration via matrix multiplication

### Architecture

#### Quadrature Point Selection

Use Gauss-Legendre quadrature for the interval [0, T]:

```python
def get_quadrature_points(n_points: int, time_end: float):
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    scaled_nodes = (nodes + 1) * time_end / 2
    scaled_weights = weights * time_end / 2
    return scaled_nodes, scaled_weights
```

Gauss-Legendre is optimal for smooth functions and concentrates points near endpoints where the integrand changes faster.

#### Precomputation

```python
def precompute_poisson_distributions(
    times: np.ndarray,
    dps: np.ndarray,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    bn: float
) -> np.ndarray:
    """
    Precompute Poisson PMFs at all quadrature points.

    Returns shape (n_times, n_dps) matrix.
    """
    nups = np.array([
        living_chain_dp(alpha, init_mon, init, order, t, bn)
        for t in times
    ])
    return poisson.pmf(dps[np.newaxis, :], nups[:, np.newaxis])
```

#### Integration

```python
def compute_dead_fracs_fixed_quadrature(
    times: np.ndarray,
    weights: np.ndarray,
    poisson_matrix: np.ndarray,
    dps: np.ndarray,
    init: float,
    order: float
) -> np.ndarray:
    b_vals = np.array([living_chain_concentration(init, order, t) for t in times])
    integrand_weights = (b_vals ** order) * (init ** (1 - order))
    return (weights * integrand_weights) @ poisson_matrix
```

#### Sparse DP Range (Optional Enhancement)

Only compute Poisson PMFs for DPs with significant probability mass (threshold 1e-6):

```python
def get_significant_dp_range(nup_min: float, nup_max: float, threshold: float = 1e-6):
    dp_min = max(1, int(poisson.ppf(threshold, nup_min)))
    dp_max = int(poisson.ppf(1 - threshold, nup_max)) + 1
    return dp_min, dp_max
```

## Performance Analysis

### Current Cost (nup=1000, 3000 DPs)

- `quad_vec` calls integrand ~50-100 times adaptively
- Each call: `poisson.pmf(dps, nup)` on 3000 elements
- Total: ~150,000-300,000 Poisson evaluations per objective call

### Proposed Cost

- Precompute: 50 time points x ~300 significant DPs = 15,000 Poisson evaluations
- Integration: Matrix multiplication (50 x 300) @ (300,)
- Total: ~15,000 Poisson evaluations + cheap linear algebra

### Expected Speedup

10-20x for high molecular weight samples.

## Implementation Plan

### Files to Modify

1. `polyterm/models/fitting.py`:
   - Add `_get_quadrature_points()` helper
   - Add `_precompute_poisson_matrix()` helper
   - Add `_compute_dead_fracs_quadrature()` helper
   - Modify `_calculate_mwd_internal()` to use fixed quadrature
   - Add `n_quadrature_points` parameter to `fit_mwd()` (default: 50)

2. `polyterm/core/distributions.py`:
   - Add `get_significant_dp_range()` helper for sparse optimization

### Testing Strategy

1. Accuracy test: Compare fixed quadrature vs `quad_vec` on reference cases
2. Performance test: Benchmark time-to-convergence for high DP samples
3. Edge cases: Very low conversion, very high conversion, different orders

## Decisions

- Probability threshold: 1e-6 (captures >99.9999% of distribution mass)
- Scope: `fit_mwd` only (not `calculate_mwd`, `estimate_living_fraction`, etc.)
- Default quadrature points: 50 (configurable)
