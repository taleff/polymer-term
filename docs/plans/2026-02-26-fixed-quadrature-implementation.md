# Fixed Quadrature Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Speed up `fit_mwd` by replacing adaptive `quad_vec` with fixed Gauss-Legendre quadrature and precomputed Poisson distributions.

**Architecture:** Add helper functions for quadrature points and Poisson precomputation. Modify `_calculate_mwd_internal` to use matrix multiplication instead of `quad_vec`. Add `n_quadrature_points` parameter to `fit_mwd`.

**Tech Stack:** NumPy (polynomial.legendre.leggauss), SciPy (stats.poisson)

---

### Task 1: Add Quadrature Helper Function

**Files:**
- Modify: `polyterm/models/fitting.py` (add after line 35, before `__all__`)
- Test: `tests/test_fit_mwd.py`

**Step 1: Write the failing test**

Add to `tests/test_fit_mwd.py` after the imports (around line 11):

```python
from polyterm.models.fitting import _get_quadrature_points


class TestQuadratureHelper:
    """Test the quadrature helper function."""

    def test_returns_correct_shapes(self):
        """Test that quadrature returns correct array shapes."""
        n_points = 50
        time_end = 10.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        assert nodes.shape == (n_points,)
        assert weights.shape == (n_points,)

    def test_nodes_in_correct_range(self):
        """Test that nodes are scaled to [0, time_end]."""
        n_points = 50
        time_end = 10.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        assert np.all(nodes >= 0)
        assert np.all(nodes <= time_end)

    def test_weights_sum_to_interval(self):
        """Test that weights sum to the interval length."""
        n_points = 50
        time_end = 10.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        # For Gauss-Legendre, weights sum to interval length
        assert np.isclose(np.sum(weights), time_end, rtol=1e-10)

    def test_integrates_polynomial_exactly(self):
        """Test that quadrature integrates low-degree polynomials exactly."""
        n_points = 10
        time_end = 5.0
        nodes, weights = _get_quadrature_points(n_points, time_end)

        # Integrate x^2 from 0 to 5: should be 5^3/3 = 125/3
        f_vals = nodes ** 2
        integral = np.sum(weights * f_vals)
        expected = (time_end ** 3) / 3

        assert np.isclose(integral, expected, rtol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mwd.py::TestQuadratureHelper -v`
Expected: FAIL with "cannot import name '_get_quadrature_points'"

**Step 3: Write minimal implementation**

Add to `polyterm/models/fitting.py` after the imports (around line 40):

```python
def _get_quadrature_points(n_points: int, time_end: float) -> tuple:
    """
    Get Gauss-Legendre quadrature nodes and weights scaled to [0, time_end].

    Parameters
    ----------
    n_points : int
        Number of quadrature points.
    time_end : float
        Upper limit of integration interval [0, time_end].

    Returns
    -------
    nodes : ndarray
        Quadrature nodes in [0, time_end].
    weights : ndarray
        Corresponding quadrature weights.
    """
    # Get standard Gauss-Legendre points on [-1, 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_points)

    # Transform from [-1, 1] to [0, time_end]
    scaled_nodes = (nodes + 1) * time_end / 2
    scaled_weights = weights * time_end / 2

    return scaled_nodes, scaled_weights
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mwd.py::TestQuadratureHelper -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add polyterm/models/fitting.py tests/test_fit_mwd.py
git commit -m "feat: add Gauss-Legendre quadrature helper function"
```

---

### Task 2: Add Poisson Precomputation Function

**Files:**
- Modify: `polyterm/models/fitting.py`
- Test: `tests/test_fit_mwd.py`

**Step 1: Write the failing test**

Add to `tests/test_fit_mwd.py` after the `_get_quadrature_points` import:

```python
from polyterm.models.fitting import _get_quadrature_points, _precompute_poisson_matrix
```

Add new test class after `TestQuadratureHelper`:

```python
class TestPoissonPrecomputation:
    """Test the Poisson precomputation function."""

    def test_returns_correct_shape(self):
        """Test that precomputed matrix has correct shape."""
        times = np.array([0.1, 0.5, 1.0, 2.0])
        dps = np.arange(1, 100)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0

        matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        assert matrix.shape == (len(times), len(dps))

    def test_rows_sum_approximately_to_one(self):
        """Test that each row sums to approximately 1 (Poisson normalization)."""
        times = np.array([0.1, 0.5, 1.0])
        dps = np.arange(1, 500)  # Wide range to capture most probability mass
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0

        matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        # Each row should sum close to 1 (Poisson PMF)
        row_sums = np.sum(matrix, axis=1)
        assert np.all(row_sums > 0.99)  # Allow some truncation

    def test_values_are_nonnegative(self):
        """Test that all Poisson probabilities are non-negative."""
        times = np.array([0.1, 1.0, 5.0])
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0

        matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        assert np.all(matrix >= 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mwd.py::TestPoissonPrecomputation -v`
Expected: FAIL with "cannot import name '_precompute_poisson_matrix'"

**Step 3: Write minimal implementation**

Add to `polyterm/models/fitting.py` after `_get_quadrature_points`:

```python
def _precompute_poisson_matrix(
    times: np.ndarray,
    dps: np.ndarray,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    bn: float
) -> np.ndarray:
    """
    Precompute Poisson PMFs at all quadrature time points.

    Parameters
    ----------
    times : ndarray
        Quadrature time points.
    dps : ndarray
        Degrees of polymerization.
    alpha : float
        Ratio kt/kp.
    init_mon : float
        Initial monomer concentration.
    init : float
        Initial initiator concentration.
    order : float
        Termination order.
    bn : float
        Inverse propagation order.

    Returns
    -------
    poisson_matrix : ndarray, shape (n_times, n_dps)
        poisson_matrix[i, j] = poisson.pmf(dps[j], nup(times[i]))
    """
    n_times = len(times)

    # Compute nup at each time point
    nups = np.array([
        living_chain_dp(alpha, init_mon, init, order, t, bn)
        for t in times
    ])

    # Vectorized Poisson computation: shape (n_times, n_dps)
    poisson_matrix = poisson.pmf(dps[np.newaxis, :], nups[:, np.newaxis])

    return poisson_matrix
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mwd.py::TestPoissonPrecomputation -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add polyterm/models/fitting.py tests/test_fit_mwd.py
git commit -m "feat: add Poisson matrix precomputation function"
```

---

### Task 3: Add Fixed Quadrature Integration Function

**Files:**
- Modify: `polyterm/models/fitting.py`
- Test: `tests/test_fit_mwd.py`

**Step 1: Write the failing test**

Update import in `tests/test_fit_mwd.py`:

```python
from polyterm.models.fitting import (
    _get_quadrature_points,
    _precompute_poisson_matrix,
    _compute_dead_fracs_quadrature,
)
```

Add new test class:

```python
class TestDeadFracsQuadrature:
    """Test the fixed quadrature dead fraction computation."""

    def test_returns_correct_shape(self):
        """Test that dead fractions have correct shape."""
        n_points = 50
        time_end = 5.0
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        combination = False

        times, weights = _get_quadrature_points(n_points, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        assert dead_fracs.shape == (len(dps),)

    def test_values_are_nonnegative(self):
        """Test that dead fractions are non-negative."""
        n_points = 50
        time_end = 5.0
        dps = np.arange(1, 200)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        combination = False

        times, weights = _get_quadrature_points(n_points, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        assert np.all(dead_fracs >= 0)

    def test_sum_is_reasonable(self):
        """Test that total dead fraction is between 0 and init."""
        n_points = 50
        time_end = 5.0
        dps = np.arange(1, 500)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        combination = False

        times, weights = _get_quadrature_points(n_points, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )

        dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        total_dead = np.sum(dead_fracs)
        assert 0 < total_dead < init
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fit_mwd.py::TestDeadFracsQuadrature -v`
Expected: FAIL with "cannot import name '_compute_dead_fracs_quadrature'"

**Step 3: Write minimal implementation**

Add to `polyterm/models/fitting.py` after `_precompute_poisson_matrix`:

```python
def _compute_dead_fracs_quadrature(
    times: np.ndarray,
    weights: np.ndarray,
    poisson_matrix: np.ndarray,
    init: float,
    order: float,
    combination: bool
) -> np.ndarray:
    """
    Compute dead chain fractions using fixed quadrature.

    Parameters
    ----------
    times : ndarray
        Quadrature time points.
    weights : ndarray
        Quadrature weights.
    poisson_matrix : ndarray, shape (n_times, n_dps)
        Precomputed Poisson PMFs.
    init : float
        Initial initiator concentration.
    order : float
        Termination order.
    combination : bool
        Whether termination is by combination.

    Returns
    -------
    dead_fracs : ndarray, shape (n_dps,)
        Mole fraction of dead chains at each DP.
    """
    # Compute b(t) at each time point
    b_vals = np.array([
        living_chain_concentration(init, order, t) for t in times
    ])

    # Compute integrand weights: (b^order) * (init^(1-order))
    integrand_weights = (b_vals ** order) * (init ** (1 - order))

    # For combination termination, divide by 2
    if combination:
        integrand_weights = integrand_weights / 2

    # Weighted sum: (n_times,) @ (n_times, n_dps) -> (n_dps,)
    dead_fracs = (weights * integrand_weights) @ poisson_matrix

    return dead_fracs
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fit_mwd.py::TestDeadFracsQuadrature -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add polyterm/models/fitting.py tests/test_fit_mwd.py
git commit -m "feat: add fixed quadrature dead fraction computation"
```

---

### Task 4: Add Accuracy Test Comparing quad_vec vs Fixed Quadrature

**Files:**
- Modify: `tests/test_fit_mwd.py`

**Step 1: Write the test**

Add new test class to `tests/test_fit_mwd.py`:

```python
from scipy.integrate import quad_vec
from polyterm.core.distributions import living_distribution_integrand


class TestQuadratureAccuracy:
    """Test that fixed quadrature matches quad_vec accuracy."""

    def test_matches_quad_vec_order_1(self):
        """Test fixed quadrature matches quad_vec for order=1."""
        dps = np.arange(1, 300)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.0
        bn = 1.0
        time_end = 5.0
        combination = False

        # Reference: quad_vec
        args = (dps, alpha, init_mon, init, order, combination, bn)
        ref_dead_fracs, _ = quad_vec(
            living_distribution_integrand, 0, time_end, args=args
        )

        # Fixed quadrature
        times, weights = _get_quadrature_points(50, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )
        test_dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        # Should match within 1%
        relative_error = np.abs(test_dead_fracs - ref_dead_fracs) / (ref_dead_fracs + 1e-15)
        # Only check where there's significant probability
        significant = ref_dead_fracs > 1e-10
        assert np.all(relative_error[significant] < 0.01)

    def test_matches_quad_vec_order_1_5(self):
        """Test fixed quadrature matches quad_vec for order=1.5."""
        dps = np.arange(1, 300)
        alpha = 0.002
        init_mon = 1.0
        init = 0.005
        order = 1.5
        bn = 1.0
        time_end = 5.0
        combination = False

        # Reference: quad_vec
        args = (dps, alpha, init_mon, init, order, combination, bn)
        ref_dead_fracs, _ = quad_vec(
            living_distribution_integrand, 0, time_end, args=args
        )

        # Fixed quadrature
        times, weights = _get_quadrature_points(50, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )
        test_dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        # Should match within 1%
        relative_error = np.abs(test_dead_fracs - ref_dead_fracs) / (ref_dead_fracs + 1e-15)
        significant = ref_dead_fracs > 1e-10
        assert np.all(relative_error[significant] < 0.01)

    def test_matches_quad_vec_order_2(self):
        """Test fixed quadrature matches quad_vec for order=2."""
        dps = np.arange(1, 300)
        alpha = 0.5
        init_mon = 1.0
        init = 0.005
        order = 2.0
        bn = 1.0
        time_end = 5.0
        combination = False

        # Reference: quad_vec
        args = (dps, alpha, init_mon, init, order, combination, bn)
        ref_dead_fracs, _ = quad_vec(
            living_distribution_integrand, 0, time_end, args=args
        )

        # Fixed quadrature
        times, weights = _get_quadrature_points(50, time_end)
        poisson_matrix = _precompute_poisson_matrix(
            times, dps, alpha, init_mon, init, order, bn
        )
        test_dead_fracs = _compute_dead_fracs_quadrature(
            times, weights, poisson_matrix, init, order, combination
        )

        # Should match within 1%
        relative_error = np.abs(test_dead_fracs - ref_dead_fracs) / (ref_dead_fracs + 1e-15)
        significant = ref_dead_fracs > 1e-10
        assert np.all(relative_error[significant] < 0.01)
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_fit_mwd.py::TestQuadratureAccuracy -v`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
git add tests/test_fit_mwd.py
git commit -m "test: add accuracy tests comparing fixed quadrature vs quad_vec"
```

---

### Task 5: Modify _calculate_mwd_internal to Use Fixed Quadrature

**Files:**
- Modify: `polyterm/models/fitting.py:993-1045`

**Step 1: Modify the function**

Replace `_calculate_mwd_internal` in `polyterm/models/fitting.py`:

```python
def _calculate_mwd_internal(
    alpha: float,
    init: float,
    dps: np.ndarray,
    mws: np.ndarray,
    broadenings: np.ndarray,
    init_mon: float,
    order: float,
    combination: bool,
    bn: float,
    time: Optional[float] = None,
    conv: Optional[float] = None,
    n_quadrature_points: int = 50
) -> np.ndarray:
    """
    Calculate MWD given either time or conversion.

    Internal function used during optimization. Uses fixed Gauss-Legendre
    quadrature with precomputed Poisson distributions for efficiency.

    Parameters
    ----------
    alpha : float
        Ratio kt/kp.
    init : float
        Initial initiator concentration.
    dps : ndarray
        Degrees of polymerization.
    mws : ndarray
        Molecular weights for output.
    broadenings : ndarray
        Broadening matrix.
    init_mon : float
        Initial monomer concentration.
    order : float
        Termination order.
    combination : bool
        Whether termination is by combination.
    bn : float
        Inverse propagation order.
    time : float, optional
        Reduced time (if known).
    conv : float, optional
        Conversion (if time not known).
    n_quadrature_points : int, optional
        Number of Gauss-Legendre quadrature points. Default 50.

    Returns
    -------
    ndarray
        Normalized MWD at the specified molecular weights.
    """
    # Guard against invalid parameters
    if not (np.isfinite(alpha) and np.isfinite(init)):
        return np.full(len(mws), np.nan)

    if alpha <= 0 or init <= 0:
        return np.full(len(mws), np.nan)

    # Determine time from conversion if needed
    if time is None:
        if conv is None:
            return np.full(len(mws), np.nan)
        if np.isclose(conv, 1):
            time = time_to_chain_death(0.9999, init, order)
        else:
            time = conversion_to_time(alpha, init, order, conv, bn)

    # Guard against runaway integration
    if not np.isfinite(time) or time < 0:
        return np.full(len(mws), np.nan)

    # Fixed quadrature: precompute Poisson distributions
    quad_times, quad_weights = _get_quadrature_points(n_quadrature_points, time)

    # Handle combination termination: need 2*nup for dead chains
    if combination:
        # Precompute with 2*nup for combination termination
        poisson_matrix = _precompute_poisson_matrix_combination(
            quad_times, dps, alpha, init_mon, init, order, bn
        )
    else:
        poisson_matrix = _precompute_poisson_matrix(
            quad_times, dps, alpha, init_mon, init, order, bn
        )

    dead_fracs = _compute_dead_fracs_quadrature(
        quad_times, quad_weights, poisson_matrix, init, order, combination
    )

    # Living chains at final time
    b = living_chain_concentration(init, order, time)
    nup = living_chain_dp(alpha, init_mon, init, order, time, bn)
    alive_fracs = b * poisson.pmf(dps, nup)

    total_fracs = alive_fracs + dead_fracs

    pred_mwd = np.matmul(broadenings, total_fracs * dps)
    norm = np.trapezoid(pred_mwd, np.log(mws))
    if norm == 0 or not np.isfinite(norm):
        return np.full(len(mws), np.nan)
    return pred_mwd / norm
```

**Step 2: Add combination helper function**

Add after `_precompute_poisson_matrix`:

```python
def _precompute_poisson_matrix_combination(
    times: np.ndarray,
    dps: np.ndarray,
    alpha: float,
    init_mon: float,
    init: float,
    order: float,
    bn: float
) -> np.ndarray:
    """
    Precompute Poisson PMFs for combination termination (2*nup).

    Same as _precompute_poisson_matrix but uses 2*nup for the Poisson
    parameter, as combination termination produces chains with ~2x the DP.
    """
    n_times = len(times)

    nups = np.array([
        living_chain_dp(alpha, init_mon, init, order, t, bn)
        for t in times
    ])

    # Use 2*nup for combination termination
    poisson_matrix = poisson.pmf(dps[np.newaxis, :], 2 * nups[:, np.newaxis])

    return poisson_matrix
```

**Step 3: Run existing tests to verify no regression**

Run: `pytest tests/test_fit_mwd.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add polyterm/models/fitting.py
git commit -m "refactor: use fixed quadrature in _calculate_mwd_internal"
```

---

### Task 6: Add n_quadrature_points Parameter to fit_mwd

**Files:**
- Modify: `polyterm/models/fitting.py` (fit_mwd function signature and _create_objective)

**Step 1: Update fit_mwd signature**

In `polyterm/models/fitting.py`, modify the `fit_mwd` function signature (around line 687):

```python
def fit_mwd(
    molecular_weights: np.ndarray,
    intensities: np.ndarray,
    order: float,
    monomer_mw: float,
    init_mon: float,
    *,
    sigma: Optional[float] = None,
    tau: Optional[float] = None,
    conversion: Optional[float] = None,
    init: Optional[float] = None,
    combination: bool = False,
    bn: float = 1.0,
    max_fit_points: int = 500,
    n_quadrature_points: int = 50,
) -> FitResult:
```

Update the docstring to include:

```python
    n_quadrature_points : int, optional
        Number of Gauss-Legendre quadrature points for integration.
        Higher values improve accuracy but slow computation. Default 50.
```

**Step 2: Pass parameter through to _create_objective**

Update the call to `_create_objective` (around line 817):

```python
    objective = _create_objective(
        fit_mws, fit_ints, dps, monomer_mw, init_mon, order,
        sigma=sigma, tau=tau_val, combination=combination, bn=bn,
        param_spec=param_spec, fit_sigma=fit_sigma,
        n_quadrature_points=n_quadrature_points
    )
```

**Step 3: Update _create_objective signature and pass to _calculate_mwd_internal**

Modify `_create_objective` signature (around line 918):

```python
def _create_objective(
    fit_mws: np.ndarray,
    fit_ints: np.ndarray,
    dps: np.ndarray,
    monomer_mw: float,
    init_mon: float,
    order: float,
    *,
    sigma: Optional[float],
    tau: float,
    combination: bool,
    bn: float,
    param_spec: Dict,
    fit_sigma: bool,
    n_quadrature_points: int = 50
) -> callable:
```

Update the calls to `_calculate_mwd_internal` inside the objective function (around lines 967 and 976):

```python
                pred = _calculate_mwd_internal(
                    alpha, init_val, dps, fit_mws, cache['broadening'],
                    init_mon, order, combination, bn,
                    time=params['time'],
                    n_quadrature_points=n_quadrature_points
                )
```

and:

```python
                pred = _calculate_mwd_internal(
                    alpha, init_val, dps, fit_mws, cache['broadening'],
                    init_mon, order, combination, bn,
                    conv=fixed_conversion,
                    n_quadrature_points=n_quadrature_points
                )
```

**Step 4: Run tests**

Run: `pytest tests/test_fit_mwd.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add polyterm/models/fitting.py
git commit -m "feat: add n_quadrature_points parameter to fit_mwd"
```

---

### Task 7: Add Integration Test for High DP Performance

**Files:**
- Modify: `tests/test_fit_mwd.py`

**Step 1: Add performance-focused test**

Add new test class to `tests/test_fit_mwd.py`:

```python
class TestHighDPFitting:
    """Test fitting with high degree of polymerization samples."""

    def test_fit_high_dp_sample(self):
        """Test that fit_mwd works correctly for high DP (nu=500)."""
        # Use first_order_params which has nu=500
        mws = np.logspace(3, 6, 500)  # 1k to 1M
        params = {
            "monomer_mw": 100.0,
            "nu": 500.0,
            "alpha": 0.0005,
            "init_mon": 1.0,
            "init": 0.001,
            "order": 1.0,
            "sigma": 0.12
        }

        mwd_ints = calculate_mwd(
            mws,
            params['monomer_mw'],
            params['nu'],
            params['alpha'],
            params['init_mon'],
            params['init'],
            params['order'],
            params['sigma']
        )

        result = fit_mwd(
            mws, mwd_ints,
            order=params['order'],
            monomer_mw=params['monomer_mw'],
            init_mon=params['init_mon'],
            sigma=params['sigma']
        )

        # Should achieve good fit
        assert result.r_squared > 0.95
        # Should recover alpha within 20%
        assert np.isclose(result.alpha, params['alpha'], rtol=0.2)

    def test_quadrature_points_parameter(self):
        """Test that n_quadrature_points parameter works."""
        mws = np.logspace(3, 5, 200)
        params = {
            "monomer_mw": 100.0,
            "nu": 100.0,
            "alpha": 0.002,
            "init_mon": 1.0,
            "init": 0.005,
            "order": 1.5,
            "sigma": 0.12
        }

        mwd_ints = calculate_mwd(
            mws,
            params['monomer_mw'],
            params['nu'],
            params['alpha'],
            params['init_mon'],
            params['init'],
            params['order'],
            params['sigma']
        )

        # Fit with different quadrature points
        result_30 = fit_mwd(
            mws, mwd_ints,
            order=params['order'],
            monomer_mw=params['monomer_mw'],
            init_mon=params['init_mon'],
            sigma=params['sigma'],
            n_quadrature_points=30
        )

        result_100 = fit_mwd(
            mws, mwd_ints,
            order=params['order'],
            monomer_mw=params['monomer_mw'],
            init_mon=params['init_mon'],
            sigma=params['sigma'],
            n_quadrature_points=100
        )

        # Both should give good fits
        assert result_30.r_squared > 0.95
        assert result_100.r_squared > 0.95
        # Results should be similar
        assert np.isclose(result_30.alpha, result_100.alpha, rtol=0.05)
```

**Step 2: Run test**

Run: `pytest tests/test_fit_mwd.py::TestHighDPFitting -v`
Expected: PASS (2 tests)

**Step 3: Commit**

```bash
git add tests/test_fit_mwd.py
git commit -m "test: add high DP fitting tests"
```

---

### Task 8: Update _build_fit_result to Use Fixed Quadrature

**Files:**
- Modify: `polyterm/models/fitting.py` (_build_fit_result function)

**Step 1: Update _build_fit_result signature and implementation**

The `_build_fit_result` function also calls `_calculate_mwd_internal`. Update it to pass `n_quadrature_points`:

In the function signature (around line 1073), add parameter:

```python
def _build_fit_result(
    opt_result: Dict,
    fit_mws: np.ndarray,
    fit_ints: np.ndarray,
    dps: np.ndarray,
    monomer_mw: float,
    init_mon: float,
    order: float,
    *,
    sigma: Optional[float],
    tau: float,
    combination: bool,
    bn: float,
    param_spec: Dict,
    fit_sigma: bool,
    n_quadrature_points: int = 50
) -> FitResult:
```

Update the call to `_calculate_mwd_internal` (around line 1114):

```python
    pred_ints = _calculate_mwd_internal(
        alpha, init_val, dps, fit_mws, broadenings,
        init_mon, order, combination, bn,
        time=time,
        n_quadrature_points=n_quadrature_points
    )
```

**Step 2: Update fit_mwd to pass parameter to _build_fit_result**

Update the call to `_build_fit_result` (around line 827):

```python
    return _build_fit_result(
        opt_result, fit_mws, fit_ints, dps, monomer_mw, init_mon, order,
        sigma=sigma, tau=tau_val, combination=combination, bn=bn,
        param_spec=param_spec, fit_sigma=fit_sigma,
        n_quadrature_points=n_quadrature_points
    )
```

**Step 3: Run all tests**

Run: `pytest tests/test_fit_mwd.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add polyterm/models/fitting.py
git commit -m "refactor: pass n_quadrature_points through _build_fit_result"
```

---

### Task 9: Final Verification and Cleanup

**Files:**
- All test files

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run with coverage**

Run: `pytest tests/ --cov=polyterm --cov-report=term-missing`
Expected: Coverage >= 80%

**Step 3: Final commit with any cleanup**

```bash
git status
# If any files need cleanup, add them
git commit -m "chore: final cleanup for fixed quadrature optimization"
```

---

## Summary

This plan implements fixed Gauss-Legendre quadrature to replace adaptive `quad_vec` in `fit_mwd`. The key optimizations are:

1. **Precomputed Poisson distributions**: All Poisson PMFs are computed once per objective function evaluation
2. **Matrix multiplication**: Integration becomes a simple weighted sum via matrix multiplication
3. **Configurable accuracy**: Users can adjust `n_quadrature_points` (default 50) to trade off speed vs accuracy

Expected speedup: 10-20x for high molecular weight samples (DP > 500).
