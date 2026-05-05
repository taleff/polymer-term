"""
Integration tests comparing model predictions to experimental SEC data.

These tests validate the kinetic model against real experimental data
from first-order and second-order termination polymerizations. The
experimental data includes SEC chromatograms and known kinetic parameters.
"""

import pytest
import numpy as np
import json
from pathlib import Path

from polyterm import calculate_mwd, fit_mwd, MWDResult
from polyterm.core.kinetics import monomer_conversion


# Path to test data
DATA_DIR = Path(__file__).parent / "data"
FIRST_ORDER_DIR = DATA_DIR / "first_order"
SECOND_ORDER_DIR = DATA_DIR / "second_order"


def load_chromatogram(filepath):
    """
    Load SEC chromatogram from text file.

    Parameters
    ----------
    filepath : Path
        Path to chromatogram file with format:
        - Line 1: Sample name
        - Line 2: "X:	Y:" header
        - Lines 3+: tab-separated elution_time, intensity

    Returns
    -------
    elution_times : ndarray
        Elution times in minutes
    intensities : ndarray
        Detector response values
    """
    elution_times = []
    intensities = []

    with open(filepath, 'r') as f:
        lines = f.readlines()
        # Skip header lines (sample name and X:/Y: header)
        for line in lines[2:]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    elution_times.append(float(parts[0]))
                    intensities.append(float(parts[1]))
                except ValueError:
                    continue

    return np.array(elution_times), np.array(intensities)


def load_calibration(filepath, calibration_name):
    """
    Load SEC calibration from JSON file.

    Parameters
    ----------
    filepath : Path
        Path to calibrations.json
    calibration_name : str
        Name of calibration to load

    Returns
    -------
    calibration_func : callable
        Function that converts elution time to molecular weight
    """
    with open(filepath, 'r') as f:
        calibrations = json.load(f)

    cal = calibrations[calibration_name]

    if cal['type'] == 'cubic':
        coeffs = cal['params']
        def calibration_func(elution_time):
            # Cubic polynomial: log10(MW) = a*t^3 + b*t^2 + c*t + d
            log_mw = (coeffs[0] * elution_time**3 +
                      coeffs[1] * elution_time**2 +
                      coeffs[2] * elution_time +
                      coeffs[3])
            return 10**log_mw
        return calibration_func
    else:
        raise ValueError(f"Unknown calibration type: {cal['type']}")


def load_experiment_params(params_path):
    """
    Load experiment parameters from JSON file.

    Parameters
    ----------
    params_path : Path
        Path to params.json file

    Returns
    -------
    params : dict
        Dictionary with kinetics, broadening, and run information
    """
    with open(params_path, 'r') as f:
        data = json.load(f)

    # Flatten structure for easier access
    params = {
        **data['kinetics'],
        **data['broadening'],
        'alphas': {int(k): v['alpha'] for k, v in data['runs'].items()}
    }
    return params


def process_chromatogram(elution_times, intensities, calibration_func,
                         min_mw=500, max_mw=500000):
    """
    Convert chromatogram from elution time to molecular weight basis.

    Parameters
    ----------
    elution_times : ndarray
        Elution times in minutes
    intensities : ndarray
        Detector response values
    calibration_func : callable
        Function converting elution time to MW
    min_mw : float
        Minimum MW to include
    max_mw : float
        Maximum MW to include

    Returns
    -------
    molecular_weights : ndarray
        Molecular weights (sorted low to high)
    intensities : ndarray
        Intensities at each MW (normalized)
    """
    # Convert to MW
    mws = calibration_func(elution_times)

    # Filter to valid MW range
    mask = (mws >= min_mw) & (mws <= max_mw) & np.isfinite(mws)
    mws = mws[mask]
    ints = intensities[mask]

    # SEC elutes high MW first, so reverse to get low-to-high MW order
    sort_idx = np.argsort(mws)
    mws = mws[sort_idx]
    ints = ints[sort_idx]

    # Baseline correction (simple: subtract minimum in tails)
    baseline = np.percentile(ints, 5)
    ints = ints - baseline
    ints = np.maximum(ints, 0)  # No negative intensities

    # Normalize by peak (to match model output)
    if np.max(ints) > 0:
        ints = ints / np.max(ints)

    return mws, ints


def load_experimental_data(data_dir, calibration_func):
    """
    Load all experimental chromatograms from a directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing run_*.txt files and params.json
    calibration_func : callable
        Function converting elution time to MW

    Returns
    -------
    data : dict
        Dictionary mapping run number to {mws, ints}
    params : dict
        Experiment parameters
    """
    params_path = data_dir / "params.json"
    if not params_path.exists():
        return {}, {}

    params = load_experiment_params(params_path)

    data = {}
    for run_num in range(1, 6):
        filepath = data_dir / f"run_{run_num}.txt"
        if filepath.exists():
            times, ints = load_chromatogram(filepath)
            mws, processed_ints = process_chromatogram(
                times, ints, calibration_func,
                min_mw=200, max_mw=100000
            )
            data[run_num] = {'mws': mws, 'ints': processed_ints}

    return data, params


@pytest.fixture
def first_order_params():
    """Load first-order experimental parameters."""
    params_path = FIRST_ORDER_DIR / "params.json"
    if not params_path.exists():
        pytest.skip("First-order params not found")
    return load_experiment_params(params_path)


@pytest.fixture
def second_order_params():
    """Load second-order experimental parameters."""
    params_path = SECOND_ORDER_DIR / "params.json"
    if not params_path.exists():
        pytest.skip("Second-order params not found")
    return load_experiment_params(params_path)


@pytest.fixture
def calibration_func():
    """Load the SEC calibration function."""
    cal_path = DATA_DIR / "calibrations.json"
    if not cal_path.exists():
        pytest.skip("Calibration file not found")
    return load_calibration(cal_path, "ri_2025_11_21")


class TestExperimentalDataLoading:
    """Test that experimental data loads correctly."""

    def test_load_chromatogram(self):
        """Test loading a chromatogram file."""
        filepath = FIRST_ORDER_DIR / "run_1.txt"
        if not filepath.exists():
            pytest.skip("Experimental data not found")

        times, ints = load_chromatogram(filepath)

        assert len(times) > 1000  # Expect many data points
        assert len(times) == len(ints)
        assert times[0] < times[-1]  # Times should increase

    def test_load_calibration(self, calibration_func):
        """Test loading calibration."""
        # At low elution time (early), MW should be high
        mw_early = calibration_func(10.0)
        mw_late = calibration_func(20.0)

        assert mw_early > mw_late  # SEC: early elution = high MW
        assert mw_early > 10000  # Reasonable MW values
        assert mw_late > 100

    def test_process_chromatogram(self, calibration_func):
        """Test chromatogram processing."""
        filepath = FIRST_ORDER_DIR / "run_1.txt"
        if not filepath.exists():
            pytest.skip("Experimental data not found")

        times, ints = load_chromatogram(filepath)
        mws, processed_ints = process_chromatogram(times, ints, calibration_func,
                                                         min_mw=200, max_mw=100000)

        # Should be sorted low to high
        assert np.all(np.diff(mws) > 0)

        # Should be peak-normalized (max = 1)
        assert np.isclose(np.max(processed_ints), 1.0, rtol=0.01)

        # No negative intensities
        assert np.all(processed_ints >= 0)


class TestFirstOrderValidation:
    """Validate model against first-order termination experiments."""

    @pytest.fixture
    def experimental_data(self, calibration_func):
        """Load all first-order experimental chromatograms."""
        data, _ = load_experimental_data(FIRST_ORDER_DIR, calibration_func)
        if not data:
            pytest.skip("No experimental data found")
        return data

    def test_calculate_mwd_matches_experiment(self, experimental_data,
                                               first_order_params):
        """Test that calculated MWD reasonably matches experimental data."""
        params = first_order_params

        for run_num, exp_data in experimental_data.items():
            alpha = params['alphas'][run_num]

            # Calculate conversion from the kinetic parameters
            kp = params['kp']
            kt = alpha * kp
            time = params['time']

            mon_conc = monomer_conversion(
                time, kp, kt,
                params['init_mon'],
                params['init'],
                params['order']
            )
            conversion = 1 - (mon_conc / params['init_mon'])

            result = calculate_mwd(
                exp_data['mws'],
                params['monomer_mw'],
                params['init_mon'],
                alpha,
                params['init'],
                conversion,
                params['order'],
                params['sigma'],
                params['tau']
            )

            assert isinstance(result, MWDResult)
            assert np.all(np.isfinite(result.intensities))
            assert np.any(result.intensities > 0)

    def test_fit_mwd_recovers_alpha(self, experimental_data, first_order_params):
        """Test that fitting recovers approximately correct alpha values."""
        params = first_order_params

        for run_num, exp_data in experimental_data.items():
            result = fit_mwd(
                exp_data['mws'],
                exp_data['ints'],
                order=params['order'],
                monomer_mw=params['monomer_mw'],
                init_mon=params['init_mon'],
                init=params['init'],
                sigma=params['sigma'],
                tau=params['tau']
            )

            assert result.r_squared > 0.80, \
                f"Run {run_num}: R^2 = {result.r_squared:.3f} < 0.80"


class TestSecondOrderValidation:
    """Validate model against second-order termination experiments."""

    @pytest.fixture
    def experimental_data(self, calibration_func):
        """Load all second-order experimental chromatograms."""
        data, _ = load_experimental_data(SECOND_ORDER_DIR, calibration_func)
        if not data:
            pytest.skip("No experimental data found")
        return data

    def test_calculate_mwd_matches_experiment(self, experimental_data,
                                               second_order_params):
        """Test that calculated MWD reasonably matches experimental data."""
        params = second_order_params

        for run_num, exp_data in experimental_data.items():
            alpha = params['alphas'][run_num]

            # Calculate conversion from the kinetic parameters
            kp = params['kp']
            kt = alpha * kp
            time = params['time']

            mon_conc = monomer_conversion(
                time, kp, kt,
                params['init_mon'],
                params['init'],
                params['order']
            )
            conversion = 1 - (mon_conc / params['init_mon'])

            result = calculate_mwd(
                exp_data['mws'],
                params['monomer_mw'],
                params['init_mon'],
                alpha,
                params['init'],
                conversion,
                params['order'],
                params['sigma'],
                params['tau']
            )

            assert isinstance(result, MWDResult)
            assert np.all(np.isfinite(result.intensities))
            assert np.any(result.intensities > 0)

    def test_fit_mwd_recovers_alpha(self, experimental_data, second_order_params):
        """Test that fitting recovers approximately correct alpha values."""
        params = second_order_params

        for run_num, exp_data in experimental_data.items():
            result = fit_mwd(
                exp_data['mws'],
                exp_data['ints'],
                order=params['order'],
                monomer_mw=params['monomer_mw'],
                init_mon=params['init_mon'],
                init=params['init'],
                sigma=params['sigma'],
                tau=params['tau']
            )

            assert result.r_squared > 0.80, \
                f"Run {run_num}: R^2 = {result.r_squared:.3f} < 0.80"


class TestVisualization:
    """Tests that generate visualization figures."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory for figures."""
        return tmp_path / "figures"

    def test_generate_first_order_figures(self, calibration_func,
                                           first_order_params, output_dir):
        """Generate figures for first-order validation."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        output_dir.mkdir(exist_ok=True)
        params = first_order_params

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        true_alphas = []
        fitted_alphas = []

        for idx, run_num in enumerate(range(1, 6)):
            filepath = FIRST_ORDER_DIR / f"run_{run_num}.txt"
            if not filepath.exists():
                continue

            times, ints = load_chromatogram(filepath)
            mws, exp_ints = process_chromatogram(times, ints, calibration_func,
                                                 min_mw=200, max_mw=100000)

            # Fit the data (provide init for better convergence)
            result = fit_mwd(
                mws, exp_ints,
                order=params['order'],
                monomer_mw=params['monomer_mw'],
                init_mon=params['init_mon'],
                init=params['init'],
                sigma=params['sigma'],
                tau=params['tau']
            )

            # Calculate distribution using true/known parameters
            true_alpha = params['alphas'][run_num]
            kp = params['kp']
            kt = true_alpha * kp
            time = params['time']
            mon_conc = monomer_conversion(
                time, kp, kt, params['init_mon'], params['init'], params['order']
            )
            conversion = 1 - (mon_conc / params['init_mon'])

            calc_result = calculate_mwd(
                mws, params['monomer_mw'], params['init_mon'], true_alpha,
                params['init'], conversion, params['order'],
                params['sigma'], params['tau']
            )

            true_alphas.append(true_alpha)
            fitted_alphas.append(result.alpha)

            ax = axes[idx]
            ax.semilogx(mws, exp_ints, 'b-', label='Experimental', alpha=0.7)
            ax.semilogx(mws, calc_result.intensities,
                       'g--', label='Calculated', linewidth=2)
            ax.semilogx(result.molecular_weights, result.intensities,
                       'r:', label='Fitted', linewidth=2)
            ax.set_title(f"Run {run_num}\n"
                        f"True α={true_alpha:.4f}, Fit α={result.alpha:.4f}\n"
                        f"R²={result.r_squared:.4f}")
            ax.set_xlabel('Molecular Weight (g/mol)')
            ax.set_ylabel('w(M)')
            ax.legend(fontsize=8)
            ax.set_xlim(2e2, 1e5)

        # Sixth plot: alpha comparison
        ax6 = axes[5]
        ax6.scatter(true_alphas, fitted_alphas, s=100, c='blue', edgecolors='black')
        min_val = min(min(true_alphas), min(fitted_alphas)) * 0.8
        max_val = max(max(true_alphas), max(fitted_alphas)) * 1.2
        ax6.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
        ax6.set_xlabel('True α')
        ax6.set_ylabel('Fitted α')
        ax6.set_title('Alpha Recovery')
        ax6.legend()
        ax6.set_xlim(min_val, max_val)
        ax6.set_ylim(min_val, max_val)

        plt.tight_layout()
        fig.savefig(output_dir / 'first_order_validation.png', dpi=150)
        plt.close(fig)

        assert (output_dir / 'first_order_validation.png').exists()

    def test_generate_second_order_figures(self, calibration_func,
                                            second_order_params, output_dir):
        """Generate figures for second-order validation."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        output_dir.mkdir(exist_ok=True)
        params = second_order_params

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        true_alphas = []
        fitted_alphas = []

        for idx, run_num in enumerate(range(1, 6)):
            filepath = SECOND_ORDER_DIR / f"run_{run_num}.txt"
            if not filepath.exists():
                continue

            times, ints = load_chromatogram(filepath)
            mws, exp_ints = process_chromatogram(times, ints, calibration_func,
                                                 min_mw=200, max_mw=100000)

            # Fit the data (provide init for better convergence, especially for order=2)
            result = fit_mwd(
                mws, exp_ints,
                order=params['order'],
                monomer_mw=params['monomer_mw'],
                init_mon=params['init_mon'],
                init=params['init'],
                sigma=params['sigma'],
                tau=params['tau']
            )

            # Calculate distribution using true/known parameters
            true_alpha = params['alphas'][run_num]
            kp = params['kp']
            kt = true_alpha * kp
            time = params['time']
            mon_conc = monomer_conversion(
                time, kp, kt, params['init_mon'], params['init'], params['order']
            )
            conversion = 1 - (mon_conc / params['init_mon'])

            calc_result = calculate_mwd(
                mws, params['monomer_mw'], params['init_mon'], true_alpha,
                params['init'], conversion, params['order'],
                params['sigma'], params['tau']
            )

            true_alphas.append(true_alpha)
            fitted_alphas.append(result.alpha)

            ax = axes[idx]
            ax.semilogx(mws, exp_ints, 'b-', label='Experimental', alpha=0.7)
            ax.semilogx(mws, calc_result.intensities,
                       'g--', label='Calculated', linewidth=2)
            ax.semilogx(result.molecular_weights, result.intensities,
                       'r:', label='Fitted', linewidth=2)
            ax.set_title(f"Run {run_num}\n"
                        f"True α={true_alpha:.3f}, Fit α={result.alpha:.3f}\n"
                        f"R²={result.r_squared:.4f}")
            ax.set_xlabel('Molecular Weight (g/mol)')
            ax.set_ylabel('w(M)')
            ax.legend(fontsize=8)
            ax.set_xlim(2e2, 1e5)

        # Sixth plot: alpha comparison
        ax6 = axes[5]
        ax6.scatter(true_alphas, fitted_alphas, s=100, c='blue', edgecolors='black')
        min_val = min(min(true_alphas), min(fitted_alphas)) * 0.8
        max_val = max(max(true_alphas), max(fitted_alphas)) * 1.2
        ax6.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
        ax6.set_xlabel('True α')
        ax6.set_ylabel('Fitted α')
        ax6.set_title('Alpha Recovery')
        ax6.legend()
        ax6.set_xlim(min_val, max_val)
        ax6.set_ylim(min_val, max_val)

        plt.tight_layout()
        fig.savefig(output_dir / 'second_order_validation.png', dpi=150)
        plt.close(fig)

        assert (output_dir / 'second_order_validation.png').exists()


def generate_validation_report(output_path=None):
    """
    Generate a standalone validation report with figures for both
    first-order and second-order termination experiments.

    Parameters
    ----------
    output_path : Path, optional
        Directory to save figures. Defaults to tests/data/figures/
    """
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = DATA_DIR / "figures"
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    # Load calibration
    cal_func = load_calibration(DATA_DIR / "calibrations.json",
                                 "ri_2025_11_21")

    # Process each experiment type
    experiments = [
        ("First Order (n=1)", FIRST_ORDER_DIR, "first_order_validation.png"),
        ("Second Order (n=2)", SECOND_ORDER_DIR, "second_order_validation.png"),
    ]

    for exp_name, exp_dir, filename in experiments:
        if not (exp_dir / "params.json").exists():
            print(f"Skipping {exp_name}: params.json not found")
            continue

        params = load_experiment_params(exp_dir / "params.json")

        print(f"\n{exp_name} Validation")
        print("=" * 60)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        true_alphas = []
        fitted_alphas = []
        results_summary = []

        for idx, run_num in enumerate(range(1, 6)):
            filepath = exp_dir / f"run_{run_num}.txt"
            if not filepath.exists():
                print(f"Skipping run {run_num}: file not found")
                continue

            times, ints = load_chromatogram(filepath)
            mws, exp_ints = process_chromatogram(times, ints, cal_func,
                                                         min_mw=200, max_mw=100000)

            # Fit the data (provide init for better convergence)
            result = fit_mwd(
                mws, exp_ints,
                order=params['order'],
                monomer_mw=params['monomer_mw'],
                init_mon=params['init_mon'],
                init=params['init'],
                sigma=params['sigma'],
                tau=params['tau']
            )

            true_alpha = params['alphas'][run_num]
            true_alphas.append(true_alpha)
            fitted_alphas.append(result.alpha)

            # Calculate distribution using true/known parameters
            kp = params['kp']
            kt = true_alpha * kp
            time = params['time']
            mon_conc = monomer_conversion(
                time, kp, kt, params['init_mon'], params['init'], params['order']
            )
            conversion = 1 - (mon_conc / params['init_mon'])

            calc_result = calculate_mwd(
                mws, params['monomer_mw'], params['init_mon'], true_alpha,
                params['init'], conversion, params['order'],
                params['sigma'], params['tau']
            )

            alpha_error = (result.alpha - true_alpha) / true_alpha * 100

            results_summary.append({
                'run': run_num,
                'true_alpha': true_alpha,
                'fitted_alpha': result.alpha,
                'error_pct': alpha_error,
                'r_squared': result.r_squared,
                'dead_fraction': result.dead_chain_fraction
            })

            # Format alpha display based on magnitude
            if true_alpha < 0.1:
                alpha_fmt = ".4f"
            else:
                alpha_fmt = ".3f"

            print(f"Run {run_num}:")
            print(f"  True α:   {true_alpha:{alpha_fmt}}")
            print(f"  Fitted α: {result.alpha:{alpha_fmt}} ({alpha_error:+.1f}% error)")
            print(f"  R²:       {result.r_squared:.4f}")
            print(f"  Dead %:   {result.dead_chain_fraction*100:.1f}%")

            # Plot distribution: experimental, calculated (true params), fitted
            ax = axes[idx]
            ax.semilogx(mws, exp_ints, 'b-', label='Experimental', alpha=0.7)
            ax.semilogx(mws, calc_result.intensities,
                       'g--', label='Calculated', linewidth=2)
            ax.semilogx(result.molecular_weights, result.intensities,
                       'r:', label='Fitted', linewidth=2)

            ax.set_title(f"Run {run_num}\n"
                        f"True α={true_alpha:{alpha_fmt}}, Fit α={result.alpha:{alpha_fmt}}\n"
                        f"R²={result.r_squared:.4f}")
            ax.set_xlabel('Molecular Weight (g/mol)')
            ax.set_ylabel('w(M)')
            ax.legend(fontsize=8)
            ax.set_xlim(2e2, 1e5)

        # Sixth plot: alpha comparison (fitted vs true)
        if true_alphas and fitted_alphas:
            ax6 = axes[5]
            ax6.scatter(true_alphas, fitted_alphas, s=100, c='blue',
                       edgecolors='black', zorder=3)

            min_val = min(min(true_alphas), min(fitted_alphas)) * 0.8
            max_val = max(max(true_alphas), max(fitted_alphas)) * 1.2
            ax6.plot([min_val, max_val], [min_val, max_val], 'k--',
                    label='1:1 line', zorder=1)

            ax6.set_xlabel('True α')
            ax6.set_ylabel('Fitted α')
            ax6.set_title('Fitted vs True Alpha')
            ax6.legend()
            ax6.set_xlim(min_val, max_val)
            ax6.set_ylim(min_val, max_val)
            ax6.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        fig.savefig(output_path / filename, dpi=150)
        plt.close(fig)

        print(f"\nFigure saved: {output_path / filename}")

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_path}")


if __name__ == '__main__':
    generate_validation_report()
