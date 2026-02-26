# polyterm: Polymer Termination Kinetics Analysis

A Python library for analyzing termination rates in fast-initiating polymerizations. Provides tools for generating molecular weight distributions from kinetic parameters and fitting experimental SEC/GPC data to determine termination kinetics.

## Features

- **Generate theoretical MWDs** from kinetic parameters
- **Fit experimental SEC/GPC data** to determine termination kinetics
- **Calibrate instrument broadening** using EMG or EGH models
- **Flexible termination models**: supports arbitrary termination orders
- **Comprehensive kinetic calculations** for living polymerizations with termination
- **Unit-agnostic**: works with any consistent unit system

## Installation

### From source

```bash
git clone https://github.com/yourusername/polyterm.git
cd polyterm
pip install -e .
```

### With development dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### Fitting Experimental Data

```python
import numpy as np
from polyterm import fit_mwd

# Load your SEC/GPC data
molecular_weights = np.array([...])  # in g/mol
intensities = np.array([...])        # detector response

# Fit kinetic model to determine termination parameters
result = fit_mwd(
    molecular_weights, intensities,
    order=1.5,              # termination reaction order
    monomer_mw=104.15,      # g/mol
    init_mon=1.0            # mol/L (or any consistent unit)
)

# View results
print(f"alpha (kt/kp) = {result.alpha:.6f}")
print(f"[I]_0 = {result.init:.6f} mol/L")
print(f"Conversion = {result.conversion:.2%}")
print(f"R^2 = {result.r_squared:.6f}")
print(f"Dead chains = {result.dead_chain_fraction:.2%}")
```

### Calibrating Instrument Broadening

Before fitting experimental data, calibrate your instrument using a narrow molecular weight standard:

```python
from polyterm import calibrate_emg_broadening, calibrate_egh_broadening

# Load narrow standard data
standard_mws = np.array([...])
standard_ints = np.array([...])

# Calibrate using EMG (Exponentially Modified Gaussian) model
emg_result = calibrate_emg_broadening(standard_mws, standard_ints)
print(f"sigma = {emg_result.sigma:.4f}")
print(f"tau = {emg_result.tau:.4f}")
print(f"R^2 = {emg_result.r_squared:.4f}")

# Or use EGH (Exponential-Gaussian Hybrid) model - more numerically stable
egh_result = calibrate_egh_broadening(standard_mws, standard_ints)
```

### Fitting with Calibrated Broadening

```python
from polyterm import fit_mwd, calibrate_egh_broadening

# Step 1: Calibrate from narrow standard
cal_result = calibrate_egh_broadening(standard_mws, standard_ints)

# Step 2: Fit experimental data with calibrated broadening
result = fit_mwd(
    mws, ints,
    order=1.5,
    monomer_mw=104.15,
    init_mon=1.0,
    sigma=cal_result.sigma,
    tau=cal_result.tau
)

print(f"kt/kp = {result.alpha:.4f}")
print(f"Conversion = {result.conversion:.2%}")
print(f"R^2 = {result.r_squared:.4f}")
```

### Generating Theoretical Distributions

```python
from polyterm import MolecularWeightDistribution
import numpy as np

# Define MW range
mws = np.logspace(3, 6, 500)  # 1k to 1M Da

# Generate MWD from kinetic parameters
mwd_theory = MolecularWeightDistribution.from_kinetics(
    molecular_weights=mws,
    monomer_mw=104.15,    # g/mol
    nu=50.0,              # kinetic chain length
    alpha=0.01,           # kt/kp ratio
    init_mon=1.0,         # mol/L
    init=0.02,            # mol/L
    order=1.5,            # termination order
    sigma=0.05            # SEC broadening
)

# Access properties
print(f"Mn = {mwd_theory.number_average_mw:.0f} g/mol")
print(f"Mw = {mwd_theory.weight_average_mw:.0f} g/mol")
print(f"D = {mwd_theory.dispersity:.3f}")
```

### Fitting with Known Parameters

```python
# Fit with known conversion (measured separately)
result = fit_mwd(
    mws, ints,
    order=1.5,
    monomer_mw=104.15,
    init_mon=1.0,
    conversion=0.95  # known from gravimetry
)

# Fit with known initiator concentration
result = fit_mwd(
    mws, ints,
    order=1.5,
    monomer_mw=104.15,
    init_mon=1.0,
    init=0.02  # known from formulation
)
```

### Batch Processing

```python
from functools import partial
from polyterm import fit_mwd

# Create partially applied function for your instrument
fit_my_instrument = partial(
    fit_mwd,
    monomer_mw=104.15,
    init_mon=1.0,
    sigma=0.05,
    tau=0.02
)

# Process multiple samples
results = [
    fit_my_instrument(mws, ints, order=1.5)
    for mws, ints in samples
]
```

## Important: Unit Consistency

**All parameters must use consistent units throughout.** The library is unit-agnostic, but mixing units will give incorrect results.

### Example: SI Units
```python
# If monomer_mw is in g/mol:
monomer_mw = 104.15      # g/mol
init_mon = 1.0           # mol/L
init = 0.02              # mol/L
# Then kp, kt must be in L/(mol*s) and molecular_weights in g/mol
```

### Example: Different Units
```python
# If monomer_mw is in Da:
monomer_mw = 104150      # Da (Daltons)
init_mon = 1000          # mmol/L
init = 20                # mmol/L
# Then molecular_weights must be in Da
```

## Theoretical Background

This library implements kinetic models for controlled living polymerizations with chain termination, assuming:

- **Fast initiation**: all chains start growing simultaneously
- **No chain transfer**: termination is the only chain-stopping event
- **First-order propagation**: growth rate proportional to [M][P*]
- **Nth-order termination**: termination rate proportional to [P*]^n

The molecular weight distribution is calculated by:
1. Computing living and dead chain distributions at discrete DPs
2. Applying broadening for SEC instrumental effects (EMG or EGH model)
3. Converting from number to weight distribution

### Key Parameters

- **alpha**: Ratio of termination to propagation rate constants (kt/kp)
- **nu**: Kinetic chain length, ([M]_0 - [M]) / [I]_0
- **order**: Termination reaction order (typically 1.0, 1.5, or 2.0)
- **sigma**: SEC line broadening (std dev in log MW space)
- **tau**: Exponential tailing parameter for asymmetric broadening
- **bn**: Inverse propagation order in living chains (usually 1.0)
- **combination**: Whether termination occurs by bimolecular combination

## API Reference

### Main Functions

#### `fit_mwd(molecular_weights, intensities, order, monomer_mw, init_mon, **kwargs) -> FitResult`

Fit kinetic model to molecular weight distribution data.

**Required Parameters:**
- `molecular_weights`: Array of MW values from SEC/GPC
- `intensities`: Array of detector response (weight fraction)
- `order`: Termination reaction order (e.g., 1.0, 1.5, 2.0)
- `monomer_mw`: Molecular weight of one monomer unit
- `init_mon`: Initial monomer concentration

**Optional Parameters:**
- `sigma`: SEC broadening parameter (if None, fitted)
- `tau`: EGH/EMG tailing parameter (requires sigma)
- `conversion`: Monomer conversion (if None, fitted)
- `init`: Initiator concentration (if None, fitted)
- `combination`: Whether termination by chain combination (default False)
- `bn`: Inverse propagation order (default 1.0)
- `max_fit_points`: Max points for fitting (default 400)

**Returns:** `FitResult` with fitted parameters and diagnostics

#### `calibrate_emg_broadening(molecular_weights, intensities, max_sigma=0.5, max_tau=0.2) -> EMGCalibrationResult`

Calibrate SEC broadening using EMG (Exponentially Modified Gaussian) model from a narrow standard.

**Returns:** `EMGCalibrationResult` with `sigma`, `tau`, `center`, `r_squared`

#### `calibrate_egh_broadening(molecular_weights, intensities, max_sigma=0.5, max_tau=0.2) -> EGHCalibrationResult`

Calibrate SEC broadening using EGH (Exponential-Gaussian Hybrid) model. More numerically stable for asymmetric peaks.

**Returns:** `EGHCalibrationResult` with `sigma`, `tau`, `center`, `r_squared`

#### `fit_living_peak(molecular_weights, intensities, broadening_type, sigma, tau=0.0) -> LivingPeakResult`

Fit the right edge of a MWD to determine living chain distribution. This function separates living and dead chain contributions using calibrated broadening parameters.

**Required Parameters:**
- `molecular_weights`: Array of MW values from SEC/GPC
- `intensities`: Array of detector response (weight fraction)
- `broadening_type`: Type of broadening ('gaussian', 'emg', or 'egh')
- `sigma`: Gaussian broadening parameter (from calibration)

**Optional Parameters:**
- `tau`: Tailing parameter for EMG/EGH (default 0.0, ignored for 'gaussian')

**Returns:** `LivingPeakResult` with living/dead distributions and metrics

**Example:**
```python
from polyterm import fit_living_peak, calibrate_egh_broadening

# Calibrate instrument from narrow standard
cal = calibrate_egh_broadening(standard_mws, standard_ints)

# Fit living peak to separate living/dead chains
result = fit_living_peak(
    sample_mws, sample_ints,
    broadening_type='egh',
    sigma=cal.sigma,
    tau=cal.tau
)

print(f"Living peak MW = {result.living_peak_mw:.0f}")
print(f"Dead chain fraction = {result.dead_chain_fraction:.2%}")
```

### Main Classes

#### `FitResult`

Immutable container for fitting results.

**Attributes:**
- `alpha`: Fitted kt/kp ratio
- `init`: Fitted initiator concentration
- `order`: Termination order used
- `sigma`: Broadening parameter
- `tau`: Tailing parameter (0 for Gaussian)
- `conversion`: Monomer conversion (0-1)
- `r_squared`: Coefficient of determination
- `molecular_weights`: MW array used in fit
- `predicted_intensities`: Model predictions
- `dead_chain_intensities`: Dead chain contribution
- `dead_chain_fraction`: Fraction of terminated chains
- `fit_message`: Optimizer status message
- `fun`: Final objective value
- `jac`: Jacobian at solution
- `hess_inv`: Inverse Hessian approximation

#### `MolecularWeightDistribution`

Immutable container for MWD data with analysis methods.

**Class Methods:**
- `from_data(molecular_weights, intensities, monomer_mw, normalize=True)`: Create from experimental data
- `from_kinetics(molecular_weights, monomer_mw, nu, alpha, init_mon, init, order, sigma, tau=0, ...)`: Generate from kinetic theory

**Properties:**
- `number_average_dp`: Number average degree of polymerization
- `number_average_mw`: Number average molecular weight (Mn)
- `weight_average_mw`: Weight average molecular weight (Mw)
- `dispersity`: D = Mw/Mn
- `peak_molecular_weight`: MW at peak intensity

**Methods:**
- `normalize()`: Return normalized copy
- `downsample(max_points)`: Return downsampled copy
- `normalize_on_log_scale()`: Normalize on log(MW) scale

#### `EMGCalibrationResult` / `EGHCalibrationResult`

Immutable containers for calibration results.

**Attributes:**
- `sigma`: Gaussian standard deviation in log(MW) space
- `tau`: Exponential decay/tailing parameter
- `center`: Fitted peak center MW
- `r_squared`: Fit quality metric

#### `LivingPeakResult`

Immutable container for living peak fitting results.

**Attributes:**
- `living_intensities`: Fitted living chain distribution (weight fractions)
- `dead_intensities`: Dead chain distribution (experimental - living)
- `dead_chain_fraction`: Fraction of terminated chains (0-1)
- `living_peak_mw`: Molecular weight at the living chain peak
- `r_squared`: Coefficient of determination for right edge fit
- `molecular_weights`: MW array used in fit
- `broadening_type`: Type of broadening used ('gaussian', 'emg', 'egh')
- `sigma`: Gaussian broadening parameter
- `tau`: Tailing parameter (0 for Gaussian)
- `coefficient`: Scaling coefficient from fit
- `fit_message`: Optimizer status message

### Utility Functions

- `calculate_number_average_dp(mws, intensities, monomer_mw)`: Calculate DPn from distribution
- `fit_right_edge(mws, intensities, monomer_mw)`: Estimate peak DP and broadening from right edge
- `calculate_r_squared(observed, predicted)`: Calculate coefficient of determination

### Core Functions (Advanced Users)

Low-level kinetic functions for custom calculations:

- `monomer_conversion(times, kp, kt, init_mon, init, order, bn)`: Monomer concentration vs time
- `living_chain_concentration(init, order, time)`: [P*] vs reduced time
- `living_chain_dp(alpha, init_mon, init, order, time, bn)`: DP of living chains
- `conversion_to_time(alpha, init, order, conversion, bn)`: Convert conversion to reduced time
- `calculate_distribution(dps, nu, alpha, ...)`: Calculate DP distributions
- `calculate_mwd(mws, monomer_mw, nu, alpha, ...)`: Calculate complete MWD with broadening

## Examples

### Example 1: Complete Workflow

```python
import numpy as np
from polyterm import (
    fit_mwd,
    calibrate_egh_broadening,
    MolecularWeightDistribution
)

# Load calibration standard data
standard_mws = np.loadtxt('standard_mws.txt')
standard_ints = np.loadtxt('standard_ints.txt')

# Calibrate instrument
cal = calibrate_egh_broadening(standard_mws, standard_ints)
print(f"Calibration: sigma={cal.sigma:.4f}, tau={cal.tau:.4f}, R^2={cal.r_squared:.4f}")

# Load experimental data
mws = np.loadtxt('sample_mws.txt')
ints = np.loadtxt('sample_ints.txt')

# Fit kinetics with calibrated broadening
result = fit_mwd(
    mws, ints,
    order=1.5,
    monomer_mw=104.15,
    init_mon=1.0,
    sigma=cal.sigma,
    tau=cal.tau
)

print(f"\nFitted Parameters:")
print(f"alpha (kt/kp) = {result.alpha:.6f}")
print(f"[I]_0 = {result.init:.6f} mol/L")
print(f"Conversion = {result.conversion:.2%}")
print(f"R^2 = {result.r_squared:.6f}")
print(f"Dead chain fraction = {result.dead_chain_fraction:.2%}")
```

### Example 2: Comparing Theory and Experiment

```python
import numpy as np
import matplotlib.pyplot as plt
from polyterm import fit_mwd, MolecularWeightDistribution

# Load experimental data
mwd_exp = MolecularWeightDistribution.from_data(
    molecular_weights=mws_exp,
    intensities=ints_exp,
    monomer_mw=104.15
)

# Fit model
result = fit_mwd(
    mws_exp, ints_exp,
    order=1.5,
    monomer_mw=104.15,
    init_mon=1.0,
    sigma=0.05
)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.semilogx(mwd_exp.molecular_weights, mwd_exp.intensities,
             'o', label='Experimental', markersize=4)
plt.semilogx(result.molecular_weights, result.predicted_intensities,
             '-', label=f'Fit (R^2 = {result.r_squared:.4f})', linewidth=2)
plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Weight Fraction')
plt.legend()
plt.title(f'alpha = {result.alpha:.6f}, order = {result.order:.2f}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Example 3: Exploring Parameter Space

```python
import numpy as np
from polyterm import MolecularWeightDistribution

# Define MW range
mws = np.logspace(3, 6, 500)

# Generate MWDs for different alpha values
alphas = [0.001, 0.01, 0.1]

for alpha in alphas:
    mwd = MolecularWeightDistribution.from_kinetics(
        molecular_weights=mws,
        monomer_mw=104.15,
        nu=100,
        alpha=alpha,
        init_mon=1.0,
        init=0.01,
        order=1.5,
        sigma=0.05
    )
    print(f"alpha = {alpha:.4f}: D = {mwd.dispersity:.3f}")
```

## Testing

Run the test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=polyterm --cov-report=html
```

## Limitations

- Assumes fast, complete initiation
- No chain transfer reactions
- Single monomer type
- Termination is irreversible
- SEC broadening modeled as EMG or EGH

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite:

```
[Your publication details here]
```

## License

MIT License - see LICENSE file for details.

## Authors

[Your name and affiliation]

## Acknowledgments

This library implements kinetic models from the polymer science literature on controlled living polymerizations with termination.
