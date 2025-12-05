# polyterm: Polymer Termination Kinetics Analysis

A Python library for analyzing termination rates in fast-initiating polymerizations. Provides tools for generating molecular weight distributions from kinetic parameters and fitting experimental SEC/GPC data to determine termination kinetics.

## Features

- **Generate theoretical MWDs** from kinetic parameters
- **Fit experimental SEC/GPC data** to determine termination kinetics
- **Flexible termination models**: single-order termination (most common)
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
from polyterm import MolecularWeightDistribution, SingleOrderModel

# Load your SEC/GPC data
molecular_weights = np.array([...])  # in g/mol
intensities = np.array([...])        # detector response

# Create MWD object
mwd = MolecularWeightDistribution.from_data(
    molecular_weights=molecular_weights,
    intensities=intensities,
    monomer_mw=104.15,  # g/mol
    normalize=True
)

# Create and fit model
model = SingleOrderModel(
    monomer_mw=104.15,  # g/mol
    init_mon=1.0,       # mol/L (or any consistent unit)
    order=1.5           # termination reaction order
)

result = model.fit(mwd)

# View results
print(f"α (kt/kp) = {result.alpha:.6f}")
print(f"[I]₀ = {result.init:.6f} mol/L")
print(f"Conversion = {result.conversion:.2%}")
print(f"R² = {result.r_squared:.6f}")
print(f"Dead chains = {result.dead_chain_fraction:.2%}")
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
print(f"Đ = {mwd_theory.dispersity:.3f}")
```

### Fitting with Unknown Parameters

```python
# Fit order, conversion, initiator, α, and σ simultaneously
model = SingleOrderModel(
    monomer_mw=104.15,
    init_mon=1.0
)
result = model.fit(mwd)
print(f"Fitted order = {result.order:.3f}")

# Fit with known conversion
model = SingleOrderModel(
    monomer_mw=104.15,
    init_mon=1.0,
    conversion=0.95,  # known from separate measurement
    order=1.5
)
result = model.fit(mwd)
```

## Important: Unit Consistency

**All parameters must use consistent units throughout.** The library is unit-agnostic, but mixing units will give incorrect results.

### Example: SI Units
```python
# If monomer_mw is in g/mol:
monomer_mw = 104.15      # g/mol
init_mon = 1.0           # mol/L
init = 0.02              # mol/L
# Then kp, kt must be in L/(mol·s) and molecular_weights in g/mol
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
- **First-order propagation**: growth rate ∝ [M][P*]
- **Nth-order termination**: termination rate ∝ [P*]ⁿ

The molecular weight distribution is calculated by:
1. Computing living and dead chain distributions at discrete DPs
2. Applying Gaussian broadening for SEC instrumental effects
3. Converting from number to weight distribution

### Key Parameters

- **α (alpha)**: Ratio of termination to propagation rate constants (kt/kp)
- **ν (nu)**: Kinetic chain length, ([M]₀ - [M]) / [I]₀
- **order**: Termination reaction order (typically 1.0, 1.5, or 2.0)
- **σ (sigma)**: SEC line broadening (std dev in log MW space)
- **bn**: Inverse propagation order in living chains (usually 1.0)

## API Reference

### Main Classes

#### `MolecularWeightDistribution`
Container for MWD data with properties and analysis methods.

**Class Methods:**
- `from_data(molecular_weights, intensities, monomer_mw)`: Create from experimental data
- `from_kinetics(molecular_weights, monomer_mw, nu, alpha, ...)`: Generate from kinetics

**Properties:**
- `number_average_dp`: Number average degree of polymerization
- `number_average_mw`: Number average molecular weight (Mn)
- `weight_average_mw`: Weight average molecular weight (Mw)
- `dispersity`: Đ = Mw/Mn
- `peak_molecular_weight`: MW at peak intensity

**Methods:**
- `normalize()`: Return normalized copy
- `downsample(max_points)`: Return downsampled copy
- `normalize_on_log_scale()`: Normalize on log(MW) scale

#### `SingleOrderModel`
Fit kinetic parameters assuming single termination pathway.

**Parameters:**
- `monomer_mw`: Monomer molecular weight
- `init_mon`: Initial monomer concentration
- `order`: Termination order (None to fit)
- `conversion`: Monomer conversion (None to fit)
- `init`: Initial initiator concentration (None to fit)
- `combination`: Whether termination is by combination (default False)
- `bn`: Inverse propagation order (default 1.0)
- `max_fit_points`: Max points for fitting (default 400)

**Methods:**
- `fit(mwd)`: Fit model to MWD, returns `FitResult`

#### `FitResult`
Container for fitting results.

**Attributes:**
- `alpha`: Fitted kt/kp ratio
- `init`: Fitted initiator concentration
- `order`: Fitted or specified termination order
- `sigma`: Fitted line broadening
- `conversion`: Monomer conversion
- `r_squared`: Coefficient of determination
- `molecular_weights`: MWs used in fit
- `predicted_intensities`: Model predictions
- `dead_chain_fraction`: Fraction of terminated chains
- `fit_message`: Optimizer status message

**Methods:**
- `predict(molecular_weights, monomer_mw, init_mon, ...)`: Generate predicted MWD

### Utility Functions

- `calculate_number_average_dp(mws, intensities, monomer_mw)`: Calculate DPn
- `fit_right_edge(mws, intensities, monomer_mw)`: Estimate peak DP and broadening
- `calculate_r_squared(observed, predicted)`: Calculate R²

### Core Functions

Low-level kinetic functions (for advanced users):

- `monomer_conversion(times, kp, kt, init_mon, init, order, bn)`: Monomer concentration vs time
- `living_chain_concentration(init, order, time)`: [P*] vs time
- `living_chain_dp(alpha, init_mon, init, order, time, bn)`: DP of living chains
- `conversion_to_time(alpha, init, order, conversion, bn)`: Convert conversion to time
- `calculate_distribution(dps, nu, alpha, ...)`: Calculate DP distribution
- `calculate_mwd(mws, monomer_mw, nu, alpha, ...)`: Calculate complete MWD

## Examples

### Example 1: Fitting Living Polymerization Data

```python
import numpy as np
from polyterm import MolecularWeightDistribution, SingleOrderModel

# Experimental data from SEC
mws = np.loadtxt('molecular_weights.txt')
intensities = np.loadtxt('intensities.txt')

# Create MWD
mwd = MolecularWeightDistribution.from_data(
    molecular_weights=mws,
    intensities=intensities,
    monomer_mw=104.15
)

print(f"Experimental Mn = {mwd.number_average_mw:.0f} g/mol")
print(f"Experimental Đ = {mwd.dispersity:.3f}")

# Fit kinetics
model = SingleOrderModel(
    monomer_mw=104.15,
    init_mon=1.0,
    order=1.5
)
result = model.fit(mwd)

print(f"\nFitted Parameters:")
print(f"α (kt/kp) = {result.alpha:.6f}")
print(f"[I]₀ = {result.init:.6f} mol/L")
print(f"Conversion = {result.conversion:.2%}")
print(f"σ = {result.sigma:.6f}")
print(f"R² = {result.r_squared:.6f}")
```

### Example 2: Comparing Theory and Experiment

```python
import numpy as np
import matplotlib.pyplot as plt
from polyterm import MolecularWeightDistribution, SingleOrderModel

# Load experimental data
mwd_exp = MolecularWeightDistribution.from_data(
    molecular_weights=mws_exp,
    intensities=ints_exp,
    monomer_mw=104.15
)

# Fit model
model = SingleOrderModel(monomer_mw=104.15, init_mon=1.0, order=1.5)
result = model.fit(mwd_exp)

# Generate smooth predicted curve
mws_smooth = np.logspace(3, 6, 1000)
mwd_pred = result.predict(
    molecular_weights=mws_smooth,
    monomer_mw=104.15,
    init_mon=1.0
)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.semilogx(mwd_exp.molecular_weights, mwd_exp.intensities,
             'o', label='Experimental', markersize=4)
plt.semilogx(mws_smooth, mwd_pred.intensities,
             '-', label=f'Fit (R² = {result.r_squared:.4f})', linewidth=2)
plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Weight Fraction')
plt.legend()
plt.title(f'α = {result.alpha:.6f}, order = {result.order:.2f}')
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

# Generate MWDs for different α values
alphas = [0.001, 0.01, 0.1]
mwds = []

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
    mwds.append(mwd)
    print(f"α = {alpha:.4f}: Đ = {mwd.dispersity:.3f}, "
          f"Dead chains = {(1-mwd.intensities[0]/max(mwd.intensities))*100:.1f}%")
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

## Performance Notes

- Fitting typically takes 5-15 seconds per MWD on modern hardware
- Downsampling to 400 points (default) significantly speeds up fitting
- Large maximum DP (>5000) can slow calculations
- For very high conversions, numerical integration may require more time

## Limitations

- Assumes fast, complete initiation
- No chain transfer reactions
- Single monomer type
- Termination is irreversible
- SEC broadening modeled as log-normal Gaussian

## Future Development

- Multi-order termination model (multiple simultaneous pathways)
- Global fitting across multiple experiments
- Alternative optimization algorithms for difficult fits
- Support for constrained fitting (parameter bounds)

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
