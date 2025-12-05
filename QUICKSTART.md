# polyterm Quick Start Guide

## Installation

```bash
cd polymer-term
pip install -e .
```

## 5-Minute Tutorial

### 1. Import the library

```python
import numpy as np
from polyterm import MolecularWeightDistribution, SingleOrderModel
```

### 2. Load your SEC/GPC data

```python
# Your molecular weights and detector intensities
molecular_weights = np.array([...])  # e.g., from CSV file
intensities = np.array([...])         # e.g., from CSV file

# Create MWD object
mwd = MolecularWeightDistribution.from_data(
    molecular_weights=molecular_weights,
    intensities=intensities,
    monomer_mw=104.15  # styrene, for example
)
```

### 3. Examine your distribution

```python
print(f"Number average MW: {mwd.number_average_mw:.0f} g/mol")
print(f"Weight average MW: {mwd.weight_average_mw:.0f} g/mol")
print(f"Dispersity (Đ): {mwd.dispersity:.3f}")
print(f"Peak MW: {mwd.peak_molecular_weight:.0f} g/mol")
```

### 4. Fit termination kinetics

```python
# Create model with known parameters
model = SingleOrderModel(
    monomer_mw=104.15,     # g/mol
    init_mon=1.0,          # mol/L (your initial concentration)
    order=1.5              # termination order (1.0, 1.5, or 2.0)
)

# Fit the model
result = model.fit(mwd)

# View results
print(f"\nFitted Parameters:")
print(f"α (kt/kp) = {result.alpha:.6f}")
print(f"[I]₀ = {result.init:.6f} mol/L")
print(f"Conversion = {result.conversion:.1%}")
print(f"σ (broadening) = {result.sigma:.6f}")
print(f"R² = {result.r_squared:.6f}")
print(f"Dead chains = {result.dead_chain_fraction:.1%}")
```

### 5. Generate predicted distribution

```python
# Create smooth curve for plotting
mws_smooth = np.logspace(3, 6, 1000)
mwd_predicted = result.predict(
    molecular_weights=mws_smooth,
    monomer_mw=104.15,
    init_mon=1.0
)

# Now plot experimental vs predicted
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.semilogx(mwd.molecular_weights, mwd.intensities,
             'o', label='Experimental', markersize=4)
plt.semilogx(mws_smooth, mwd_predicted.intensities,
             '-', label=f'Fit (R² = {result.r_squared:.4f})', linewidth=2)
plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Weight Fraction')
plt.legend()
plt.title(f'α = {result.alpha:.6f}')
plt.grid(True, alpha=0.3)
plt.show()
```

## Common Use Cases

### Case 1: Fit everything (no known parameters except monomer MW)

```python
model = SingleOrderModel(
    monomer_mw=104.15,
    init_mon=1.0  # You must know this from your experiment
)
result = model.fit(mwd)

print(f"Fitted order: {result.order:.3f}")
print(f"Fitted conversion: {result.conversion:.1%}")
print(f"Fitted [I]₀: {result.init:.6f} mol/L")
```

### Case 2: Known conversion from separate measurement

```python
model = SingleOrderModel(
    monomer_mw=104.15,
    init_mon=1.0,
    conversion=0.95,  # Measured by NMR, for example
    order=1.5
)
result = model.fit(mwd)
```

### Case 3: Known initiator concentration

```python
model = SingleOrderModel(
    monomer_mw=104.15,
    init_mon=1.0,
    init=0.02,  # You know you added 0.02 mol/L initiator
    order=1.5
)
result = model.fit(mwd)
```

### Case 4: Generate theoretical MWD

```python
# No experimental data, just want to see what the distribution looks like
mws = np.logspace(3, 6, 500)

mwd_theory = MolecularWeightDistribution.from_kinetics(
    molecular_weights=mws,
    monomer_mw=104.15,
    nu=100,           # kinetic chain length
    alpha=0.01,       # kt/kp ratio
    init_mon=1.0,
    init=0.01,
    order=1.5,
    sigma=0.05        # SEC broadening
)

plt.semilogx(mws, mwd_theory.intensities)
plt.xlabel('Molecular Weight (g/mol)')
plt.ylabel('Weight Fraction')
plt.title(f'Theoretical MWD (Đ = {mwd_theory.dispersity:.3f})')
plt.show()
```

## Tips

### Downsampling for Speed

Fitting is faster with fewer points:

```python
# Downsample before fitting (will be done automatically, but you can control it)
mwd_downsampled = mwd.downsample(max_points=100)

# Or set in model
model = SingleOrderModel(
    monomer_mw=104.15,
    init_mon=1.0,
    order=1.5,
    max_fit_points=100  # default is 400
)
```

### Unit Consistency

**IMPORTANT**: All units must be consistent!

```python
# Example: Everything in SI units
monomer_mw = 104.15      # g/mol
init_mon = 1.0           # mol/L
init = 0.02              # mol/L
molecular_weights = ...  # g/mol (from SEC)

# Example: Different but consistent units
monomer_mw = 104150      # Da (Daltons)
init_mon = 1000          # mmol/L
init = 20                # mmol/L
molecular_weights = ...  # Da (from SEC, converted from g/mol)
```

### Checking Fit Quality

Good fits should have:
- R² > 0.95 for clean experimental data
- R² > 0.90 for noisy data
- Reasonable parameter values (α > 0, 0 < order < 2.5, etc.)

If R² is low:
1. Check your monomer MW is correct
2. Check your initial monomer concentration
3. Try fitting order if you fixed it
4. Check for experimental artifacts in SEC trace

### Plotting Results

```python
# Compare experimental and fitted data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax1.plot(mwd.molecular_weights, mwd.intensities, 'o', label='Experimental')
ax1.plot(result.molecular_weights, result.predicted_intensities, '-', label='Fit')
ax1.set_xlabel('Molecular Weight (g/mol)')
ax1.set_ylabel('Weight Fraction')
ax1.legend()
ax1.set_title('Linear Scale')

# Log scale
ax2.semilogx(mwd.molecular_weights, mwd.intensities, 'o', label='Experimental')
ax2.semilogx(result.molecular_weights, result.predicted_intensities, '-', label='Fit')
ax2.set_xlabel('Molecular Weight (g/mol)')
ax2.set_ylabel('Weight Fraction')
ax2.legend()
ax2.set_title(f'Log Scale (R² = {result.r_squared:.4f})')

plt.tight_layout()
plt.show()
```

## Troubleshooting

### ImportError
```
pip install -e .
```

### Fit fails or gives nonsense results
- Check unit consistency
- Try providing more known parameters (order, conversion, etc.)
- Check if SEC trace has artifacts (baseline drift, air peaks, etc.)
- Ensure monomer MW is correct

### Fit is slow
- Reduce max_fit_points (default 200, try 100 or 50)
- Downsample your MWD before fitting
- Note: First fit is slower due to JIT compilation

### R² is low but parameters look reasonable
- May indicate model assumptions don't perfectly match reality
- Try different termination orders
- Consider if other processes (chain transfer, slow initiation) are occurring

## Next Steps

- Read the full README.md for detailed documentation
- Check IMPLEMENTATION_NOTES.md for technical details
- Run the test suite to see more examples: `pytest tests/`
- Explore the examples in README.md

## Getting Help

- Check the documentation in README.md
- Look at test files for usage examples
- Report issues on GitHub

Happy fitting!
