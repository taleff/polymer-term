"""
Notebook reference: MTT_2_116
"""

import numpy as np
import matplotlib.pyplot as plt
from data_import import load_gpc_trace

plt.style.use('../example_figures/figure.mplstyle')
from polyterm import calibrate_egh_broadening, compute_poisson_broadened_mwd


# --- Load data ---
# Polystyrene standard (~90 kDa) from Pressure Chemical
DATA_FILE = '../example_data/polystyrene_standard_90kda.csv'
CAL_FILE = '../example_data/calibrations.json'
CAL_NAME = 'ri_2025_11_21'

mws, ints = load_gpc_trace(
    DATA_FILE, CAL_FILE, CAL_NAME,
    bounds=(5e3, 8e5)
)

# --- Calibrate broadening ---
# The calibration fits an Exponential Gaussian Hybrid (EGH) peak
# to the standard. Since this is a polystyrene standard with high
# DP, the Poisson contribution to peak width is negligible, so
# monomer_mw is not needed.
result = calibrate_egh_broadening(mws, ints)

print(f'Sigma:    {result.sigma:.4f}')
print(f'Tau:      {result.tau:.4f}')
print(f'Center:   {result.center:.1f} g/mol')
print(f'R-squared: {result.r_squared:.4f}')

# --- Calibrate with Poisson correction ---
# When calibrating with a living polymer standard at low DP, the
# Poisson chain length distribution contributes measurable width.
# Providing monomer_mw accounts for this, yielding more accurate
# broadening parameters.
ps_monomer_mw = 104.15  # g/mol (styrene)
result_poisson = calibrate_egh_broadening(
    mws, ints, monomer_mw=ps_monomer_mw
)

print(f'\nWith Poisson correction (monomer_mw={ps_monomer_mw}):')
print(f'Sigma:    {result_poisson.sigma:.4f}')
print(f'Tau:      {result_poisson.tau:.4f}')
print(f'Center:   {result_poisson.center:.1f} g/mol')
print(f'R-squared: {result_poisson.r_squared:.4f}')

# --- Plot the fit ---
# Generate the theoretical broadened peak using the fitted
# parameters for visual comparison
broadened = compute_poisson_broadened_mwd(
    mws, result_poisson.center / ps_monomer_mw,
    ps_monomer_mw, result_poisson.sigma, result_poisson.tau
)

fig, ax = plt.subplots(figsize=(5, 5.5), layout='constrained')
ax.plot(mws, ints / np.max(ints), 'k-', label='Measured')
ax.plot(mws, broadened / np.max(broadened), 'b--', label='EGH Fit')
ax.set_xscale('log')
ax.set_xlabel('Molecular Weight (g/mol)')
ax.set_ylabel('Intensity (A.U.)')
ax.set_xlim((5e3, 8e5))
ax.set_ylim((-0.1, 1.2))
ax.legend(loc='upper left', fontsize=18)
fig.savefig('../example_figures/calibrate_egh_broadening.svg')
