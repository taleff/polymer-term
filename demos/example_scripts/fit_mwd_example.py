"""
Notebook reference: MTT_2_113 (GPC data), MTT_2_121 (analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
from data_import import load_gpc_trace

plt.style.use('../example_figures/figure.mplstyle')
from polyterm import fit_mwd


# --- Load data ---
# GTP of MMA, target DP 500 (third trace in the multi-sample file)
DATA_FILE = '../example_data/gtp_mma_dp300_dp400_dp500.csv'
CAL_FILE = '../example_data/calibrations.json'
CAL_NAME = 'ri_2025_11_21'

mws, ints = load_gpc_trace(
    DATA_FILE, CAL_FILE, CAL_NAME,
    bounds=(1e4, 2e6),
    trace_index=2  # DP 500 sample
)

# --- Experimental parameters ---
# The GPC is calibrated with polystyrene standards, so we need
# to convert to PMMA-equivalent molecular weights using
# Mark-Houwink parameters
ps_k, ps_a = 0.00151, 0.706   # polystyrene
pmma_k, pmma_a = 0.00122, 0.690  # poly(methyl methacrylate)
monomer_mw = ((pmma_k * 100.15 ** (1 + pmma_a)) / ps_k) ** (1 / (1 + ps_a))

init_mon = 1.0  # M (initial monomer concentration)
order = 1.0     # first order termination for GTP

# Broadening parameters from calibration (see calibrate_egh example)
sigma = 0.128
tau = 0.0456

# --- Fit the MWD ---
# When sigma and tau are provided, broadening is fixed and only
# kinetic parameters are fitted. This is the recommended approach.
result = fit_mwd(
    mws, ints, order, monomer_mw, init_mon,
    sigma=sigma, tau=tau
)

print(f'Alpha (kt/kp): {result.alpha:.4e}')
print(f'Initiator:      {result.init:.5f} M')
print(f'Conversion:     {result.conversion:.2%}')
print(f'Dead fraction:  {result.dead_chain_fraction:.1%}')
print(f'R-squared:      {result.r_squared:.4f}')

# --- Plot the fit ---
fig, ax = plt.subplots(figsize=(5.5, 5), layout='constrained')

# Normalize both to peak intensity
norm_data = np.max(ints)
norm_fit = np.max(result.intensities)

ax.plot(mws, ints / norm_data, 'k-', label='Measured')
ax.plot(
    result.molecular_weights,
    result.intensities / norm_fit,
    'b-', label='Fit'
)
ax.plot(
    result.molecular_weights,
    result.live_chain_intensities / norm_fit,
    'g--', alpha=0.6, label='Living chains'
)
ax.plot(
    result.molecular_weights,
    result.dead_chain_intensities / norm_fit,
    'r--', alpha=0.6, label='Dead chains'
)
ax.set_xscale('log')
ax.set_xlabel('Molecular Weight (g/mol)')
ax.set_ylabel('Intensity (A.U.)')
ax.set_xlim((1e4, 2e6))
ax.set_ylim((-0.1, 1.1))
ax.legend(loc='upper left', fontsize=15)
fig.savefig('../example_figures/fit_mwd.svg')
