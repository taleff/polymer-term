"""
Notebook reference: MTT_2_125 (GPC data), MTT_2_135 (analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
from data_import import load_all_traces, peak_molecular_weight

plt.style.use('../example_figures/figure.mplstyle')
from polyterm import estimate_alpha, monomer_conversion


# --- Load data ---
# Timed aliquots from anionic polymerization of styrene
DATA_FILE = '../example_data/anionic_styrene_timed_aliquots.csv'
CAL_FILE = '../example_data/calibrations.json'
CAL_NAME = 'ri_2025_11_21'

traces = load_all_traces(
    DATA_FILE, CAL_FILE, CAL_NAME,
    bounds=(5e2, 1e5)
)

# --- Experimental parameters ---
monomer_mw = 104.15  # g/mol (styrene)
init_mon = 1.00      # M (initial monomer concentration)
init = 0.00734       # M (initiator concentration)
kp = 11.6            # L/mol/min (propagation rate constant)
kt = 0.0773          # 1/min (termination rate constant)

# --- Calculate conversions and peak MWs ---
# Aliquots were taken at 2 min intervals
times = np.arange(2, 21, 2)

# Calculate conversion at each time point using the known kinetics
# monomer_conversion returns the remaining monomer fraction
convs = 1 - monomer_conversion(times, kp, kt, init_mon, init, order=1.0)

# Extract peak molecular weight from each trace
peaks = np.array([
    peak_molecular_weight(mws, ints)
    for mws, ints in traces
])

# --- Estimate alpha ---
result = estimate_alpha(
    convs, peaks, monomer_mw, init_mon, init, order=1.0
)

# Compare to the known value
alpha_set = kt / kp
print(f'Set alpha:       {alpha_set:.5f} M')
print(f'Estimated alpha: {result["alpha"]:.5f} M')
print(f'R-squared:       {result["r_squared"]:.4f}')

# --- Plot results ---
# Theoretical Mn for a perfectly living polymerization
pred_mn = monomer_mw * (init_mon / init)

fig, ax = plt.subplots(figsize=(5.5, 5), layout='constrained')

# Theoretical Mn line (perfectly living)
ax.plot(
    [0, 1], [0, pred_mn / 1000], 'r-',
    alpha=0.4, label='Theory Mn'
)

# Predicted living Mn from the alpha fit
ax.plot(
    convs, result['predicted_mns'] / 1000, 'b-',
    alpha=0.5, label=f'Theory Mp'
)

# Experimental peak MWs
ax.plot(
    convs, peaks / 1000, 'ks',
    markersize=6, label='Meas. Mp'
)

ax.set_xlabel('Conversion')
ax.set_ylabel('Mol. Weight (kg/mol)')
ax.set_xlim((0, 0.7))
ax.set_ylim((0, 18))
ax.legend(loc='upper left', fontsize=18)
fig.savefig('../example_figures/estimate_alpha.svg')
