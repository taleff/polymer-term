"""
Notebook reference: MTT_2_125
"""

import numpy as np
import matplotlib.pyplot as plt
from data_import import load_all_traces

plt.style.use('../example_figures/figure.mplstyle')
from polyterm import estimate_death


# --- Load data ---
# Timed aliquots from anionic polymerization of styrene with
# continuous quenching at 2 min intervals (2, 4, ..., 20 min)
DATA_FILE = '../example_data/anionic_styrene_timed_aliquots.csv'
CAL_FILE = '../example_data/calibrations.json'
CAL_NAME = 'ri_2025_11_21'

traces = load_all_traces(
    DATA_FILE, CAL_FILE, CAL_NAME,
    bounds=(5e2, 1e5)
)

# --- Broadening parameters ---
# From calibration with polystyrene standards (see calibrate_egh
# example)
sigma = 0.128
tau = 0.0456
monomer_mw = 104.15  # g/mol (styrene)

# --- Estimate dead chain fraction for each aliquot ---
times = np.arange(2, 21, 2)  # minutes
dead_fracs = np.zeros(len(traces))

fig, axs = plt.subplots(2, 5, figsize=(25, 10), layout='constrained')

for i, (mws, ints) in enumerate(traces):
    result = estimate_death(
        mws, ints, sigma=sigma, tau=tau, monomer_mw=monomer_mw
    )

    dead_fracs[i] = result.dead_chain_fraction
    norm = np.max(result.intensities)

    # Plot decomposition: total, living, dead
    row, col = i // 5, i % 5
    axs[row, col].plot(
        result.molecular_weights,
        result.intensities / norm,
        'k-', label='Total'
    )
    axs[row, col].plot(
        result.molecular_weights,
        result.live_chain_intensities / norm,
        'g-', label='Living'
    )
    axs[row, col].plot(
        result.molecular_weights,
        result.dead_chain_intensities / norm,
        'r-', label='Dead'
    )
    axs[row, col].set_xscale('log')
    axs[row, col].set_xlim((5e2, 1e5))
    axs[row, col].set_ylim((-0.1, 1.2))
    axs[row, col].annotate(
        f't={times[i]} min; Dead: {dead_fracs[i]:.0%}',
        (6e2, 1.05)
    )

    print(f't={times[i]:2d} min: Dead fraction = {dead_fracs[i]:.1%}')

fig.supxlabel('Molecular Weight (g/mol)')
fig.supylabel('Intensity (A.U.)')
fig.savefig('../example_figures/estimate_death.svg')

# --- Plot dead fraction vs time ---
# For first order termination, -ln(1 - dead_fraction) should be
# linear with time, with slope equal to kt
fig2, ax2 = plt.subplots(figsize=(5.5, 5), layout='constrained')
ax2.plot(times, -np.log(1 - dead_fracs), 'k.', markersize=12)

# Theoretical line using known kt
kt = 0.0773  # 1/min
ax2.plot([0, 25], [0, 25 * kt], 'r--', label=f'Theory (kt={kt})')

ax2.set_xlabel('Time (min)')
ax2.set_ylabel(r'$-\ln([I]/[I]_0)$')
ax2.set_xlim((0, 25))
ax2.set_ylim((0, 2.0))
fig2.savefig('../example_figures/estimate_death_kinetics.svg')
