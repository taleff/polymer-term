"""
Notebook reference: MTT_2_104
"""

import numpy as np
import matplotlib.pyplot as plt
from polyterm import calculate_mwd

plt.style.use('../example_figures/figure.mplstyle')


# --- Kinetic parameters ---
# Simulated polymerization with moderate termination
# Target DP 200, ratio of Rp to Rt is 500 initially
monomer_mw = 100.0      # g/mol
init_mon = 1.0           # M (initial monomer concentration)
init = 1.0 / 200         # M (initiator concentration for DP 200)
alpha = 1.0 / 500        # kt/kp ratio (Rp/Rt = 500)
order = 1.0              # first order termination
sigma = 0.128            # SEC broadening parameter
tau = 0.0456             # SEC tailing parameter

# Molecular weight axis
mws = np.logspace(2, 5, 1000)

# --- Calculate MWD at different conversions ---
# The maximum monomer conversion for these parameters is
# 1 - exp(-init/alpha). We sweep fractions of this maximum.
max_conv = 1 - np.exp(-init / alpha)
fractions = [0.20, 0.40, 0.60, 0.80, 0.99]
conversions = [f * max_conv for f in fractions]

fig, axs = plt.subplots(
    1, len(conversions),
    figsize=(4.5 * len(conversions), 5),
    sharey=True, layout='constrained'
)

for i, conv in enumerate(conversions):
    result = calculate_mwd(
        mws, monomer_mw, init_mon, alpha,
        init, conv, order, sigma, tau
    )

    # Normalize to peak intensity
    norm = np.max(result.intensities)

    # Use monomer conversion for the bar visualization,
    # matching the convention in MTT_2_104
    live_frac = 1 - conv

    # Plot total, living, and dead distributions
    axs[i].plot(
        result.molecular_weights,
        result.intensities / norm,
        '-', color='black'
    )
    axs[i].plot(
        result.molecular_weights,
        result.live_chain_intensities / norm,
        '--', color='#006230'
    )
    axs[i].plot(
        result.molecular_weights,
        result.dead_chain_intensities / norm,
        '--', color='#DC0F0F'
    )
    axs[i].set_xscale('log')
    axs[i].set_xlim((3e2, 1e5))
    axs[i].set_ylim((-0.1, 1.1))

    # Colored rectangles showing live/dead fraction
    axs[i].add_patch(
        plt.Rectangle(
            (1.3e5, -0.1), 0.4e5, 1.2 * live_frac, clip_on=False,
            linewidth=2, facecolor='#006230'
        )
    )
    axs[i].add_patch(
        plt.Rectangle(
            (1.3e5, -0.1 + 1.2 * live_frac), 0.4e5, 1.2 * (1 - live_frac),
            clip_on=False, linewidth=2, facecolor='#DC0F0F'
        )
    )

    # Print live chain fraction
    print(f'Conversion: {conv:.0%}, '
          f'Live chains: {live_frac:.1%}')

    if i == 0:
        axs[i].set_ylabel('Intensity (a.u.)')

axs[len(conversions) // 2].set_xlabel('Molecular Weight (g/mol)')
fig.savefig('../example_figures/calculate_mwd.svg')
plt.show()
