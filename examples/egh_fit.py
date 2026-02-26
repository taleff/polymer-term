  from polychemtools.analysis.gpc_trace import GPCTrace
  from polyterm.core.broadening import egh_broadening
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.optimize import least_squares

  gpcs = GPCTrace.from_file(
      instrument='tosoh',
      file_path='../gpc/MTT_2_117_RI.txt',
      calibration='../gpc/calibrations.json:ri_2025_11_21',
      bounds=(1e4, 2e6),
      correction='span'
  )

  plt.style.use('guironnet_figure_default')
  fig, axs = plt.subplots(1, 6, figsize=(30, 5), layout='constrained')

  dead_fracs = np.zeros(6)

  for i, gpc in enumerate(gpcs[:6]):
      # Extract right edge (from peak maximum onwards)
      peak_idx = np.argmax(gpc.intensities[::-1])
      edge_mws = gpc.molecular_weights[::-1][peak_idx:]
      edge_ints = gpc.intensities[::-1][peak_idx:]
      peak_intensity = np.max(gpc.intensities[::-1])

      def residual(params):
          center, coeff = params
          predicted = coeff * egh_broadening(edge_mws, center, 0.128, 0.0456)
          return 1e9 * (predicted - edge_ints)**2

      # Initial guess: center at first point, small broadening
      initial_guess = (edge_mws[0], 0.01)
      bounds = (0, np.inf)

      result = least_squares(residual, x0=initial_guess, bounds=bounds)

      # Convert center MW to DP
      center, norm = result['x']

      mws = gpc.molecular_weights[::-1]
      mole_frac_int = gpc.get_mole_fractions()[::-1]
      mole_frac_int = mole_frac_int / np.max(mole_frac_int)
      
      mass_frac_egh = egh_broadening(gpc.molecular_weights[::-1], center, 0.128, 0.0456)
      mole_frac_egh = mass_frac_egh / gpc.molecular_weights[::-1]
      mole_frac_egh = mole_frac_egh / np.max(mole_frac_egh)

      mole_frac_dead = mole_frac_int - mole_frac_egh

      alive_frac = np.sum(mole_frac_egh) / np.sum(mole_frac_int)
      dead_frac = np.sum(mole_frac_dead) / np.sum(mole_frac_int)

      axs[i].plot(mws, mole_frac_int, 'k-')
      axs[i].plot(mws, mole_frac_egh, 'g--')
      axs[i].plot(mws, mole_frac_dead, 'r--')
      axs[i].set_xscale('log')
      axs[i].set_xlim((1e4, 2e6))
      axs[i].set_ylim((-0.1, 1.1))

      axs[i].annotate(f'Dead: {dead_frac*100:.1f}%', (1.1e4, 1.0))

      print(f'Alive: {alive_frac*100:.1f}%')
      print(f'Dead: {dead_frac*100:.1f}%\n')

      dead_fracs[i] = dead_frac
      
  fig.supxlabel('Molecular Weight (g/mol)', fontsize=28)
  fig.supylabel('Intensity (A.U.)', fontsize=28)
  fig.savefig('../fig/MTT_2_117_egh.svg')

  times = np.array([30, 60, 120, 240, 480, 960]) # s

  fig, ax = plt.subplots(figsize=(5.5, 5), layout='constrained')
  ax.plot(times, -np.log(1-dead_fracs), 'k.', markersize=15)
  ax.set_xlabel('Time (s)')
  ax.set_ylabel(r'$-\frac{[Ru]_{live}}{[Ru]}$')
  fig.savefig('../fig/MTT_2_117_kt.svg')
