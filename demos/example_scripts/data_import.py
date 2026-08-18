"""
Utility functions for loading GPC data from CSV files.

Handles parsing of exported Tosoh GPC data, applying calibration
curves to convert elution time to molecular weight, baseline
correction, and molecular weight bounds filtering.
"""

import json
import numpy as np
from pathlib import Path


# Calibration functions map elution time to molecular weight
# using log10(MW) = f(time)
CALIBRATION_TYPES = {
    'linear': lambda t, a, b: 10 ** (a * t + b),
    'cubic': lambda t, a, b, c, d: 10 ** (a*t**3 + b*t**2 + c*t + d),
}


def load_calibration(calibration_file, calibration_name):
    """
    Load a calibration curve from a JSON file.

    Parameters
    ----------
    calibration_file : str
        Path to the calibrations JSON file.
    calibration_name : str
        Name of the calibration entry to load.

    Returns
    -------
    callable
        Function that converts elution time to molecular weight.
    """
    with open(calibration_file) as f:
        calibrations = json.load(f)

    if calibration_name not in calibrations:
        raise KeyError(
            f"Calibration '{calibration_name}' not found. "
            f"Available: {list(calibrations.keys())}"
        )

    cal = calibrations[calibration_name]
    cal_func = CALIBRATION_TYPES[cal['type']]
    params = cal['params']

    return lambda t: cal_func(t, *params)


def _baseline_correct(times, intensities):
    """
    Apply span baseline correction by subtracting a linear fit
    from the first to last data point.
    """
    slope = (intensities[-1] - intensities[0]) / (times[-1] - times[0])
    return intensities - (slope * (times - times[0]) + intensities[0])


def load_gpc_trace(data_file, calibration_file, calibration_name,
                   bounds=None, trace_index=0):
    """
    Load a single GPC trace from a CSV data file.

    Reads the exported Tosoh GPC data, applies calibration to
    convert elution time to molecular weight, filters to the
    specified molecular weight bounds, and then applies baseline
    correction on the bounded data.

    Parameters
    ----------
    data_file : str
        Path to the CSV data file. First column is elution time,
        subsequent columns are intensity traces.
    calibration_file : str
        Path to the calibrations JSON file.
    calibration_name : str
        Name of the calibration entry to use.
    bounds : tuple of (float, float), optional
        (min_mw, max_mw) to filter the trace. If None, no
        filtering is applied.
    trace_index : int, optional
        Which intensity trace to load (0-indexed). Default 0.

    Returns
    -------
    molecular_weights : ndarray
        Molecular weights in ascending order.
    intensities : ndarray
        Detector response at each molecular weight.
    """
    # Load raw data
    data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
    times = data[:, 0]
    intensities = data[:, trace_index + 1]

    # Convert elution time to molecular weight
    cal_func = load_calibration(calibration_file, calibration_name)
    molecular_weights = cal_func(times)

    # GPC elutes high MW first, so reverse for ascending order
    molecular_weights = molecular_weights[::-1]
    intensities = intensities[::-1]

    # Apply bounds filter before baseline correction. This ensures
    # the baseline is fitted to the edges of the region of interest
    # rather than to distant parts of the chromatogram, matching
    # the convention used in polychemtools.
    if bounds is not None:
        low, high = bounds
        mask = (molecular_weights >= low) & (molecular_weights <= high)
        molecular_weights = molecular_weights[mask]
        intensities = intensities[mask]

    # Apply span baseline correction on the bounded data
    intensities = _baseline_correct(molecular_weights, intensities)

    return molecular_weights, intensities


def load_all_traces(data_file, calibration_file, calibration_name,
                    bounds=None):
    """
    Load all GPC traces from a multi-trace CSV data file.

    Parameters
    ----------
    data_file : str
        Path to the CSV data file.
    calibration_file : str
        Path to the calibrations JSON file.
    calibration_name : str
        Name of the calibration entry to use.
    bounds : tuple of (float, float), optional
        (min_mw, max_mw) to filter each trace.

    Returns
    -------
    list of (molecular_weights, intensities) tuples
        One tuple per trace in the file.
    """
    data = np.genfromtxt(data_file, delimiter=',', skip_header=1)
    n_traces = data.shape[1] - 1

    traces = []
    for i in range(n_traces):
        mws, ints = load_gpc_trace(
            data_file, calibration_file, calibration_name,
            bounds=bounds, trace_index=i
        )
        traces.append((mws, ints))

    return traces


def peak_molecular_weight(molecular_weights, intensities):
    """
    Find the molecular weight at the peak of the distribution.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights.
    intensities : ndarray
        Detector response at each molecular weight.

    Returns
    -------
    float
        Molecular weight at peak intensity.
    """
    return molecular_weights[np.argmax(intensities)]
