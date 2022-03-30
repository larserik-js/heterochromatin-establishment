import numpy as np


def rms_vals_within_bounds(array_or_list):
    # Converts input to ndarray
    rms_vals = np.asarray(array_or_list)

    # Bounds for rms
    RMS_LOWER_BOUND, RMS_UPPER_BOUND = 1.677, 4.130

    # Truth values
    truth_values = (RMS_LOWER_BOUND <= rms_vals) & (rms_vals <= RMS_UPPER_BOUND)

    return np.all(truth_values)

