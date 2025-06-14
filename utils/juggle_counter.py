from scipy.signal import find_peaks
import numpy as np


def update_juggle_count(history_dict, current_count, juggle):

    # find upwards movement
    y_vals = -np.array(history_dict['Ball']) # negative because y is 0 at top of image
    
    if len(y_vals) > 10:

        max_peak, _ = find_peaks(y_vals)
        min_peak, _ = find_peaks(-y_vals)

        if min_peak.any():
            # ball moving upwards
            juggle = True
        elif max_peak.any() and juggle:
            # ball moving downards
            juggle = False
            current_count += 1
            for key in history_dict:
                history_dict[key] = []

    return current_count, juggle
