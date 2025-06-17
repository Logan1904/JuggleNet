from scipy.signal import find_peaks
import numpy as np


def update_juggle_count(predictions, current_count):
    # juggle using simple minimum point detection

    y = np.array(predictions['Ball'][:,1])
    
    if len(y) > 10:
        # find a minimum point in position
        min_peak, _ = find_peaks(y)    # max peaks of y will be min peaks of height trajectory

        if min_peak.any():
            current_count += 1

            # Reset history to avoid double-counting
            for key in predictions:
                predictions[key] = np.empty(shape=(0,4))

    return current_count
