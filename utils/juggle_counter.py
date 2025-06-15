from scipy.signal import find_peaks
import numpy as np


def update_juggle_count(predictions, current_count, juggle):

    y = -np.array(predictions['Ball']) # negative so aligned with gravity
    
    if len(y) > 10:

        # find a minimum point in position
        min_peak, _ = find_peaks(-y,prominence=2)

        if min_peak.any():
            current_count += 1
            print("Juggle detected at frame")

            # Reset history to avoid double-counting
            for key in predictions:
                predictions[key] = []


    return current_count, juggle
