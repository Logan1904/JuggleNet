from scipy.signal import find_peaks
import numpy as np


def update_stroke_count(predictions, current_count):
    # stroke count using simple minimum point detection

    x = np.array(predictions['Left_Wrist'][:,0])
    
    if len(x) > 10:
        # find a minimum point in position
        min_peak, _ = find_peaks(x,prominence=0.02 )    # max peaks of x will be min peaks of height trajectory

        if min_peak.any():
            current_count += 1

            # Reset history to avoid double-counting
            for key in predictions:
                predictions[key] = np.empty(shape=(0,4))

    return current_count
