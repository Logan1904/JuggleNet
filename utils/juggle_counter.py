from scipy.signal import find_peaks
import numpy as np


def update_juggle_count(predictions, current_count):
    # juggle using simple minimum point detection

    y = np.array(predictions['Ball'][:,1])

    
    if len(y) > 10:
        # find a minimum point in position
        min_peak, _ = find_peaks(y,prominence=0.02)    # max peaks of y will be min peaks of height trajectory

        if min_peak.any():

            # find minimum point position
            min_index = min_peak[0]
            ball_pos = np.array(predictions['Ball'][min_index,:2])

            min_dist = np.inf
            point = ""
            for key,value in predictions.items():
                if key == "Ball":
                    continue
                
                pos = value[min_index,:2]
                dist = np.linalg.norm(ball_pos-pos)

                if dist < min_dist:
                    min_dist = dist
                    point = key
            
            current_count[point] += 1

            # Reset history to avoid double-counting
            for key in predictions:
                predictions[key] = np.empty(shape=(0,4))

    return current_count
