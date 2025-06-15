import numpy as np
from utils.Kalman1D import Kalman1D

MAX_LEN = 100

kalman_filter = {}

def update_measurements(measurements, ball):

    if ball:
        ball_x, ball_y, ball_radius = ball
    else:
        ball_y = np.nan

    measurements["Ball"].append(ball_y)

    trim_histories(measurements, MAX_LEN)

    return measurements

def predict_KF(measurements, predictions):
    
    for key, y_values in measurements.items():
        if key not in kalman_filter:
            kalman_filter[key] = Kalman1D()

        kf = kalman_filter[key]

        # Use the latest valid measurement
        valid_y = [y for y in y_values if not np.isnan(y)]
        if not valid_y:
            predictions[key].append(np.nan)
            continue

        measurement = valid_y[-1]

        # Kalman update/predict
        if not kf.initialized:
            kf.x[0, 0] = measurement  # initialize position
            kf.x[1, 0] = 0            # assume starting velocity = 0
            kf.initialized = True

        kf.update(measurement)
        predicted_y = kf.predict()

        predictions[key].append(float(predicted_y))

    trim_histories(predictions, MAX_LEN)

    return predictions
        
def predict_para(measurements, predictions, n_points=5):

    for key, y_values in measurements.items():
        
        # if measurement valid, accept it
        if not np.isnan(y_values[-1]):
            predictions[key].append(y_values[-1])
            continue

        # Use the latest valid measurement
        valid_y = [y for y in y_values if not np.isnan(y)]
        if not valid_y or len(valid_y) < 3:
            predictions[key].append(np.nan)
            continue

        # Take last n_points (at most)
        y_fit = np.array(valid_y[-n_points:])
        x_fit = np.arange(len(y_fit))

        try:
            # Fit a quadratic polynomial
            coeffs = np.polyfit(x_fit, y_fit, deg=2)
            poly = np.poly1d(coeffs)

            # Predict next y (at time x = len(y_fit))
            next_y = poly(len(y_fit))
            prediction = float(next_y)
        except:
            prediction = np.isnan  # fail-safe

        predictions[key].append(prediction)

    trim_histories(predictions, MAX_LEN)

    return predictions



def trim_histories(measurements, max_len):
    for key in measurements:
        measurements[key] = measurements[key][-max_len:]

def get_landmark_y(landmarks, landmark_id):
    if landmarks:
        lm = landmarks.landmark[landmark_id]
        if lm.visibility > 0.5:
            return int(lm.y * 640)
            
        
    return None