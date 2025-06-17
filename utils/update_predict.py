import numpy as np
from utils.Kalman1D import Kalman1D

MAX_LEN = 100
kalman_filter = {}

def update_measurements(measurements, POIs):
    # measurements in format {'Object': np.array([pos_x_norm, pos_y_norm, width_norm, height_norm])}
    # No readings will result in np.array([np.nan, np.nan, 0, 0])
    # width and height of body pose landmarks always 0, 0

    for point,val in POIs.items():
        if val[0] and val[1]:
            measurements[point] = np.vstack([measurements[point], val])
        else:
            measurements[point] = np.vstack([measurements[point], np.full((val.shape), np.nan)])

    trim_histories(measurements, MAX_LEN)

    return measurements

def predict_KF(measurements, predictions):
    # predictions in format {'Object': np.array([pos_x_norm, pos_y_norm, width_norm, height_norm])}
    # No readings will result in np.array([np.nan, np.nan, 0, 0])
    # width and height of body pose landmarks always 0, 0
    
    for key,val in measurements.items():
        x_vals = val[:,0]
        y_vals = val[:,1]

        if key not in kalman_filter:
            kalman_filter[key] = [Kalman1D(),Kalman1D()]

        kf_x = kalman_filter[key][0]
        kf_y = kalman_filter[key][1]

        # Use the latest measurement
        measurement_x = x_vals[-1]
        measurement_y = y_vals[-1]

        # If latest measurement is invalid, just predict
        if not np.isnan(measurement_x):
            if not kf_x.initialised:
                kf_x.x[0,0] = measurement_x     # initialise position
                kf_x.x[1,0] = 0                 # assume starting velocity is 0
                kf_x.initialised = True
            kf_x.update(measurement_x)
        if not np.isnan(measurement_y):
            if not kf_y.initialised:
                kf_y.x[0,0] = measurement_y     # initialise position
                kf_y.x[1,0] = 0                 # assume starting velocity is 0
                kf_y.initialised = True
            kf_y.update(measurement_y)
        
        predicted_x = kf_x.predict()
        predicted_y = kf_y.predict()

        predictions[key] = np.vstack([predictions[key], np.array([predicted_x, predicted_y, val[-1,2], val[-1,3]])])

    trim_histories(predictions, MAX_LEN)

    return predictions
        
def predict_para(measurements, predictions, min_points=5):
    # predictions in format {'Object': np.array([pos_x_norm, pos_y_norm, width_norm, height_norm])}
    # No readings will result in np.array([np.nan, np.nan, 0, 0])
    # width and height of body pose landmarks always 0, 0

    for key,val in measurements.items():
        x_vals = val[:,0]
        y_vals = val[:,1]

        # if measurement valid, accept it
        if not np.isnan(x_vals[-1]):
            predicted_x = x_vals[-1]
        else:
            
            # check if at least min_points exist
            valid_x = [x for x in x_vals if not np.isnan(x)]
            if not valid_x or len(valid_x) < min_points:
                predicted_x = np.nan
            else:
                x_fit = np.array(valid_x[-min_points:])
                xy_fit = np.arange(len(x_fit))

                try:
                    # Fit a quadratic polynomial
                    coeffs = np.polyfit(xy_fit, x_fit, deg=2)
                    poly = np.poly1d(coeffs)

                    predicted_x = poly(len(xy_fit))

                except:
                    predicted_x = np.nan
        
        # if measurement valid, accept it
        if not np.isnan(y_vals[-1]):
            predicted_y = y_vals[-1]
        else:
            # check if at least min_points exist
            valid_y = [y for y in y_vals if not np.isnan(y)]
            if not valid_y or len(valid_y) < min_points:
                predicted_y = np.nan
            else:
                y_fit = np.array(valid_y[-min_points:])
                yy_fit = np.arange(len(y_fit))

                try:
                    # Fit a quadratic polynomial
                    coeffs = np.polyfit(yy_fit, y_fit, deg=2)
                    poly = np.poly1d(coeffs)

                    predicted_y = poly(len(yy_fit))
                except:
                    predicted_y = np.nan

        predictions[key] = np.vstack([predictions[key], np.array([predicted_x, predicted_y, val[-1,2], val[-1,3]])])

    trim_histories(predictions, MAX_LEN)

    return predictions

def trim_histories(measurements, max_len):
    for key in measurements:
        measurements[key] = measurements[key][-max_len:,:]