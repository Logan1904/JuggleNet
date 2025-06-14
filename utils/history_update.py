import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose

def update_history(history_dict, ball, landmarks, max_len=25):

    if ball:
        ball_x, ball_y, ball_radius = ball
    else:
        ball_y = None

    history_dict["Ball"].append(ball_y)

    history_dict["Left Foot"].append(get_landmark_y(landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX))
    history_dict["Right Foot"].append(get_landmark_y(landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX))
    history_dict["Left Knee"].append(get_landmark_y(landmarks, mp_pose.PoseLandmark.LEFT_KNEE))
    history_dict["Right Knee"].append(get_landmark_y(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE))
    history_dict["Left Hip"].append(get_landmark_y(landmarks, mp_pose.PoseLandmark.LEFT_HIP))
    history_dict["Right Hip"].append(get_landmark_y(landmarks, mp_pose.PoseLandmark.RIGHT_HIP))

    predict_histories(history_dict)

    trim_histories(history_dict, max_len)

    return history_dict

def predict_histories(history_dict, n_points=5):
    for key, y_values in history_dict.items():
        valid_y = [y for y in y_values if y]
        if len(valid_y) < 3:
            # not enough points for prediction
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
            prediction = None  # fail-safe

        if not y_values[-1]:
            history_dict[key][-1] = prediction

        



def trim_histories(history_dict, max_len):
    for key in history_dict:
        history_dict[key] = history_dict[key][-max_len:]

def get_landmark_y(landmarks, landmark_id):
    if landmarks:
        lm = landmarks.landmark[landmark_id]
        if lm.visibility > 0.5:
            return int(lm.y * 640)
            
        
    return None