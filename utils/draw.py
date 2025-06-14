import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

selected_landmarks = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

def draw_info(frame, landmarks, ball, juggle_count):

    if landmarks:
        draw_selected_landmarks(frame, landmarks, selected_landmarks)
    
    if ball:
        ball_x, ball_y, ball_radius = ball
        cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 255, 255), 2)

    # Juggle counter
    cv2.putText(frame, f"Juggles: {juggle_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)


def draw_selected_landmarks(frame, landmarks, landmark_ids, color=(0, 255, 0), radius=6):
    h, w, _ = frame.shape
    for idx in landmark_ids:
        lm = landmarks.landmark[idx]
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), radius, color, -1)