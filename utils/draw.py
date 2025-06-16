import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose

selected_landmarks = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

def draw_info(frame, POI, juggle_count):

    for point,val in POI.items():

        if val[0]:
            h, w, _ = frame.shape

            cx, cy = val[0]*w, val[1]*h
            bw, bh = val[2]*w, val[3]*h

            x1 = cx - bw / 2     # top-left x
            y1 = cy - bh / 2     # top-left y
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            if point == 'Ball':
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            else:
                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)

    
    # Juggle counter
    cv2.putText(frame, f"Juggles: {juggle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)