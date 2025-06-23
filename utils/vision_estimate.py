import mediapipe as mp
import cv2
from ultralytics import YOLO
import numpy as np

MP_POSE = mp.solutions.pose
POSE = MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

POI_MP_MAPPING = {
    "Left_Knee": MP_POSE.PoseLandmark.LEFT_KNEE,
    "Right_Knee": MP_POSE.PoseLandmark.RIGHT_KNEE,
    "Left_Foot": MP_POSE.PoseLandmark.LEFT_FOOT_INDEX,
    "Right_Foot": MP_POSE.PoseLandmark.RIGHT_FOOT_INDEX,
    "Head": MP_POSE.PoseLandmark.NOSE
}

# custom fine-tuned model
MODEL = YOLO("./models/best.pt")
MODEL.to('cuda')

def get_POI(frame):
    # POI in format {'Object': np.array([pos_x_norm, pos_y_norm, width_norm, height_norm])}
    # No readings will result in np.array([None, None, 0, 0])
    # width and height of body pose landmarks always 0, 0
    
    landmarks_mp = detect_landmarks_mp(frame)
    POI_landmarks = parse_landmarks_mp(landmarks_mp, POI_MP_MAPPING)

    ball = detect_football_yolo(frame)
    POI_ball = {'Ball': ball}

    POI = POI_landmarks | POI_ball

    return POI

def detect_landmarks_mp(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = POSE.process(image_rgb)

    landmarks = results.pose_landmarks

    return landmarks

def parse_landmarks_mp(landmarks_mp, POI_MP_MAPPING):
    POI = {}

    for key,val in POI_MP_MAPPING.items():

        if landmarks_mp:
            output = landmarks_mp.landmark[val]
            if output.visibility > 0.5:
                POI[key] = np.array([output.x, output.y, 0, 0])
            else:
                POI[key] = np.array([None, None, 0, 0])
        else:
            POI[key] = np.array([None, None, 0, 0])

    return POI    

    
def detect_football_yolo(frame):
    results = MODEL.predict(source=frame, conf=0.3, verbose=False, device=0)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:     # football
                x1, y1, x2, y2 = map(float, box.xyxyn[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                return np.array([cx, cy, x2-x1, y2-y1])
            
    return np.array([None, None, 0, 0])