import mediapipe as mp
import cv2
from ultralytics import YOLO
import numpy as np
import torch

MP_POSE = mp.solutions.pose
POSE = MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

POI_MP_MAPPING = {
    "Left_Knee": MP_POSE.PoseLandmark.LEFT_KNEE,
    "Left_Heel": MP_POSE.PoseLandmark.LEFT_HEEL,
    "Left_Wrist": MP_POSE.PoseLandmark.LEFT_WRIST,
    "Right_Wrist": MP_POSE.PoseLandmark.RIGHT_WRIST,
    "Left_Shoulder": MP_POSE.PoseLandmark.LEFT_SHOULDER,
    "Right_Shoulder": MP_POSE.PoseLandmark.RIGHT_SHOULDER,
    "Left_Elbow": MP_POSE.PoseLandmark.LEFT_ELBOW,
    "Right_Elbow": MP_POSE.PoseLandmark.RIGHT_ELBOW,
    # "Left_Ankle": MP_POSE.PoseLandmark.LEFT_ANKLE,
    # "Right_Ankle": MP_POSE.PoseLandmark.RIGHT_ANKLE,
    "Left_Hip": MP_POSE.PoseLandmark.LEFT_HIP,
    "Right_Hip": MP_POSE.PoseLandmark.RIGHT_HIP,
    "Right_Heel": MP_POSE.PoseLandmark.RIGHT_HEEL,
    "Right_Knee": MP_POSE.PoseLandmark.RIGHT_KNEE,
    "Head": MP_POSE.PoseLandmark.NOSE
}

# Automatically detect the best available device
def get_device():
    if torch.backends.mps.is_available():
        return 'mps'  # Mac Silicon (M1/M2/M3)
    elif torch.cuda.is_available():
        return 'cuda'  # NVIDIA GPU
    else:
        return 'cpu'  # CPU fallback

device = get_device()
print(f"Using device: {device}")

# custom fine-tuned model
MODEL = YOLO("./models/finetuned.pt")
MODEL.to(device)

def get_POI(frame):
    # POI in format {'Object': np.array([pos_x_norm, pos_y_norm, width_norm, height_norm])}
    # No readings will result in np.array([None, None, 0, 0])
    # width and height of body pose landmarks always 0, 0
    
    landmarks_mp = detect_landmarks_mp(frame)
    POI_landmarks = parse_landmarks_mp(landmarks_mp, POI_MP_MAPPING)

    return POI_landmarks

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

