import mediapipe as mp
import cv2
from ultralytics import YOLO

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_pose_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    landmarks = results.pose_landmarks

    return landmarks


def get_pose_landmarks_yolo(frame):
    model = YOLO('yolo11n-pose.pt')
    results = model.predict(source=frame, conf=0.3, verbose=False)

    return results