import cv2
import argparse
import os

from utils.pose import get_pose_landmarks, get_pose_landmarks_yolo
from utils.juggle_counter import update_juggle_count
from utils.draw import draw_info
from utils.ball_tracker import detect_football_yolo
from utils.graph_plot import init_plot, update_plot
from utils.history_update import update_history

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Football Juggle Counter")
parser.add_argument('--video', type=str, default=None, help='Path to video file. Leave empty to use webcam.')
args = parser.parse_args()

# --- Choose Video Source ---
if args.video:
    if not os.path.exists(args.video):
        print(f"Error: File '{args.video}' not found.")
        exit(1)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    print(f"Running on pre-recorded video: {args.video}")
else:
    cap = cv2.VideoCapture(0)
    print("Running on live webcam.")

# --- App Loop ---
juggle_count = 0
fig, ax = init_plot()

history_dict = {
    "Ball": [],
    "Left Foot": [],
    "Right Foot": [],
    "Left Knee": [],
    "Right Knee": [],
    "Left Hip": [],
    "Right Hip": [],
}

juggle=False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video stream ended or camera disconnected.")
        break

    # resize frame
    frame = cv2.resize(frame, (640, 360))
    # rotate frame (for pre-recorded iphone)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # detect pose landmarks
    landmarks = get_pose_landmarks(frame)

    # detect football via YOLO
    ball = detect_football_yolo(frame)

    # update graphs
    history_dict = update_history(history_dict, ball, landmarks)
    update_plot(ax, history_dict)

    juggle_count, juggle = update_juggle_count(history_dict, juggle_count, juggle)
    print(juggle)

    draw_info(frame, landmarks, ball, juggle_count)

    cv2.imshow("Football Juggle Counter", frame)
    key = cv2.waitKey(1) & 0xFF
    

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
