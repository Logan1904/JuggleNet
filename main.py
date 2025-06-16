import cv2
import argparse
import os
import numpy as np

from utils.pose import get_POI
from utils.juggle_counter import update_juggle_count
from utils.draw import draw_info
from utils.graph_plot import init_plot, update_plot
from utils.history_update import update_measurements, predict_KF, predict_para

POI = ["Ball", "Head", "Left_Knee", "Right_Knee", "Right_Foot", "Left_Foot"]

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
fig, ax = init_plot()

measurements, predictions = {},{}
for point in POI:
    measurements[point] = np.empty(shape=(0,4))
    predictions[point] = np.empty(shape=(0,4))

juggle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video stream ended or camera disconnected.")
        break

    # detect POI
    POIs = get_POI(frame)

    # update arrays
    measurements = update_measurements(measurements, POIs)
    
    predictions = predict_KF(measurements, predictions)
    #predictions_para = predict_para(measurements, predictions)
    
    update_plot(ax, predictions)

    juggle_count = update_juggle_count(predictions, juggle_count)

    draw_info(frame, POIs, juggle_count)

    cv2.imshow("Football Juggle Counter", frame)
    key = cv2.waitKey(1) & 0xFF
    

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
