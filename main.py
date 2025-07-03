import cv2
import argparse
import os
import numpy as np

from utils.vision_estimate import get_POI
from utils.draw_POI import draw_info
from utils.stroke_counter import update_stroke_count
from utils.plot_graph import init_landmark_plot, update_landmark_plot
from utils.update_predict import update_measurements, predict_KF

POI = ["Head", "Left_Knee", "Right_Knee", "Right_Hip", "Left_Hip", "Right_Heel", "Left_Heel", "Left_Wrist", "Right_Wrist", "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow"] # "Right_Ankle", "Left_Ankle",

def parse_args():
    parser = argparse.ArgumentParser(description="Rowing pose estimator")
    parser.add_argument('--video', type=str, default=None, help='Path to video file. Leave empty to use webcam.')
    parser.add_argument('--save', type=str, default=None, help='Path to save directory. Leave empty to not save.')
    parser.add_argument('--plot', action='store_true', help='Plot ball Y-trajectory.')

    return parser.parse_args()

def main():

    # parse arguments
    args = parse_args()

    # Video source
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: File '{args.video}' not found.")
            exit(1)
            
        cap = cv2.VideoCapture(args.video)
        print(f"Running on pre-recorded video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        print("Running on live webcam.")
    
    # FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0 or np.isnan(fps):
        fps = 30    # in case no fps is defined

    print(f"Video FPS: {fps}")

    # Video writer
    video_writer = None
    if args.save:
        if args.video:
            video_name, ext_name = os.path.splitext(os.path.basename(args.video))
            save_path = os.path.join(args.save, video_name + "_Analysed.mp4")
        else:
            save_path = os.path.join(args.save, "Analysed.mp4")

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # Initialise plot
    if args.plot:
        fig_lm, ax_lm = init_landmark_plot()

    # Initialise history variables
    measurements, predictions = {},{}
    for point in POI:
        measurements[point] = np.empty(shape=(0,4))
        predictions[point] = np.empty(shape=(0,4))

    # Initialise stroke counter
    stroke_count = 0

    # Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or camera disconnected.")
            break

        # detect POI
        POIs = get_POI(frame)

        # update measurement history
        measurements = update_measurements(measurements, POIs)
        
        # update and predict
        predictions = predict_KF(measurements, predictions)
        #predictions= predict_para(measurements, predictions)

        # update stroke count
        stroke_count = update_stroke_count(predictions, stroke_count)

        # update plot
        if args.plot:
            update_landmark_plot(ax_lm, measurements, predictions, POI)

        # draw on image
        draw_info(frame, POIs, stroke_count)

        # show image
        cv2.imshow("Rowing Stroke Counter", frame)
        
        # write video
        if video_writer:
            video_writer.write(frame) 
        
        # exit with 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # cleanup
    if video_writer:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()