import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

selected_landmarks = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
]

class Visualiser:
    """
    Handles all visualisation: drawing detections, pose landmarks, and juggle counts on frames.
    """

    def __init__(self, 
                 ball_colour: tuple = (0, 255, 0),
                 landmark_colour: tuple = (0, 0, 255),
                 text_colour: tuple = (0, 0, 0),
                 text_thickness: int = 3,
                 bbox_thickness: int = 2,
                 landmark_radius: int = 6):
        """
        Args:
            ball_colour: Colour tuple for ball bounding circle
            landmark_colour: Colour tuple for body part landmarks
            text_colour: Colour tuple for text overlay
            text_thickness: Thickness of text rendering
            bbox_thickness: Thickness of bounding box lines
            landmark_radius: Radius of landmark circles
        """

        self.ball_colour = ball_colour
        self.landmark_colour = landmark_colour
        self.text_colour = text_colour
        self.text_thickness = text_thickness
        self.bbox_thickness = bbox_thickness
        self.landmark_radius = landmark_radius

    def draw(self, frame: np.ndarray, POI: dict, counts: dict) -> np.ndarray:
        """
        Draw all visualisations on the frame

        Args:
            frame: Input frame
            POI: Points of interest dict {name: [x, y, w, h]}
            counts: Juggle counts dict {body_part: count}

        Returns:
            Annotated frame
        """

        self.draw_detections(frame, POI)
        self.draw_counts(frame, counts)
        return frame
    
    def draw_detections(self, frame: np.ndarray, POI: dict):
        """Draw detected ball and body landmarks on frame"""

        h, w, _ = frame.shape

        for point, val in POI.items():
            # Skip if no detection
            if not val[0]:
                continue
            
            # Convert normalised coordinates to pixel coordinates
            cx, cy = val[0]*w, val[1]*h
            bw, bh = val[2]*w, val[3]*h

            if point == 'Ball':
                # Draw bounding box for ball
                x1 = cx - bw / 2        # top left x
                y1 = cy - bh / 2        # top left y
                x2 = cx + bw / 2        # bottom right x
                y2 = cy + bh / 2        # bottom right y

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.ball_colour, self.bbox_thickness)

                # Optional: Draw centre point
                cv2.circle(frame, (int(cx), int(cy)), self.landmark_radius, self.landmark_colour, -1)
            else:
                cv2.circle(frame, (int(cx), int(cy)), self.landmark_radius, self.landmark_colour, -1)

    def draw_counts(self, frame: np.ndarray, counts: dict):
        """Draw juggle count statistics on frame"""
    
        line_num = 0

        for key, val in counts.items():
            if key == "Ball":
                continue
        
            # Position text vertically with spacing
            y_position = 30 + line_num * 30

            cv2.putText(
                frame,
                f"{key}: {val}",
                (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.text_colour,
                self.text_thickness
            )

            line_num += 1
    
    def draw_fps(self, frame: np.ndarray, fps: float):
        """
        Draw FPS counter on frame

        Args:
            frame: Input frame
            fps: Current frames per second
        """

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame.shape[1] - 150, 30), # Top right
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),  # Cyan
            2
        )

    def draw_total_count(self, frame: np.ndarray, counts:dict):
        """Draw total juggle count across all body parts"""

        total = sum(val for key, val in counts.items() if key != "Ball")

        cv2.putText(
            frame,
            f"Total: {total}",
            (10, frame.shape[0] - 20),   # Bottom left
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),  # Yellow
            3
        )
