from ultralytics import YOLO

# Load YOLOv8 model (choose the right one)
model = YOLO("./models/best.pt")  # Or your custom .pt model if you trained one
model.to('cuda')

# Optionally restrict to relevant class names (if your model includes 'sports ball' or 'football')
BALL_CLASS_NAMES = ["football"]

def detect_football_yolo(frame):
    results = model.predict(source=frame, conf=0.3, verbose=False, device=0)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in BALL_CLASS_NAMES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max(x2 - x1, y2 - y1) // 2
                return cx, cy, radius
    return None
