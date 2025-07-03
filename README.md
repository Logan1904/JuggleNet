# 🚣 StrokeCountNet

> **AI-powered rowing stroke counter**

StrokeCountNet is a Computer Vision project that tracks body landmarks count strokes and aid video analysis. 

<p align="center">
  <img src="./assets/Vid0_Analysed.gif" width="300"/>
</p>

---

## 🚀 Features

- ✅ **Ball Detection** with **Fine-Tuned** YOLOv8
- 🦿 **Body Pose Estimation** with MediaPipe (hips, wrist, knees, feet)
- 🧮 **Smoothed Estimation** via Kalman Filter
- 📈 **Live Graphing** of landmark X-positions
- 🧠 **Stroke Count Logic** using physics-inspired heuristic
- 🔁 Works on **webcam or video input**
- 🎨 Visual overlay showing keypoints, and live counts

---

## 🗂️ Project Structure
```
.
├── utils/
│   ├── draw_POI.py          # Draw POI landmarks
│   ├── plot_graph.py        # Plot landmark time series & positions
│   ├── update_predict.py    # Update measurements, perform predictions
│   ├── stroke_counter.py    # Count strokes
│   ├── Kalman1D.py          # 1D Kalman Filter
│   ├── vision_estimate.py   # Extract POIs via vision pipeline
│   └── download_youtube_video.sh  # Helper script for YouTube downloads
├── models/
│   └── finetuned.pt         # Fine-tuned YOLOv8 model
├── source_data/
│   └── Vid0.mp4             # Example input video
├── save/
│   └── Vid0_Analysed.mp4    # Analysed example video
├── main.py                  # Main driver script
├── pyproject.toml           # Python project configuration & dependencies
└── README.md                # Project documentation
```

## 🧰 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/shug3502/strokecountnet.git
cd strokecountnet
```

### 2. Install Dependencies
```bash
uv venv .venv
uv pip install -e .
```

### 3. Run Script
```bash
python main.py --video source_data/Vid0.mp4 --save save/ --plot  # Run on video in source_data folder
```

#### Optional Arguments
```bash
--video <your_video>          # To run on pre-recorded video
--save <your_save_directory>  # To save analysed video
-- plot                       # To plot ball Y-position
```

## 👷 Based on a football juggle counter

This was built on a similar project for counting the number of juggles of a football, [JuggleNet](https://github.com/Logan1904/JuggleNet)

⚙️ Model Details

Uses the model fine-tuned in the [JuggleNet](https://github.com/Logan1904/JuggleNet) project. 

 - Base model: YOLOv8n (Ultralytics)
 - Task: Object detection 

🧪 Training Configuration

Previously fine tuning was done, and to improve detection for specifically for rowing pose estimation, you could run something like:

```bash
yolo task=detect \
     mode=train \
     model=yolov8n.pt \
     data=<finetune_data_directory> \
     epochs=50 \
     imgsz=640 \
     batch=8 \
```

