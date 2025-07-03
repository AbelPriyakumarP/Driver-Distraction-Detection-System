# 🚘 Driver Drowsiness & Distraction Detection System

A real-time driver safety monitoring system using **YOLOv8**, **MediaPipe FaceMesh**, and **OpenCV** to detect drowsiness, phone usage, and distracted driving behavior — triggering audio, visual, and logged alerts.

---

## 📽️ Features

✅ Detects driver presence using YOLOv8  
✅ Detects phone usage while driving  
✅ Monitors facial landmarks to calculate Eye Aspect Ratio (EAR) for drowsiness detection  
✅ Displays real-time status bar (Attentive / Drowsy / Distracted / Using Phone)  
✅ Triggers audio and visual alerts on unsafe driving behavior  
✅ Logs all incidents to `alert_log.txt` with timestamps  
✅ Captures snapshots during alerts  
✅ Fatigue monitoring: warns driver after multiple alerts

---

## 📷 Demo

![Image alt](https://github.com/AbelPriyakumarP/Driver-Distraction-Detection-System/blob/6ca10c0faf6ea0a5ecff55c895eca96eea1367f2/driver_distraction_detection_system/alert_1751531346.jpg)

---

## 📦 Dependencies

- Python 3.8+
- OpenCV
- NumPy
- MediaPipe
- Ultralytics YOLOv8
- winsound (Windows only)

Install dependencies:

```bash
pip install opencv-python mediapipe ultralytics numpy
```

## 📂 How to Run
Clone this repository:

```bash
Copy
Edit
git clone https://github.com/your-username/driver-safety-monitor.git
cd driver-safety-monitor
Download YOLOv8 model weights (if not done):
```

```bash
Copy
Edit
yolo task=detect mode=predict model=yolov8n.pt
Run the system:
```

```bash
Copy
Edit
python driver_monitor.py
Press q to quit the window.
```

## 📑 How it Works
Module	Role
YOLOv8	Detects 'person' (driver) and 'phone' objects
MediaPipe FaceMesh	Detects 468 facial landmarks, calculates EAR for drowsiness detection
OpenCV	Frame capture, drawing overlays, status bar, and saving images
Audio & Logging	Beep alert using winsound and logs incidents to text file

## 📒 Logs & Outputs
All alert logs stored in alert_log.txt

Alert snapshots saved as alert_<timestamp>.jpg

## ⚠️ Notes
This project currently uses winsound for beeping alerts — available only on Windows.

Tested on Python 3.10, YOLOv8n, MediaPipe 0.10+

Adjust EAR threshold and frame counters for your custom video or environment.

## 📃 License
MIT License — free for personal and academic use.

## ✨ Acknowledgments
Ultralytics YOLOv8

MediaPipe by Google

OpenCV community

## 📬 Contact
For suggestions or improvements, open an issue or reach me at: [roshabel001@gmail.com]

yaml
Copy
Edit

---

✅ Would you like me to turn this into an actual **GitHub repository template structure** for you as well? I can generate a file/folder layout and `.gitignore` if you like 🚀


