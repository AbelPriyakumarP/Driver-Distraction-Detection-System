import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import winsound  # For audio alert on Windows

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # Use custom-trained YOLO model for 'person', 'mobile phone', 'steering wheel'

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture("./people-distracted-by-smartphone-white-man-on-road-trip-scared-driver-shocked-p-SBV-334989703-preview.mp4")  # Replace with video path if needed

# EAR calculation function (Eye Aspect Ratio)
def calculate_EAR(landmarks, eye_indices):
    points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmark indices for MediaPipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Thresholds
EAR_THRESHOLD = 0.23  # Adjust experimentally
drowsy_frames = 0
distracted_frames = 0
phone_usage_frames = 0
alert_triggered = False
fatigue_counter = 0

def play_alert_sound():
    duration = 500  # milliseconds
    freq = 1000  # Hz
    winsound.Beep(freq, duration)

def log_alert(message):
    with open("alert_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(frame_rgb)

    # YOLOv8 Detection
    results_yolo = model(frame)
    detections = results_yolo[0].boxes.data.cpu().numpy()

    phone_detected = False
    driver_detected = False

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        label = model.names[cls]
        
        if label == "person":
            driver_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, "Driver", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        elif label in ["cell phone", "mobile phone", "phone"]:
            phone_detected = True
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            cv2.putText(frame, "Phone", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # MediaPipe Face Mesh for Drowsiness Detection
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            left_ear = calculate_EAR(face_landmarks.landmark, LEFT_EYE)
            right_ear = calculate_EAR(face_landmarks.landmark, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            # Drowsiness detection
            if avg_ear < EAR_THRESHOLD:
                drowsy_frames += 1
                if drowsy_frames > 20:
                    cv2.putText(frame, "Drowsiness Detected!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                drowsy_frames = 0

    # Distraction Monitoring
    if phone_detected:
        phone_usage_frames += 1
        distracted_frames += 1
        if phone_usage_frames > 10:
            cv2.putText(frame, "Using Phone Detected!", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        if distracted_frames > 10:
            cv2.putText(frame, "Distracted Detected!", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
    else:
        phone_usage_frames = 0
        distracted_frames = 0

    # Status Bar
    status = "Attentive"
    color = (0,255,0)
    if drowsy_frames > 20:
        status = "Drowsy"
        color = (0,0,255)
    elif distracted_frames > 10:
        status = "Distracted"
        color = (255,0,0)
    elif phone_usage_frames > 10:
        status = "Using Phone"
        color = (0,0,255)
    cv2.rectangle(frame, (0,0), (frame.shape[1], 40), color, -1)
    cv2.putText(frame, f"Status: {status}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Combined Alert Logic
    if ((drowsy_frames > 20 or distracted_frames > 10 or phone_usage_frames > 10) and not alert_triggered):
        cv2.putText(frame, "ALERT: Unsafe Driving Detected!", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        print("[ALERT] Unsafe Driving Detected!")
        play_alert_sound()
        log_alert("Unsafe Driving Detected!")
        cv2.imwrite(f"alert_{int(time.time())}.jpg", frame)
        fatigue_counter += 1
        alert_triggered = True
    elif drowsy_frames == 0 and distracted_frames == 0 and phone_usage_frames == 0:
        alert_triggered = False

    # Fatigue Monitoring
    if fatigue_counter > 5:
        cv2.putText(frame, "FATIGUE WARNING: Take a Break!", (50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Driver Safety Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()