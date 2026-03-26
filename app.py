import cv2
import mediapipe as mp
import numpy as np
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

import xgboost
print(xgboost.__version__)
# Download model if not present
if not os.path.exists("hand_landmarker.task"):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )
    print("Downloaded!")

# Load XGBoost model and label encoder
from xgboost import XGBClassifier

# Load model
clf = XGBClassifier()
clf.load_model("asl_model_xgb.json")
le = joblib.load("label_encoder.pkl")
print("Models loaded!")

# Normalization function
def normalize_landmarks(row):
    landmarks = np.array(row).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    hand_size = np.linalg.norm(landmarks[12])
    if hand_size > 0:
        landmarks = landmarks / hand_size
    return landmarks.flatten().tolist()

# Setup MediaPipe
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)
print(" Webcam started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Extract and normalize
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        row = normalize_landmarks(row)

        # Predict with XGBoost
        prediction_enc = clf.predict([row])[0]
        prediction = le.inverse_transform([prediction_enc])[0]
        confidence = clf.predict_proba([row]).max() * 100

        # Draw landmarks
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Display prediction
        cv2.putText(frame, f"{prediction} ({confidence:.1f}%)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)
        cv2.putText(frame, "ASL Sign Detection",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("ASL Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Closed!")