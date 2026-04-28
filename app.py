import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# 1. Set up the hand tracker
model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
if not os.path.exists(model_path):
    print("ERROR: Can't find hand_landmarker.task in the project folder.")
    exit()

hand_tracker = hand_landmarker.HandLandmarker.create_from_options(
    hand_landmarker.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
)

# 2. Load the images you want to switch between
image_dir = os.path.join(os.path.dirname(__file__), 'Monkey')
open_palm_image = cv2.imread(os.path.join(image_dir, 'monkeyknow.png'))
closed_fist_image = cv2.imread(os.path.join(image_dir, 'monkeythink.png'))

# Check if the images loaded correctly
if open_palm_image is None or closed_fist_image is None:
    print("ERROR: I can't find one of the Monkey images. Check the file names in the Monkey folder.")
    exit()

# Make both images the same size so they swap cleanly
display_size = (open_palm_image.shape[1], open_palm_image.shape[0])
open_palm_image = cv2.resize(open_palm_image, display_size)
closed_fist_image = cv2.resize(closed_fist_image, display_size)

# Create a solid black screen of the same size
blank_image = np.zeros_like(open_palm_image)

# 3. Turn on the webcam
cap = cv2.VideoCapture(0)
print("Camera is on! Put your index finger near your mouth for THINK, or point your index finger up for KNOW. Press 'q' to quit.")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

timestamp_ms = 0

while True:
    success, frame = cap.read()
    if not success:
        continue

    # Mirror the camera so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    # MediaPipe needs colors in a specific format (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Look for hands in the camera frame
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    results = hand_tracker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    show_know = False
    show_think = False

    # Detect face so we can estimate mouth position.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    mouth_center = None
    mouth_radius = None
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center = (int(x + 0.5 * w), int(y + 0.5 * h))
        face_radius = int(0.55 * max(w, h))
        mouth_center = (int(x + 0.5 * w), int(y + 0.72 * h))
        mouth_radius = int(0.15 * (w + h))
        cv2.circle(frame, face_center, face_radius, (0, 255, 255), 2)

    # 4. Check the fingers if a hand is on screen
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw landmarks and finger/hand lines on the webcam frame.
            points = []
            frame_height, frame_width, _ = frame.shape
            for landmark in hand_landmarks:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                points.append((x, y))
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            for start_index, end_index in HAND_CONNECTIONS:
                cv2.line(frame, points[start_index], points[end_index], (255, 0, 0), 2)

            # Index fingertip near mouth -> THINK image.
            index_tip = points[8]
            if mouth_center is not None and mouth_radius is not None:
                distance_to_mouth = np.hypot(index_tip[0] - mouth_center[0], index_tip[1] - mouth_center[1])
                if distance_to_mouth <= mouth_radius:
                    show_think = True

            # Index up (other fingers down) -> KNOW image.
            index_up = hand_landmarks[8].y < hand_landmarks[6].y < hand_landmarks[5].y
            middle_down = hand_landmarks[12].y > hand_landmarks[10].y
            ring_down = hand_landmarks[16].y > hand_landmarks[14].y
            pinky_down = hand_landmarks[20].y > hand_landmarks[18].y

            if index_up and middle_down and ring_down and pinky_down:
                show_know = True

    # 5. Show a different image depending on the gesture
    if show_think:
        cv2.imshow('Magic Window', closed_fist_image)
    elif show_know:
        cv2.imshow('Magic Window', open_palm_image)
    else:
        cv2.imshow('Magic Window', blank_image)

    # Show what the webcam sees
    cv2.imshow('Webcam View', frame)

    # 6. How to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Turn off the camera and close windows when done
hand_tracker.close()
cap.release()
cv2.destroyAllWindows()