import base64
import logging
import os
from io import BytesIO

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class InferenceEngine:
    """Handles MediaPipe hand tracking and gesture inference."""

    _instance = None

    def __new__(cls):
        """Singleton pattern to avoid reloading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # Find model file in project root or api directory
        api_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(api_dir)
        
        model_paths = [
            os.path.join(project_root, 'hand_landmarker.task'),
            os.path.join(api_dir, 'hand_landmarker.task'),
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f'hand_landmarker.task not found. Searched: {model_paths}'
            )

        logger.info(f'Loading model from {model_path}')
        self.hand_tracker = hand_landmarker.HandLandmarker.create_from_options(
            hand_landmarker.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        )
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._initialized = True
        logger.info('InferenceEngine initialized')

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        """Analyze a single video frame for hand gestures and face detection."""
        frame = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        results = self.hand_tracker.detect(mp_image)

        show_know = False
        show_think = False
        hands_payload = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        mouth_center = None
        mouth_radius = None
        if len(faces) > 0:
            x, y, w, h = faces[0]
            mouth_center = (int(x + 0.5 * w), int(y + 0.72 * h))
            mouth_radius = int(0.15 * (w + h))

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                points = []
                frame_height, frame_width, _ = frame.shape
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    points.append((x, y))

                hands_payload.append(
                    [{'x': float(lm.x), 'y': float(lm.y)} for lm in hand_landmarks]
                )

                if mouth_center is not None and mouth_radius is not None:
                    index_tip = points[8]
                    distance_to_mouth = np.hypot(
                        index_tip[0] - mouth_center[0],
                        index_tip[1] - mouth_center[1],
                    )
                    if distance_to_mouth <= mouth_radius:
                        show_think = True

                index_up = (
                    hand_landmarks[8].y
                    < hand_landmarks[6].y
                    < hand_landmarks[5].y
                )
                middle_down = hand_landmarks[12].y > hand_landmarks[10].y
                ring_down = hand_landmarks[16].y > hand_landmarks[14].y
                pinky_down = hand_landmarks[20].y > hand_landmarks[18].y

                if index_up and middle_down and ring_down and pinky_down:
                    show_know = True

        if show_think:
            state = 'THINK'
        elif show_know:
            state = 'KNOW'
        else:
            state = 'IDLE'

        return {
            'state': state,
            'handsDetected': len(results.hand_landmarks) if results.hand_landmarks else 0,
            'faceDetected': len(faces) > 0,
            'handLandmarks': hands_payload,
            'handConnections': HAND_CONNECTIONS,
        }

    def analyze_base64_frame(self, base64_str: str) -> dict:
        """Analyze a base64-encoded image frame."""
        try:
            # Remove data:image/... prefix if present
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_str)
            image_pil = PILImage.open(BytesIO(image_bytes))

            # Convert PIL to OpenCV format
            frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            return self.analyze_frame(frame_bgr)
        except Exception as e:
            logger.error(f'Error analyzing base64 frame: {e}')
            raise
