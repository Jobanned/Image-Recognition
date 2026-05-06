import argparse
import base64
import json
import logging
import os
import runpy
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

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
    def __init__(self, project_root: str) -> None:
        model_path = os.path.join(project_root, 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError('hand_landmarker.task not found in project root')

        self.hand_tracker = hand_landmarker.HandLandmarker.create_from_options(
            hand_landmarker.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        )
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.timestamp_ms = 0
        self._lock = threading.Lock()

    def analyze_frame(self, frame_bgr: np.ndarray) -> dict:
        frame = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self._lock:
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            results = self.hand_tracker.detect_for_video(mp_image, self.timestamp_ms)
            self.timestamp_ms += 33

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

                index_up = hand_landmarks[8].y < hand_landmarks[6].y < hand_landmarks[5].y
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

    def close(self) -> None:
        self.hand_tracker.close()


INFERENCE_ENGINE: InferenceEngine | None = None


class ProjectRootHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        project_root = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=project_root, **kwargs)

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != '/api/analyze':
            self._send_json(404, {'error': 'Not found'})
            return

        if INFERENCE_ENGINE is None:
            self._send_json(503, {'error': 'Inference engine unavailable'})
            return

        content_length = int(self.headers.get('Content-Length', '0'))
        if content_length <= 0:
            self._send_json(400, {'error': 'Empty body'})
            return

        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode('utf-8'))
            image_data = payload.get('image', '')
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]

            encoded = base64.b64decode(image_data)
            np_buffer = np.frombuffer(encoded, dtype=np.uint8)
            frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Invalid image')

            result = INFERENCE_ENGINE.analyze_frame(frame)
            self._send_json(200, result)
        except Exception as exc:  # broad catch to avoid killing server on malformed requests
            logger.exception('Failed to analyze frame: %s', exc)
            self._send_json(400, {'error': 'Invalid request payload'})


def run_web_mode(host: str, port: int, open_browser: bool) -> None:
    global INFERENCE_ENGINE
    project_root = os.path.dirname(os.path.abspath(__file__))
    INFERENCE_ENGINE = InferenceEngine(project_root)

    server = ThreadingHTTPServer((host, port), ProjectRootHandler)
    url = f"http://{host}:{port}/index.html"

    logger.info("Serving browser app at %s", url)
    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down web server")
    finally:
        server.server_close()
        if INFERENCE_ENGINE is not None:
            INFERENCE_ENGINE.close()


def run_desktop_mode() -> None:
    desktop_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'desktop_app.py')
    runpy.run_path(desktop_script, run_name='__main__')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BrainRot Tracker launcher')
    parser.add_argument('--mode', choices=['web', 'desktop'], default='web', help='Run browser or desktop mode')
    parser.add_argument('--host', default='127.0.0.1', help='Host for web mode')
    parser.add_argument('--port', type=int, default=8000, help='Port for web mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open browser in web mode')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == 'desktop':
        run_desktop_mode()
        return

    run_web_mode(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
