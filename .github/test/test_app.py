import os
import cv2
import mediapipe as mp

def test_files_exist():
    # Verify image paths from source 1
    assert os.path.exists("Monkey/monkeyknow.png")
    assert os.path.exists("Monkey/monkeythink.png")
    assert os.path.exists("hand_landmarker.task")

def test_model_initialization():
    # Verify the task file is valid and can be initialized by MediaPipe
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        assert landmarker is not None