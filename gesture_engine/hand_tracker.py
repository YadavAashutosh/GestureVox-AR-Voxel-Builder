"""MediaPipe hand tracking wrapper — compatible with mediapipe 0.10.33+"""
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import numpy as np
import cv2
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

FINGER_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

SKELETON_COLORS = [
    (255, 80, 80),
    (255,180, 60),
    ( 80,255,120),
    ( 80,180,255),
    (200, 80,255),
]


class _FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class HandTracker:
    def __init__(self, max_hands=2, min_detection=0.7, min_tracking=0.6):
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading hand landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded.")

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_tracking,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_result = None

    def process(self, frame_rgb: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._last_result = self._landmarker.detect(mp_image)
        return self._last_result

    def get_landmarks(self, hand_idx=0):
        r = self._last_result
        if not r or not r.hand_landmarks or hand_idx >= len(r.hand_landmarks):
            return None
        lms_raw = r.hand_landmarks[hand_idx]
        return [_FakeLandmark(lm.x, lm.y, lm.z) for lm in lms_raw]

    def get_all_hands(self):
        r = self._last_result
        if not r or not r.hand_landmarks:
            return []
        out = []
        for i, lms_raw in enumerate(r.hand_landmarks):
            lms = [_FakeLandmark(lm.x, lm.y, lm.z) for lm in lms_raw]
            label = "Right"
            if r.handedness and i < len(r.handedness):
                label = r.handedness[i][0].category_name
            out.append((lms, label))
        return out

    def draw_skeleton(self, bgr_frame: np.ndarray, hand_idx=0, colors=None):
        lms = self.get_landmarks(hand_idx)
        if lms is None:
            return bgr_frame
        H, W = bgr_frame.shape[:2]

        def pt(idx):
            return int(lms[idx].x * W), int(lms[idx].y * H)

        finger_chains = [
            [0,1,2,3,4],
            [0,5,6,7,8],
            [0,9,10,11,12],
            [0,13,14,15,16],
            [0,17,18,19,20],
        ]
        if colors is None:
            colors = SKELETON_COLORS

        for chain, color in zip(finger_chains, colors):
            for a, b in zip(chain[:-1], chain[1:]):
                cv2.line(bgr_frame, pt(a), pt(b), color, 2, cv2.LINE_AA)
            for idx in chain:
                cv2.circle(bgr_frame, pt(idx), 4, color, -1, cv2.LINE_AA)
        return bgr_frame

    def close(self):
        self._landmarker.close()