"""
Face tracking module using MediaPipe Face Landmarker (tasks API).
Extracts 478 facial landmarks from video frames in real time.
"""

import logging
import os
import urllib.request
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Landmark index constants for expression analysis
LEFT_EYE_UPPER = [159, 145]
LEFT_EYE_LOWER = [144, 153]
RIGHT_EYE_UPPER = [386, 374]
RIGHT_EYE_LOWER = [373, 380]
UPPER_LIP = [13]
LOWER_LIP = [14]
LEFT_LIP_CORNER = [61]
RIGHT_LIP_CORNER = [291]
NOSE_TIP = [1]
CHIN = [152]
LEFT_EAR = [234]
RIGHT_EAR = [454]
FOREHEAD = [10]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model() -> str:
    """Download the face landmarker model if it doesn't exist."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH
    os.makedirs(_MODEL_DIR, exist_ok=True)
    logger.info("Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    logger.info("Downloaded face_landmarker.task to %s", _MODEL_PATH)
    return _MODEL_PATH


class FaceTracker:
    """Real-time face landmark detection using MediaPipe FaceLandmarker."""

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        model_path = _ensure_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        self._frame_ts_ms = 0
        self._last_landmarks: Optional[np.ndarray] = None
        logger.info("FaceTracker initialized (max_faces=%d)", max_faces)

    def process_frame(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect face landmarks in a BGR frame.

        Returns:
            Array of shape (478, 3) with normalized (x, y, z) coordinates,
            or None if no face is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
        )
        self._frame_ts_ms += 33  # ~30 FPS
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face], dtype=np.float32
        )
        self._last_landmarks = landmarks
        return landmarks

    def get_pixel_landmarks(
        self, landmarks: np.ndarray, frame_width: int, frame_height: int
    ) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates."""
        pixel = landmarks.copy()
        pixel[:, 0] *= frame_width
        pixel[:, 1] *= frame_height
        return pixel

    @property
    def last_landmarks(self) -> Optional[np.ndarray]:
        return self._last_landmarks

    def close(self) -> None:
        self._landmarker.close()
        logger.info("FaceTracker closed")
