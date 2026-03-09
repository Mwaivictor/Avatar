"""
Face tracking module using MediaPipe Face Mesh.
Extracts 468 facial landmarks from video frames in real time.
"""

import logging
from typing import Optional, List, Tuple

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


class FaceTracker:
    """Real-time face landmark detection using MediaPipe Face Mesh."""

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_landmarks: Optional[np.ndarray] = None
        logger.info("FaceTracker initialized (max_faces=%d)", max_faces)

    def process_frame(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Detect face landmarks in a BGR frame.

        Returns:
            Array of shape (468, 3) with normalized (x, y, z) coordinates,
            or None if no face is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        landmarks = np.array(
            [(lm.x, lm.y, lm.z) for lm in face.landmark], dtype=np.float32
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
        self._face_mesh.close()
        logger.info("FaceTracker closed")
