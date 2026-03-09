"""
Facial expression analysis module.
Computes high-level expression parameters from MediaPipe face landmarks.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.tracking.face_tracker import (
    LEFT_EYE_UPPER,
    LEFT_EYE_LOWER,
    RIGHT_EYE_UPPER,
    RIGHT_EYE_LOWER,
    UPPER_LIP,
    LOWER_LIP,
    LEFT_LIP_CORNER,
    RIGHT_LIP_CORNER,
    NOSE_TIP,
    CHIN,
    LEFT_EAR,
    RIGHT_EAR,
    FOREHEAD,
)

logger = logging.getLogger(__name__)


@dataclass
class ExpressionState:
    """Computed facial expression parameters."""

    blink_left: float = 0.0
    blink_right: float = 0.0
    mouth_open: float = 0.0
    smile: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0

    def to_dict(self) -> dict:
        return {
            "blink_left": round(self.blink_left, 4),
            "blink_right": round(self.blink_right, 4),
            "mouth_open": round(self.mouth_open, 4),
            "smile": round(self.smile, 4),
            "head_yaw": round(self.head_yaw, 2),
            "head_pitch": round(self.head_pitch, 2),
            "head_roll": round(self.head_roll, 2),
        }


class ExpressionAnalyzer:
    """Analyzes facial landmarks to produce expression parameters."""

    def __init__(self):
        self._prev_state: Optional[ExpressionState] = None
        # Smoothing factor for temporal filtering (0 = no smoothing, 1 = max)
        self.smoothing = 0.3

    def analyze(self, landmarks: np.ndarray) -> ExpressionState:
        """
        Compute expression state from 468-point landmark array.

        Args:
            landmarks: Array of shape (468, 3) with normalized coordinates.

        Returns:
            ExpressionState with computed parameters.
        """
        state = ExpressionState(
            blink_left=self._compute_eye_ratio(landmarks, LEFT_EYE_UPPER, LEFT_EYE_LOWER),
            blink_right=self._compute_eye_ratio(landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER),
            mouth_open=self._compute_mouth_open(landmarks),
            smile=self._compute_smile(landmarks),
            **self._compute_head_pose(landmarks),
        )

        # Apply temporal smoothing for stability
        if self._prev_state is not None:
            state = self._smooth(state, self._prev_state)
        self._prev_state = state

        return state

    def _compute_eye_ratio(
        self,
        landmarks: np.ndarray,
        upper_indices: list,
        lower_indices: list,
    ) -> float:
        """Compute eye aspect ratio (0 = open, 1 = closed)."""
        upper = np.mean([landmarks[i] for i in upper_indices], axis=0)
        lower = np.mean([landmarks[i] for i in lower_indices], axis=0)
        distance = np.linalg.norm(upper[:2] - lower[:2])
        # Normalize: typical eye opening is ~0.01-0.04 in normalized coords
        ratio = 1.0 - min(distance / 0.04, 1.0)
        return max(0.0, ratio)

    def _compute_mouth_open(self, landmarks: np.ndarray) -> float:
        """Compute mouth opening ratio (0 = closed, 1 = fully open)."""
        upper = landmarks[UPPER_LIP[0]]
        lower = landmarks[LOWER_LIP[0]]
        distance = np.linalg.norm(upper[:2] - lower[:2])
        # Normalize: typical range 0.0 - 0.08
        return min(distance / 0.08, 1.0)

    def _compute_smile(self, landmarks: np.ndarray) -> float:
        """Compute smile intensity from lip corner positions."""
        left_corner = landmarks[LEFT_LIP_CORNER[0]]
        right_corner = landmarks[RIGHT_LIP_CORNER[0]]
        nose = landmarks[NOSE_TIP[0]]

        # Horizontal spread relative to nose
        spread = np.linalg.norm(left_corner[:2] - right_corner[:2])
        # Vertical position of corners relative to nose (upward = smile)
        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        vertical = nose[1] - avg_corner_y

        # Combine metrics
        smile = min(max(spread * 3.0 + vertical * 10.0, 0.0), 1.0)
        return smile

    def _compute_head_pose(self, landmarks: np.ndarray) -> dict:
        """
        Estimate head orientation (yaw, pitch, roll) from landmark geometry.
        Returns angles in degrees.
        """
        nose = landmarks[NOSE_TIP[0]]
        chin = landmarks[CHIN[0]]
        left_ear = landmarks[LEFT_EAR[0]]
        right_ear = landmarks[RIGHT_EAR[0]]
        forehead = landmarks[FOREHEAD[0]]

        # Yaw: asymmetry between nose and ear midpoint
        ear_mid_x = (left_ear[0] + right_ear[0]) / 2
        yaw = (nose[0] - ear_mid_x) * 180.0

        # Pitch: vertical angle from forehead to chin
        face_vertical = chin[1] - forehead[1]
        face_depth = chin[2] - forehead[2]
        pitch = math.degrees(math.atan2(face_depth, face_vertical)) if face_vertical != 0 else 0.0

        # Roll: tilt of the ear-to-ear line
        ear_dy = right_ear[1] - left_ear[1]
        ear_dx = right_ear[0] - left_ear[0]
        roll = math.degrees(math.atan2(ear_dy, ear_dx)) if ear_dx != 0 else 0.0

        return {"head_yaw": yaw, "head_pitch": pitch, "head_roll": roll}

    def _smooth(self, current: ExpressionState, previous: ExpressionState) -> ExpressionState:
        """Apply exponential moving average smoothing."""
        a = self.smoothing
        return ExpressionState(
            blink_left=a * previous.blink_left + (1 - a) * current.blink_left,
            blink_right=a * previous.blink_right + (1 - a) * current.blink_right,
            mouth_open=a * previous.mouth_open + (1 - a) * current.mouth_open,
            smile=a * previous.smile + (1 - a) * current.smile,
            head_yaw=a * previous.head_yaw + (1 - a) * current.head_yaw,
            head_pitch=a * previous.head_pitch + (1 - a) * current.head_pitch,
            head_roll=a * previous.head_roll + (1 - a) * current.head_roll,
        )
