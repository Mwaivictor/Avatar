"""
Tests for the face tracking and expression analysis modules.
"""

import numpy as np
import pytest

from app.tracking.expression_analyzer import ExpressionAnalyzer, ExpressionState


class TestExpressionAnalyzer:
    """Tests for ExpressionAnalyzer."""

    def setup_method(self):
        self.analyzer = ExpressionAnalyzer()

    def _make_neutral_landmarks(self) -> np.ndarray:
        """Create a synthetic neutral-face landmark set (468 points)."""
        rng = np.random.default_rng(42)
        landmarks = rng.random((468, 3)).astype(np.float32)
        # Place key landmarks in reasonable positions
        landmarks[1] = [0.5, 0.45, 0.0]   # nose tip
        landmarks[152] = [0.5, 0.85, 0.0]  # chin
        landmarks[10] = [0.5, 0.1, 0.0]   # forehead
        landmarks[234] = [0.15, 0.4, 0.0]  # left ear
        landmarks[454] = [0.85, 0.4, 0.0]  # right ear
        landmarks[13] = [0.5, 0.6, 0.0]   # upper lip
        landmarks[14] = [0.5, 0.62, 0.0]  # lower lip
        landmarks[61] = [0.38, 0.62, 0.0]  # left lip corner
        landmarks[291] = [0.62, 0.62, 0.0] # right lip corner
        # Eyes
        landmarks[159] = [0.35, 0.35, 0.0]
        landmarks[145] = [0.35, 0.38, 0.0]
        landmarks[144] = [0.35, 0.36, 0.0]
        landmarks[153] = [0.35, 0.39, 0.0]
        landmarks[386] = [0.65, 0.35, 0.0]
        landmarks[374] = [0.65, 0.38, 0.0]
        landmarks[373] = [0.65, 0.36, 0.0]
        landmarks[380] = [0.65, 0.39, 0.0]
        return landmarks

    def test_analyze_returns_expression_state(self):
        landmarks = self._make_neutral_landmarks()
        result = self.analyzer.analyze(landmarks)
        assert isinstance(result, ExpressionState)

    def test_expression_state_to_dict(self):
        state = ExpressionState(
            blink_left=0.1, blink_right=0.2,
            mouth_open=0.5, smile=0.3,
            head_yaw=10.0, head_pitch=-5.0, head_roll=2.0,
        )
        d = state.to_dict()
        assert "blink_left" in d
        assert "head_yaw" in d
        assert d["mouth_open"] == 0.5

    def test_mouth_open_increases_with_distance(self):
        landmarks = self._make_neutral_landmarks()
        # Closed mouth
        landmarks[14] = [0.5, 0.605, 0.0]
        closed = self.analyzer.analyze(landmarks)
        self.analyzer._prev_state = None  # Reset smoothing

        # Open mouth
        landmarks[14] = [0.5, 0.72, 0.0]
        opened = self.analyzer.analyze(landmarks)
        assert opened.mouth_open > closed.mouth_open

    def test_head_yaw_responds_to_asymmetry(self):
        landmarks = self._make_neutral_landmarks()
        # Nose centered
        result_center = self.analyzer.analyze(landmarks)
        self.analyzer._prev_state = None

        # Nose shifted right
        landmarks[1] = [0.6, 0.45, 0.0]
        result_right = self.analyzer.analyze(landmarks)

        assert abs(result_right.head_yaw) > abs(result_center.head_yaw)

    def test_smoothing_reduces_jitter(self):
        landmarks = self._make_neutral_landmarks()
        result1 = self.analyzer.analyze(landmarks)

        # Slight perturbation
        landmarks_noisy = landmarks.copy()
        landmarks_noisy += np.random.default_rng(99).normal(0, 0.001, landmarks.shape).astype(np.float32)
        result2 = self.analyzer.analyze(landmarks_noisy)

        # Smoothed result should be close to previous
        assert abs(result2.mouth_open - result1.mouth_open) < 0.1


class TestExpressionStateSerialize:
    def test_to_dict_keys(self):
        state = ExpressionState()
        d = state.to_dict()
        expected_keys = {
            "blink_left", "blink_right", "mouth_open", "smile",
            "head_yaw", "head_pitch", "head_roll",
        }
        assert set(d.keys()) == expected_keys
