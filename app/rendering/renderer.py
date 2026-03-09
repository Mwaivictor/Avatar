"""
Rendering module.
Assembles final output frames by compositing avatar over background,
applying overlays, and encoding the output stream.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from config import RenderingConfig

logger = logging.getLogger(__name__)


class Renderer:
    """Composites and encodes the final avatar video output."""

    def __init__(self, config: RenderingConfig):
        self.config = config
        self._background: Optional[np.ndarray] = None
        self._overlay_text: Optional[str] = None
        self._frame_count = 0

    def set_background(self, background: np.ndarray) -> None:
        """Set a static background image (resized to output dimensions)."""
        self._background = cv2.resize(
            background, (self.config.output_width, self.config.output_height)
        )

    def set_overlay_text(self, text: Optional[str]) -> None:
        """Set optional overlay text (e.g., username, debug info)."""
        self._overlay_text = text

    def render_frame(
        self,
        avatar_frame: Optional[np.ndarray],
        original_frame: Optional[np.ndarray] = None,
        debug_info: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Produce the final output frame.

        Args:
            avatar_frame: The animated avatar image (from AI service).
            original_frame: The raw webcam frame (used as fallback).
            debug_info: Optional dict of debug metrics to overlay.

        Returns:
            BGR image at the configured output resolution.
        """
        w, h = self.config.output_width, self.config.output_height

        # Start with background or black canvas
        if self._background is not None:
            canvas = self._background.copy()
        else:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Primary: use animated avatar frame
        if avatar_frame is not None:
            resized = cv2.resize(avatar_frame, (w, h))
            canvas = resized
        elif original_frame is not None:
            # Fallback: show original webcam frame
            canvas = cv2.resize(original_frame, (w, h))

        # Apply overlay text
        if self._overlay_text:
            cv2.putText(
                canvas,
                self._overlay_text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Debug overlay
        if debug_info:
            y_offset = 20
            for key, value in debug_info.items():
                text = f"{key}: {value}"
                cv2.putText(
                    canvas,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                y_offset += 18

        self._frame_count += 1
        return canvas

    def encode_frame_jpeg(self, frame: np.ndarray, quality: int = 85) -> bytes:
        """Encode a frame as JPEG bytes."""
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode(".jpg", frame, params)
        return buffer.tobytes()

    @property
    def frame_count(self) -> int:
        return self._frame_count
