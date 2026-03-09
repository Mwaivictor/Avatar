"""
Client for the Face Animation inference service.
Sends facial landmarks and avatar image, receives animated frames.
"""

import base64
import logging
from typing import Optional

import cv2
import numpy as np

from app.services.base_client import BaseServiceClient
from app.tracking.expression_analyzer import ExpressionState

logger = logging.getLogger(__name__)


class FaceAnimationClient(BaseServiceClient):
    """Communicates with the face animation inference server."""

    def __init__(self, base_url: str, timeout: float = 0.5):
        super().__init__(base_url, timeout)
        self._avatar_b64: Optional[str] = None

    def set_avatar(self, avatar_image: np.ndarray) -> None:
        """Cache the base64-encoded avatar image for reuse."""
        _, buffer = cv2.imencode(".png", avatar_image)
        self._avatar_b64 = base64.b64encode(buffer).decode("utf-8")
        logger.info("Avatar image cached (%d bytes encoded)", len(self._avatar_b64))

    async def animate(
        self,
        landmarks: np.ndarray,
        expression: ExpressionState,
        driving_frame: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Request an animated avatar frame from the service.

        Args:
            landmarks: (468, 3) facial landmark array.
            expression: Computed expression parameters.
            driving_frame: Raw webcam frame for FOMM motion transfer.

        Returns:
            BGR image array of the animated avatar frame, or None on failure.
        """
        if self._avatar_b64 is None:
            logger.error("No avatar image set. Call set_avatar() first.")
            return None

        payload = {
            "avatar_image": self._avatar_b64,
            "facial_landmarks": landmarks.tolist(),
            "expression_state": expression.to_dict(),
        }

        # Include driving frame for neural motion transfer
        if driving_frame is not None:
            _, buf = cv2.imencode(".jpg", driving_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            payload["driving_frame"] = base64.b64encode(buf).decode("utf-8")

        try:
            response = await self.post("/animate_face", json_data=payload)
            data = response.json()

            frame_b64 = data.get("animated_frame")
            if frame_b64 is None:
                logger.warning("No animated_frame in service response")
                return None

            frame_bytes = base64.b64decode(frame_b64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            return frame

        except Exception:
            logger.exception("Face animation request failed")
            return None
