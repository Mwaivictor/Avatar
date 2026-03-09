"""
Client for the Lip Synchronization inference service.
Adjusts avatar mouth movement to match generated speech audio.
"""

import base64
import logging
from typing import Optional

import cv2
import numpy as np

from app.services.base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class LipSyncClient(BaseServiceClient):
    """Communicates with the lip synchronization inference server."""

    def __init__(self, base_url: str, timeout: float = 0.5):
        super().__init__(base_url, timeout)

    async def synchronize(
        self,
        avatar_frame: np.ndarray,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """
        Synchronize avatar mouth movement with speech audio.

        Args:
            avatar_frame: BGR image of the animated avatar.
            audio_chunk: Float32 audio array corresponding to this frame.
            sample_rate: Audio sample rate.

        Returns:
            BGR image with corrected lip movement, or None on failure.
        """
        # Encode frame
        _, frame_buf = cv2.imencode(".jpg", avatar_frame)
        frame_b64 = base64.b64encode(frame_buf).decode("utf-8")

        # Encode audio
        audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "avatar_frame": frame_b64,
            "audio_data": audio_b64,
            "sample_rate": sample_rate,
        }

        try:
            response = await self.post("/sync_lips", json_data=payload)
            data = response.json()

            synced_b64 = data.get("synced_frame")
            if synced_b64 is None:
                logger.warning("No synced_frame in service response")
                return None

            frame_bytes = base64.b64decode(synced_b64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            return frame

        except Exception:
            logger.exception("Lip sync request failed")
            return None
