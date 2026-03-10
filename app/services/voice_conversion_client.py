"""
Client for the Voice Conversion inference service.
Sends audio chunks and receives voice-converted audio.
"""

import base64
import logging
from typing import Optional

import httpx
import numpy as np

from app.services.base_client import BaseServiceClient

logger = logging.getLogger(__name__)


class VoiceConversionClient(BaseServiceClient):
    """Communicates with the voice conversion inference server."""

    def __init__(self, base_url: str, timeout: float = 0.5):
        super().__init__(base_url, timeout)
        self._speaker_id: str = "default"

    def set_speaker(self, speaker_id: str) -> None:
        """Set the target speaker voice profile."""
        self._speaker_id = speaker_id
        logger.info("Voice conversion speaker set to: %s", speaker_id)

    async def convert(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> Optional[np.ndarray]:
        """
        Send an audio chunk for voice conversion.

        Args:
            audio_chunk: Float32 audio array (normalized -1.0 to 1.0).
            sample_rate: Sample rate of the input audio.

        Returns:
            Float32 audio array with converted voice, or None on failure.
        """
        # Encode audio as base64 for transport
        audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "audio_data": audio_b64,
            "sample_rate": sample_rate,
            "speaker_id": self._speaker_id,
        }

        try:
            response = await self.post("/convert_voice", json_data=payload)
            data = response.json()

            converted_b64 = data.get("converted_audio")
            if converted_b64 is None:
                logger.warning("No converted_audio in service response")
                return None

            raw_bytes = base64.b64decode(converted_b64)
            converted = np.frombuffer(raw_bytes, dtype=np.int16).astype(
                np.float32
            ) / 32768.0
            return converted

        except Exception:
            logger.exception("Voice conversion request failed")
            return None

    async def analyze_voice(
        self,
        audio_b64: str,
        sample_rate: int,
        speaker_id: str,
    ) -> Optional[dict]:
        """
        Send a voice sample to the VC service for analysis.
        Creates a new speaker profile from the audio characteristics.
        """
        payload = {
            "audio_data": audio_b64,
            "sample_rate": sample_rate,
            "speaker_id": speaker_id,
        }
        try:
            # Voice analysis is a heavy one-time operation; use a longer timeout
            client = await self._get_client()
            response = await client.post(
                "/speakers/analyze",
                json=payload,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            logger.exception("Voice analysis request failed")
            return None

    async def list_speakers(self) -> dict:
        """Fetch available speaker profiles from the VC service."""
        try:
            response = await self.get("/speakers")
            data = response.json()
            return data.get("speakers", {})
        except Exception:
            logger.exception("List speakers request failed")
            return {}
