"""
Tests for AI service client modules.
Uses httpx mock to test API communication without real servers.
"""

import base64
import json

import numpy as np
import cv2
import pytest
import httpx

from app.services.face_animation_client import FaceAnimationClient
from app.services.voice_conversion_client import VoiceConversionClient
from app.services.lip_sync_client import LipSyncClient
from app.tracking.expression_analyzer import ExpressionState


def _make_test_image(w=64, h=64) -> np.ndarray:
    """Create a small test image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _encode_image_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")


def _encode_audio_b64(length: int = 1024) -> str:
    audio = np.random.randint(-32768, 32767, length, dtype=np.int16)
    return base64.b64encode(audio.tobytes()).decode("utf-8")


class TestFaceAnimationClient:
    @pytest.mark.asyncio
    async def test_set_avatar(self):
        client = FaceAnimationClient("http://localhost:8001")
        img = _make_test_image()
        client.set_avatar(img)
        assert client._avatar_b64 is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_animate_without_avatar_returns_none(self):
        client = FaceAnimationClient("http://localhost:8001")
        landmarks = np.random.rand(468, 3).astype(np.float32)
        expr = ExpressionState()
        result = await client.animate(landmarks, expr)
        assert result is None
        await client.close()


class TestVoiceConversionClient:
    @pytest.mark.asyncio
    async def test_set_speaker(self):
        client = VoiceConversionClient("http://localhost:8002")
        client.set_speaker("male_1")
        assert client._speaker_id == "male_1"
        await client.close()


class TestLipSyncClient:
    @pytest.mark.asyncio
    async def test_client_creation(self):
        client = LipSyncClient("http://localhost:8003")
        assert client.base_url == "http://localhost:8003"
        await client.close()
