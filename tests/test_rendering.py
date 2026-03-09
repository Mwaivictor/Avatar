"""
Tests for the rendering and synchronization modules.
"""

import time

import numpy as np
import pytest

from config import RenderingConfig
from app.rendering.renderer import Renderer
from app.rendering.synchronizer import AVSynchronizer


class TestRenderer:
    def setup_method(self):
        self.config = RenderingConfig()
        self.renderer = Renderer(self.config)

    def test_render_black_frame_when_no_input(self):
        frame = self.renderer.render_frame(avatar_frame=None)
        assert frame.shape == (self.config.output_height, self.config.output_width, 3)
        # Should be mostly black
        assert frame.mean() < 10

    def test_render_with_avatar_frame(self):
        avatar = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = self.renderer.render_frame(avatar_frame=avatar)
        assert frame.shape == (self.config.output_height, self.config.output_width, 3)

    def test_render_with_fallback(self):
        original = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frame = self.renderer.render_frame(avatar_frame=None, original_frame=original)
        assert frame.mean() > 100  # Should show the fallback

    def test_render_with_debug_info(self):
        avatar = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = self.renderer.render_frame(
            avatar_frame=avatar,
            debug_info={"FPS": "30", "Face": "Yes"},
        )
        assert frame is not None

    def test_frame_count_increments(self):
        avatar = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(5):
            self.renderer.render_frame(avatar_frame=avatar)
        assert self.renderer.frame_count == 5

    def test_encode_frame_jpeg(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        jpg_bytes = self.renderer.encode_frame_jpeg(frame)
        assert isinstance(jpg_bytes, bytes)
        assert len(jpg_bytes) > 0
        # Check JPEG header
        assert jpg_bytes[:2] == b"\xff\xd8"


class TestAVSynchronizer:
    def setup_method(self):
        self.sync = AVSynchronizer(max_drift_ms=100)

    def test_push_and_pop_synced_pair(self):
        ts = time.monotonic()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        audio = np.zeros(1024, dtype=np.float32)

        self.sync.push_video(frame, ts)
        self.sync.push_audio(audio, ts + 0.01)  # 10ms drift

        pair = self.sync.pop_synced_pair()
        assert pair is not None
        assert pair[0] is not None
        assert pair[1] is not None

    def test_returns_none_when_empty(self):
        pair = self.sync.pop_synced_pair()
        assert pair is None

    def test_drops_video_when_audio_missing(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.sync.push_video(frame)
        # No audio pushed, but buffer should eventually emit
        pair = self.sync.pop_synced_pair()
        # With empty audio buffer, it should return frame with None audio
        if pair is not None:
            assert pair[1] is None

    def test_drift_measurement(self):
        ts = time.monotonic()
        self.sync.push_video(np.zeros((10, 10, 3), dtype=np.uint8), ts)
        self.sync.push_audio(np.zeros(100, dtype=np.float32), ts + 0.05)
        assert self.sync.drift_ms >= 40  # ~50ms drift
