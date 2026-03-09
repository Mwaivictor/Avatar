"""
Tests for the capture modules (unit tests, no hardware required).
"""

import numpy as np
import pytest

from config import VideoConfig, AudioConfig


class TestVideoConfig:
    def test_default_values(self):
        config = VideoConfig()
        assert config.camera_index == 0
        assert config.frame_width == 640
        assert config.frame_height == 480
        assert config.target_fps == 30


class TestAudioConfig:
    def test_default_values(self):
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_size == 1024
        assert config.format_width == 2
