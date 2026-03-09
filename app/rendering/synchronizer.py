"""
Audio-video synchronization module.
Manages timing alignment between processed video frames and audio chunks.
"""

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimestampedFrame:
    """A video frame tagged with its capture timestamp."""
    frame: np.ndarray
    timestamp: float


@dataclass
class TimestampedAudio:
    """An audio chunk tagged with its capture timestamp."""
    audio: np.ndarray
    timestamp: float


class AVSynchronizer:
    """
    Synchronizes video frames and audio chunks by timestamp.

    Uses a simple buffer-and-match strategy: for each video frame,
    find the closest audio chunk within an acceptable time window.
    """

    def __init__(self, max_drift_ms: float = 80.0, buffer_size: int = 30):
        self._max_drift = max_drift_ms / 1000.0
        self._video_buffer: Deque[TimestampedFrame] = deque(maxlen=buffer_size)
        self._audio_buffer: Deque[TimestampedAudio] = deque(maxlen=buffer_size * 3)
        self._lock = threading.Lock()
        self._frame_time = 0.0
        self._audio_time = 0.0

    def push_video(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add a processed video frame to the sync buffer."""
        ts = timestamp or time.monotonic()
        with self._lock:
            self._video_buffer.append(TimestampedFrame(frame=frame, timestamp=ts))
            self._frame_time = ts

    def push_audio(self, audio: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add a processed audio chunk to the sync buffer."""
        ts = timestamp or time.monotonic()
        with self._lock:
            self._audio_buffer.append(TimestampedAudio(audio=audio, timestamp=ts))
            self._audio_time = ts

    def pop_synced_pair(self) -> Optional[tuple]:
        """
        Retrieve the next synchronized (frame, audio) pair.

        Returns:
            Tuple of (np.ndarray frame, np.ndarray audio), or None
            if no synchronized pair is available.
        """
        with self._lock:
            if not self._video_buffer:
                return None

            video_item = self._video_buffer[0]

            # Find closest audio chunk
            best_audio = None
            best_drift = float("inf")
            best_idx = -1

            for i, audio_item in enumerate(self._audio_buffer):
                drift = abs(video_item.timestamp - audio_item.timestamp)
                if drift < best_drift:
                    best_drift = drift
                    best_audio = audio_item
                    best_idx = i

            if best_audio is not None and best_drift <= self._max_drift:
                # Consume both
                self._video_buffer.popleft()
                if best_idx >= 0:
                    del self._audio_buffer[best_idx]
                return (video_item.frame, best_audio.audio)

            # If no matching audio, still emit the frame with silence
            if self._audio_buffer or best_drift > self._max_drift * 2:
                self._video_buffer.popleft()
                return (video_item.frame, None)

            return None

    @property
    def drift_ms(self) -> float:
        """Current drift between video and audio streams in milliseconds."""
        return abs(self._frame_time - self._audio_time) * 1000.0

    @property
    def video_buffer_size(self) -> int:
        return len(self._video_buffer)

    @property
    def audio_buffer_size(self) -> int:
        return len(self._audio_buffer)
