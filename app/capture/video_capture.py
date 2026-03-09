"""
Video capture module using OpenCV.
Captures frames from the system webcam in a dedicated thread.
"""

import threading
import time
import logging
from typing import Optional

import cv2
import numpy as np

from config import VideoConfig

logger = logging.getLogger(__name__)


class VideoCapture:
    """Threaded webcam capture that continuously reads frames."""

    def __init__(self, config: VideoConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._fps = 0.0

    def start(self) -> None:
        """Open the camera and begin capturing frames."""
        self._cap = cv2.VideoCapture(self.config.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.config.camera_index}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Video capture started: %dx%d @ %d FPS target",
            self.config.frame_width,
            self.config.frame_height,
            self.config.target_fps,
        )

    def _capture_loop(self) -> None:
        """Continuously read frames from the camera."""
        fps_start = time.monotonic()
        frame_count = 0

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.monotonic()

    def read(self) -> Optional[np.ndarray]:
        """Return the latest captured frame (thread-safe copy)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """Stop capture and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("Video capture stopped after %d frames", self._frame_count)
