"""
Virtual camera output module using pyvirtualcam.
Exposes the transformed video stream as a virtual webcam device.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VirtualCameraOutput:
    """
    Outputs frames to a virtual camera device.

    Requires a virtual camera backend to be installed:
    - Windows: OBS Virtual Camera (included with OBS Studio)
    - macOS: OBS Virtual Camera
    - Linux: v4l2loopback
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._cam = None
        self._frame_count = 0

    def start(self) -> None:
        """Initialize the virtual camera device."""
        try:
            import pyvirtualcam

            self._cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                print_fps=False,
            )
            logger.info(
                "Virtual camera started: %s (%dx%d @ %d FPS)",
                self._cam.device,
                self.width,
                self.height,
                self.fps,
            )
        except Exception:
            logger.exception(
                "Failed to start virtual camera. "
                "Ensure OBS Virtual Camera or v4l2loopback is installed."
            )
            raise

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Send a BGR frame to the virtual camera.

        Args:
            frame: BGR image array at the configured resolution.
        """
        if self._cam is None:
            return

        import cv2

        # pyvirtualcam expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ensure correct dimensions
        if rgb_frame.shape[:2] != (self.height, self.width):
            rgb_frame = cv2.resize(rgb_frame, (self.width, self.height))

        self._cam.send(rgb_frame)
        self._cam.sleep_until_next_frame()
        self._frame_count += 1

    @property
    def device_name(self) -> Optional[str]:
        return self._cam.device if self._cam else None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._cam is not None

    def stop(self) -> None:
        """Close the virtual camera device."""
        if self._cam is not None:
            self._cam.close()
            self._cam = None
        logger.info("Virtual camera stopped after %d frames", self._frame_count)
