"""
Main application controller — the orchestration layer.
Coordinates all pipeline components: capture → tracking → AI services →
rendering → virtual device output.
"""

import asyncio
import time
import logging
import threading
from typing import Optional

import cv2
import numpy as np

from config import AppConfig
from app.capture.video_capture import VideoCapture
from app.capture.audio_capture import AudioCapture
from app.tracking.face_tracker import FaceTracker
from app.tracking.expression_analyzer import ExpressionAnalyzer
from app.services.face_animation_client import FaceAnimationClient
from app.services.voice_conversion_client import VoiceConversionClient
from app.services.lip_sync_client import LipSyncClient
from app.rendering.renderer import Renderer
from app.rendering.synchronizer import AVSynchronizer
from app.output.virtual_camera import VirtualCameraOutput
from app.output.virtual_microphone import VirtualMicrophoneOutput

logger = logging.getLogger(__name__)


class PipelineStats:
    """Tracks real-time performance metrics."""

    def __init__(self):
        self.video_fps: float = 0.0
        self.processing_fps: float = 0.0
        self.audio_latency_ms: float = 0.0
        self.av_drift_ms: float = 0.0
        self.face_detected: bool = False
        self.services_healthy: dict = {
            "face_animation": False,
            "voice_conversion": False,
            "lip_sync": False,
        }
        self.total_frames_processed: int = 0
        self.total_audio_chunks_processed: int = 0

    def to_dict(self) -> dict:
        return {
            "video_fps": round(self.video_fps, 1),
            "processing_fps": round(self.processing_fps, 1),
            "audio_latency_ms": round(self.audio_latency_ms, 1),
            "av_drift_ms": round(self.av_drift_ms, 1),
            "face_detected": self.face_detected,
            "services": self.services_healthy.copy(),
            "frames_processed": self.total_frames_processed,
            "audio_chunks_processed": self.total_audio_chunks_processed,
        }


class AvatarController:
    """
    Central orchestrator for the avatar transformation pipeline.

    Manages the complete data flow:
    1. Capture video frames and audio chunks
    2. Track face and extract expressions
    3. Send data to AI inference services
    4. Render and synchronize output
    5. Output to virtual devices
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.stats = PipelineStats()

        # Input layer
        self._video_capture = VideoCapture(config.video)
        self._audio_capture = AudioCapture(config.audio)

        # Tracking layer
        self._face_tracker = FaceTracker()
        self._expression_analyzer = ExpressionAnalyzer()

        # AI service clients
        self._face_anim = FaceAnimationClient(
            config.services.face_animation_url,
            config.services.request_timeout,
        )
        self._voice_conv = VoiceConversionClient(
            config.services.voice_conversion_url,
            config.services.request_timeout,
        )
        self._lip_sync = LipSyncClient(
            config.services.lip_sync_url,
            config.services.request_timeout,
        )

        # Rendering layer
        self._renderer = Renderer(config.rendering)
        self._synchronizer = AVSynchronizer()

        # Output layer
        self._virtual_cam: Optional[VirtualCameraOutput] = None
        self._virtual_mic: Optional[VirtualMicrophoneOutput] = None

        # Control
        self._running = False
        self._mode: str = "full"  # "full" or "audio"
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Avatar image
        self._avatar_image: Optional[np.ndarray] = None

    def load_avatar(self, image_path: str) -> bool:
        """Load the target avatar image."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to load avatar image: %s", image_path)
            return False
        self._avatar_image = img
        self._face_anim.set_avatar(img)
        logger.info("Avatar loaded: %s (%dx%d)", image_path, img.shape[1], img.shape[0])
        return True

    def set_avatar_from_array(self, image: np.ndarray) -> None:
        """Set avatar directly from a numpy array."""
        self._avatar_image = image
        self._face_anim.set_avatar(image)

    def set_speaker(self, speaker_id: str) -> None:
        """Set the target voice profile."""
        self._voice_conv.set_speaker(speaker_id)

    async def analyze_voice(self, audio_b64: str, sample_rate: int, speaker_id: str) -> dict:
        """Forward a voice sample to the VC service for analysis and profile creation."""
        return await self._voice_conv.analyze_voice(audio_b64, sample_rate, speaker_id)

    async def list_speakers(self) -> dict:
        """Fetch the list of available speaker profiles from the VC service."""
        return await self._voice_conv.list_speakers()

    async def check_services(self) -> dict:
        """Check health of all AI inference services."""
        results = await asyncio.gather(
            self._face_anim.health_check(),
            self._voice_conv.health_check(),
            self._lip_sync.health_check(),
            return_exceptions=True,
        )
        self.stats.services_healthy = {
            "face_animation": results[0] is True,
            "voice_conversion": results[1] is True,
            "lip_sync": results[2] is True,
        }
        return self.stats.services_healthy

    def start(
        self,
        enable_virtual_cam: bool = True,
        enable_virtual_mic: bool = True,
        mode: str = "full",
    ) -> None:
        """Start the transformation pipeline.

        Args:
            mode: "full" for video + audio, "audio" for voice-only.
        """
        self._mode = mode
        logger.info("Starting pipeline in %s mode...", mode)

        # Audio capture is always needed
        self._audio_capture.start()

        # Video capture and virtual camera only in full mode
        if mode == "full":
            self._video_capture.start()
            if enable_virtual_cam:
                try:
                    self._virtual_cam = VirtualCameraOutput(
                        width=self.config.rendering.output_width,
                        height=self.config.rendering.output_height,
                        fps=self.config.rendering.output_fps,
                    )
                    self._virtual_cam.start()
                except Exception:
                    logger.warning("Virtual camera unavailable, skipping")
                    self._virtual_cam = None

        if enable_virtual_mic:
            try:
                self._virtual_mic = VirtualMicrophoneOutput(
                    sample_rate=self.config.audio.sample_rate,
                    channels=self.config.audio.channels,
                    chunk_size=self.config.audio.chunk_size,
                    device_name=self.config.virtual_mic_name or None,
                )
                self._virtual_mic.start()
            except Exception:
                logger.warning("Virtual microphone unavailable, skipping")
                self._virtual_mic = None

        # Start processing loops
        self._running = True
        self._loop = asyncio.new_event_loop()

        if mode == "full":
            self._video_thread = threading.Thread(
                target=self._video_processing_loop, daemon=True
            )
            self._video_thread.start()

        self._audio_thread = threading.Thread(
            target=self._audio_processing_loop, daemon=True
        )
        self._audio_thread.start()

        logger.info("Pipeline started successfully (%s mode)", mode)

    def _video_processing_loop(self) -> None:
        """Main video processing loop running in its own thread."""
        asyncio.set_event_loop(self._loop)
        fps_counter = 0
        fps_start = time.monotonic()

        while self._running:
            frame = self._video_capture.read()
            if frame is None:
                time.sleep(0.001)
                continue

            timestamp = time.monotonic()

            # Track face
            landmarks = self._face_tracker.process_frame(frame)
            self.stats.face_detected = landmarks is not None

            # Process through AI services
            avatar_frame = None
            if landmarks is not None:
                expression = self._expression_analyzer.analyze(landmarks)

                # Request animated avatar from service
                try:
                    avatar_frame = self._loop.run_until_complete(
                        self._face_anim.animate(
                            landmarks, expression, driving_frame=frame
                        )
                    )
                except Exception:
                    logger.debug("Face animation service unavailable")

            # Render the final frame
            debug_info = None
            if self.config.debug:
                debug_info = {
                    "FPS": f"{self.stats.processing_fps:.1f}",
                    "Face": "Yes" if self.stats.face_detected else "No",
                    "Drift": f"{self.stats.av_drift_ms:.0f}ms",
                }

            output_frame = self._renderer.render_frame(
                avatar_frame=avatar_frame,
                original_frame=frame,
                debug_info=debug_info,
            )

            # Push to synchronizer and virtual camera
            self._synchronizer.push_video(output_frame, timestamp)

            if self._virtual_cam is not None:
                try:
                    self._virtual_cam.send_frame(output_frame)
                except Exception:
                    logger.debug("Failed to send frame to virtual camera")

            self.stats.total_frames_processed += 1
            fps_counter += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 1.0:
                self.stats.processing_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.monotonic()

        self.stats.video_fps = self._video_capture.fps
        self.stats.av_drift_ms = self._synchronizer.drift_ms

    def _audio_processing_loop(self) -> None:
        """Main audio processing loop running in its own thread."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()

        while self._running:
            # Accumulate a reasonable chunk for processing
            audio = self._audio_capture.read_accumulate(4)
            if audio is None:
                time.sleep(0.01)
                continue

            timestamp = time.monotonic()
            converted = None

            # Voice conversion through AI service
            try:
                converted = loop.run_until_complete(
                    self._voice_conv.convert(audio, self.config.audio.sample_rate)
                )
            except Exception:
                logger.debug("Voice conversion service unavailable")

            output_audio = converted if converted is not None else audio

            # Push to synchronizer and virtual mic
            self._synchronizer.push_audio(output_audio, timestamp)

            if self._virtual_mic is not None:
                self._virtual_mic.send_audio(output_audio)

            self.stats.total_audio_chunks_processed += 1

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recently rendered frame (for web streaming)."""
        pair = self._synchronizer.pop_synced_pair()
        if pair is not None:
            return pair[0]
        return None

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get a preview frame combining webcam input with tracking overlay."""
        frame = self._video_capture.read()
        if frame is None:
            return None

        landmarks = self._face_tracker.last_landmarks
        if landmarks is not None:
            h, w = frame.shape[:2]
            pixel_landmarks = self._face_tracker.get_pixel_landmarks(landmarks, w, h)
            for point in pixel_landmarks:
                cv2.circle(
                    frame,
                    (int(point[0]), int(point[1])),
                    1,
                    (0, 255, 0),
                    -1,
                )
        return frame

    async def stop(self) -> None:
        """Stop all pipeline components and release resources."""
        logger.info("Stopping avatar transformation pipeline...")
        self._running = False

        if self._video_thread is not None:
            self._video_thread.join(timeout=3.0)
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=3.0)

        self._audio_capture.stop()
        if self._mode == "full":
            self._video_capture.stop()
            self._face_tracker.close()

        if self._virtual_cam is not None:
            self._virtual_cam.stop()
        if self._virtual_mic is not None:
            self._virtual_mic.stop()

        # Close async HTTP clients from the current (FastAPI) event loop
        await self._face_anim.close()
        await self._voice_conv.close()
        await self._lip_sync.close()

        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()

        logger.info(
            "Pipeline stopped. Processed %d frames, %d audio chunks.",
            self.stats.total_frames_processed,
            self.stats.total_audio_chunks_processed,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def mode(self) -> str:
        return self._mode
