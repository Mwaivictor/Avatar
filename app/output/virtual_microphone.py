"""
Virtual microphone output module.
Outputs transformed audio to a virtual audio device.

On Windows, this requires a virtual audio cable driver such as:
- VB-Audio Virtual Cable
- Virtual Audio Cable (VAC)

The module writes audio to the virtual device's input using PyAudio.
"""

import logging
import threading
import queue
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VirtualMicrophoneOutput:
    """
    Outputs audio to a virtual microphone / audio loopback device.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_name: Optional[str] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_name = device_name
        self._audio_interface = None
        self._stream = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._chunk_count = 0

    def _find_device_index(self) -> Optional[int]:
        """Find the output device index matching the configured device name."""
        if self.device_name is None:
            return None  # Use system default

        for i in range(self._audio_interface.get_device_count()):
            info = self._audio_interface.get_device_info_by_index(i)
            if (
                self.device_name.lower() in info["name"].lower()
                and info["maxOutputChannels"] > 0
            ):
                logger.info(
                    "Found virtual audio device: %s (index %d)",
                    info["name"],
                    i,
                )
                return i

        logger.warning(
            "Virtual audio device '%s' not found, using default",
            self.device_name,
        )
        return None

    def start(self) -> None:
        """Open the audio output stream."""
        import pyaudio

        self._audio_interface = pyaudio.PyAudio()
        device_index = self._find_device_index()

        kwargs = {
            "format": pyaudio.paInt16,
            "channels": self.channels,
            "rate": self.sample_rate,
            "output": True,
            "frames_per_buffer": self.chunk_size,
        }
        if device_index is not None:
            kwargs["output_device_index"] = device_index

        self._stream = self._audio_interface.open(**kwargs)
        self._running = True
        self._thread = threading.Thread(target=self._output_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Virtual microphone output started: %d Hz, %d channels",
            self.sample_rate,
            self.channels,
        )

    def _output_loop(self) -> None:
        """Continuously write audio chunks to the output device."""
        while self._running:
            try:
                audio = self._queue.get(timeout=0.1)
                # Convert float32 [-1, 1] to int16
                int_data = (audio * 32768).astype(np.int16)
                self._stream.write(int_data.tobytes())
                self._chunk_count += 1
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Error writing audio output")

    def send_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Queue an audio chunk for output.

        Args:
            audio_chunk: Float32 audio array (normalized -1.0 to 1.0).
        """
        try:
            self._queue.put_nowait(audio_chunk)
        except queue.Full:
            # Drop oldest to prevent blocking
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(audio_chunk)

    @property
    def chunk_count(self) -> int:
        return self._chunk_count

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """Close the output stream and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio_interface is not None:
            self._audio_interface.terminate()
        logger.info(
            "Virtual microphone output stopped after %d chunks",
            self._chunk_count,
        )
