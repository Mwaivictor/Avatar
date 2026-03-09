"""
Audio capture module using PyAudio.
Records microphone input in a dedicated thread, buffering chunks for processing.
"""

import threading
import logging
import queue
from typing import Optional

import numpy as np

from config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """Threaded microphone capture that buffers audio chunks."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._stream = None
        self._audio_interface = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._total_chunks = 0

    def start(self) -> None:
        """Open the microphone stream and begin recording."""
        import pyaudio

        self._audio_interface = pyaudio.PyAudio()
        self._stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Audio capture started: %d Hz, %d channels, chunk=%d",
            self.config.sample_rate,
            self.config.channels,
            self.config.chunk_size,
        )

    def _capture_loop(self) -> None:
        """Continuously read audio chunks from the microphone."""
        while self._running:
            try:
                raw_data = self._stream.read(
                    self.config.chunk_size, exception_on_overflow=False
                )
                audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(
                    np.float32
                ) / 32768.0

                try:
                    self._buffer.put_nowait(audio_array)
                except queue.Full:
                    # Drop oldest chunk to prevent unbounded memory growth
                    try:
                        self._buffer.get_nowait()
                    except queue.Empty:
                        pass
                    self._buffer.put_nowait(audio_array)

                self._total_chunks += 1

            except Exception:
                logger.exception("Error reading audio")

    def read(self) -> Optional[np.ndarray]:
        """Return the next audio chunk, or None if the buffer is empty."""
        try:
            return self._buffer.get_nowait()
        except queue.Empty:
            return None

    def read_accumulate(self, num_chunks: int) -> Optional[np.ndarray]:
        """Accumulate multiple chunks into a single array for batch processing."""
        chunks = []
        for _ in range(num_chunks):
            try:
                chunks.append(self._buffer.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return None
        return np.concatenate(chunks)

    @property
    def buffer_size(self) -> int:
        return self._buffer.qsize()

    @property
    def total_chunks(self) -> int:
        return self._total_chunks

    @property
    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """Stop recording and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio_interface is not None:
            self._audio_interface.terminate()
        logger.info("Audio capture stopped after %d chunks", self._total_chunks)
