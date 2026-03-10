"""
Audio capture module using sounddevice.
Records microphone input in a dedicated thread, buffering chunks for processing.
"""

import threading
import logging
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

from config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """Threaded microphone capture that buffers audio chunks."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._stream = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buffer: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._total_chunks = 0

    def start(self) -> None:
        """Open the microphone stream and begin recording."""
        self._running = True

        def _audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio capture status: %s", status)
            audio_array = indata[:, 0].copy() if indata.shape[1] > 1 else indata.flatten().copy()
            try:
                self._buffer.put_nowait(audio_array)
            except queue.Full:
                try:
                    self._buffer.get_nowait()
                except queue.Empty:
                    pass
                self._buffer.put_nowait(audio_array)
            self._total_chunks += 1

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            dtype="float32",
            callback=_audio_callback,
        )
        self._stream.start()
        logger.info(
            "Audio capture started: %d Hz, %d channels, chunk=%d",
            self.config.sample_rate,
            self.config.channels,
            self.config.chunk_size,
        )

    def _capture_loop(self) -> None:
        """Unused — sounddevice uses a callback instead."""
        pass

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
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        logger.info("Audio capture stopped after %d chunks", self._total_chunks)
