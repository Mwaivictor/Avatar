"""
Virtual microphone output module.
Outputs transformed audio to a virtual audio device.

On Windows, this requires a virtual audio cable driver such as:
- VB-Audio Virtual Cable
- Virtual Audio Cable (VAC)

The module writes audio to the virtual device's input using sounddevice.
"""

import logging
import sys
import threading
import queue
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Common virtual audio device name patterns across platforms
_VIRTUAL_MIC_PATTERNS = [
    "cable",           # VB-Audio Virtual Cable (Windows)
    "virtual",         # Generic virtual audio drivers
    "vb-audio",        # VB-Audio variants
    "blackhole",       # BlackHole (macOS)
    "soundflower",     # Soundflower (macOS legacy)
    "pulse",           # PulseAudio virtual sink (Linux)
    "pipewire",        # PipeWire virtual device (Linux)
    "loopback",        # Loopback (macOS)
]


def detect_virtual_microphone() -> dict:
    """Detect available virtual audio output devices.

    Returns a dict with:
        device_name: The name to select in video call apps as microphone
        available: Whether a virtual audio device was found
        device_index: sounddevice device index (or None)
        all_candidates: List of all matching virtual devices
        instructions: Setup instructions if not available
    """
    try:
        devices = sd.query_devices()
        candidates = []

        for i, dev in enumerate(devices):
            if dev["max_output_channels"] <= 0:
                continue
            name_lower = dev["name"].lower()
            for pattern in _VIRTUAL_MIC_PATTERNS:
                if pattern in name_lower:
                    candidates.append({
                        "index": i,
                        "name": dev["name"],
                        "channels": dev["max_output_channels"],
                    })
                    break

        if candidates:
            best = candidates[0]
            # The output device on our side feeds into the virtual cable.
            # In the call app, the user selects the cable's INPUT side as mic.
            # For VB-Audio: we output to "CABLE Input", user selects "CABLE Output" in their app.
            display_name = best["name"]
            if "cable input" in display_name.lower():
                display_name = display_name.replace("Input", "Output").replace("input", "Output")
            return {
                "device_name": display_name,
                "available": True,
                "device_index": best["index"],
                "all_candidates": candidates,
                "instructions": None,
            }

        # No virtual device found
        if sys.platform == "win32":
            return {
                "device_name": "CABLE Output (VB-Audio Virtual Cable)",
                "available": False,
                "device_index": None,
                "all_candidates": [],
                "instructions": "Install VB-Audio Virtual Cable from https://vb-audio.com/Cable/",
            }
        elif sys.platform == "darwin":
            return {
                "device_name": "BlackHole 2ch",
                "available": False,
                "device_index": None,
                "all_candidates": [],
                "instructions": "Install BlackHole: brew install blackhole-2ch",
            }
        else:
            return {
                "device_name": "Avatar Virtual Microphone",
                "available": False,
                "device_index": None,
                "all_candidates": [],
                "instructions": "Create a PulseAudio/PipeWire virtual sink for audio routing",
            }

    except Exception as e:
        logger.warning("Failed to detect virtual audio devices: %s", e)
        return {
            "device_name": "CABLE Output (VB-Audio Virtual Cable)",
            "available": False,
            "device_index": None,
            "all_candidates": [],
            "instructions": "Install sounddevice and a virtual audio cable driver",
        }


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
        self._stream = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._chunk_count = 0

    def _find_device_index(self) -> Optional[int]:
        """Find the output device index matching the configured device name."""
        if self.device_name is not None:
            # User specified a device name — search for it
            devices = sd.query_devices()
            for i, info in enumerate(devices):
                if (
                    self.device_name.lower() in info["name"].lower()
                    and info["max_output_channels"] > 0
                ):
                    logger.info(
                        "Found virtual audio device: %s (index %d)",
                        info["name"],
                        i,
                    )
                    return i

            logger.warning(
                "Virtual audio device '%s' not found, trying auto-detect",
                self.device_name,
            )

        # Auto-detect virtual audio device
        info = detect_virtual_microphone()
        if info["available"] and info["device_index"] is not None:
            logger.info(
                "Auto-detected virtual audio device: %s (index %d)",
                info["device_name"],
                info["device_index"],
            )
            return info["device_index"]

        logger.warning("No virtual audio device found, using system default")
        return None

    def start(self) -> None:
        """Open the audio output stream."""
        device_index = self._find_device_index()

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            dtype="float32",
            device=device_index,
        )
        self._stream.start()
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
                # sounddevice expects float32 [-1, 1] shaped (frames, channels)
                if audio.ndim == 1:
                    audio = audio.reshape(-1, 1)
                self._stream.write(audio)
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
            self._stream.stop()
            self._stream.close()
        logger.info(
            "Virtual microphone output stopped after %d chunks",
            self._chunk_count,
        )
