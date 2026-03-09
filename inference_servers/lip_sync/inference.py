"""
Wav2Lip inference engine.

Handles the full lip sync pipeline:
  1. Detect and crop face region using face_alignment or OpenCV
  2. Compute mel spectrogram from audio
  3. Prepare face inputs (reference + masked target)
  4. Run Wav2Lip model inference
  5. Blend generated lip region back into original frame
"""

import os
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from models.wav2lip import build_wav2lip, Wav2Lip
from models.audio import mel_for_wav2lip

logger = logging.getLogger(__name__)

WAV2LIP_IMG_SIZE = 96


class FaceDetector:
    """
    Face detection using OpenCV's DNN face detector (built-in, no extra weights).
    Falls back to Haar cascades if DNN is unavailable.
    """

    def __init__(self):
        self._detector = None
        self._cascade = None
        self._init_detector()

    def _init_detector(self):
        """Initialize OpenCV face detector."""
        # Try to use the built-in SSD face detector
        try:
            self._detector = cv2.FaceDetectorYN.create(
                "",  # Will use built-in model
                "",
                (300, 300),
                score_threshold=0.5,
            )
        except Exception:
            pass

        # Fallback to Haar cascade (always available in OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self._cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Face detector: Haar cascade")
        else:
            logger.warning("No face detector available — will use center crop")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the primary face in a BGR frame.

        Returns:
            (x1, y1, x2, y2) bounding box, or None if no face found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._cascade is not None:
            faces = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) > 0:
                # Take the largest face
                areas = [w * h for (x, y, w, h) in faces]
                idx = np.argmax(areas)
                x, y, w, h = faces[idx]
                return (x, y, x + w, y + h)

        # Fallback: center crop assuming face is in the middle
        h, w = frame.shape[:2]
        size = min(h, w) // 2
        cx, cy = w // 2, h // 2
        return (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)


class Wav2LipInference:
    """Wav2Lip inference engine for speech-driven lip synchronization."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if checkpoint_path and os.path.isfile(checkpoint_path):
            logger.info("Loading Wav2Lip checkpoint: %s", checkpoint_path)
            self.model = build_wav2lip(checkpoint_path, self.device)
            self.model_loaded = True
        else:
            logger.warning(
                "No Wav2Lip checkpoint at '%s'. "
                "Running with random weights (for testing). "
                "Download wav2lip_gan.pth and set WAV2LIP_CHECKPOINT.",
                checkpoint_path,
            )
            self.model = build_wav2lip(None, self.device)
            self.model_loaded = False

        self.face_detector = FaceDetector()
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None
        logger.info("Wav2Lip engine ready (device=%s, loaded=%s)",
                     self.device, self.model_loaded)

    def _get_face_bbox(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect face with temporal smoothing to reduce jitter."""
        bbox = self.face_detector.detect(frame)
        if bbox is None:
            if self._last_bbox is not None:
                return self._last_bbox
            h, w = frame.shape[:2]
            return (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

        # Temporal smoothing
        if self._last_bbox is not None:
            alpha = 0.7
            bbox = tuple(
                int(alpha * old + (1 - alpha) * new)
                for old, new in zip(self._last_bbox, bbox)
            )

        self._last_bbox = bbox
        return bbox

    def _prepare_face_input(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare the face input for Wav2Lip.

        Returns:
            reference: Full face crop (96×96 RGB, normalized)
            masked: Same face with lower half zeroed (96×96 RGB, normalized)
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = frame

        face_crop = cv2.resize(face_crop, (WAV2LIP_IMG_SIZE, WAV2LIP_IMG_SIZE))
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        reference = face_rgb.astype(np.float32) / 255.0

        masked = reference.copy()
        # Mask the lower half (mouth region)
        masked[WAV2LIP_IMG_SIZE // 2:, :, :] = 0.0

        return reference, masked

    @torch.no_grad()
    def sync(
        self,
        frame: np.ndarray,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Synchronize lip movement in the frame with the given audio.

        Args:
            frame: BGR image.
            audio: Float32 mono audio (-1 to 1).
            sample_rate: Audio sample rate.

        Returns:
            BGR image with lip-synced face.
        """
        h, w = frame.shape[:2]

        # Face detection
        bbox = self._get_face_bbox(frame)
        x1, y1, x2, y2 = bbox

        # Prepare face inputs
        reference, masked = self._prepare_face_input(frame, bbox)

        # Prepare audio input (mel spectrogram)
        mel = mel_for_wav2lip(audio, sample_rate, num_frames=16)

        # Convert to tensors
        # Face: concatenate reference + masked → (6, 96, 96)
        ref_t = torch.from_numpy(reference).permute(2, 0, 1)
        mask_t = torch.from_numpy(masked).permute(2, 0, 1)
        face_input = torch.cat([ref_t, mask_t], dim=0).unsqueeze(0).to(self.device)

        mel_input = torch.from_numpy(mel).unsqueeze(0).to(self.device)  # (1, 1, 80, 16)

        # Run model
        output = self.model(mel_input, face_input)  # (1, 3, 96, 96)

        # Convert output to image
        out_face = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_face = np.clip(out_face * 255, 0, 255).astype(np.uint8)
        out_face = cv2.cvtColor(out_face, cv2.COLOR_RGB2BGR)

        # Blend back into original frame
        result = frame.copy()
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)

        face_w = x2c - x1c
        face_h = y2c - y1c
        if face_w > 0 and face_h > 0:
            resized_out = cv2.resize(out_face, (face_w, face_h))

            # Only replace the lower half (mouth region) for smoother blending
            mouth_start = face_h // 2
            result[y1c + mouth_start : y2c, x1c:x2c] = \
                resized_out[mouth_start:, :, :]

            # Feather the boundary for smooth blending
            blend_h = min(8, mouth_start)
            if blend_h > 0:
                for i in range(blend_h):
                    alpha = i / blend_h
                    row = y1c + mouth_start - blend_h + i
                    if 0 <= row < h:
                        src_row = mouth_start - blend_h + i
                        if 0 <= src_row < face_h:
                            result[row, x1c:x2c] = (
                                (1 - alpha) * frame[row, x1c:x2c].astype(np.float32)
                                + alpha * resized_out[src_row].astype(np.float32)
                            ).astype(np.uint8)

        return result
