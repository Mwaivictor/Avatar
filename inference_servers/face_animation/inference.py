"""
FOMM inference engine.

Handles model loading, image preprocessing, keypoint caching,
and the full animate pipeline.
"""

import os
import logging
from typing import Optional, Dict

import torch
import numpy as np
import cv2

from models.fomm import build_fomm, FOMM

logger = logging.getLogger(__name__)

# Input resolution expected by the FOMM vox-cpk checkpoint
FOMM_RESOLUTION = 256


class FOMMInference:
    """First Order Motion Model inference engine."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("FOMM device: %s", self.device)

        if checkpoint_path and os.path.isfile(checkpoint_path):
            logger.info("Loading FOMM checkpoint: %s", checkpoint_path)
            self.model = build_fomm(checkpoint_path, self.device)
            self.model_loaded = True
        else:
            logger.warning(
                "No FOMM checkpoint found at '%s'. "
                "Model will run with random weights (for architecture testing). "
                "Download vox-cpk.pth.tar and set FOMM_CHECKPOINT env var.",
                checkpoint_path,
            )
            self.model = build_fomm(None, self.device)
            self.model_loaded = False

        self._source_kp: Optional[Dict] = None
        self._source_tensor: Optional[torch.Tensor] = None

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize to 256x256, normalize to [0,1], convert to BCHW tensor."""
        img = cv2.resize(image, (FOMM_RESOLUTION, FOMM_RESOLUTION))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
        return tensor.to(self.device)

    def _postprocess(self, tensor: torch.Tensor, target_size: tuple) -> np.ndarray:
        """Convert BCHW [0,1] tensor back to BGR uint8 image."""
        img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if target_size != (FOMM_RESOLUTION, FOMM_RESOLUTION):
            img = cv2.resize(img, target_size)
        return img

    def set_source(self, source_image: np.ndarray) -> None:
        """
        Set and cache the source (avatar) image and its keypoints.
        Call this once when the avatar changes to avoid redundant KP detection.
        """
        self._source_tensor = self._preprocess(source_image)
        with torch.no_grad():
            self._source_kp = self.model.kp_detector(self._source_tensor)
        logger.info("Source keypoints cached (%d KPs)", self._source_kp["value"].shape[1])

    @torch.no_grad()
    def animate(self, driving_frame: np.ndarray,
                output_size: Optional[tuple] = None) -> np.ndarray:
        """
        Animate the source image using motion from the driving frame.

        Args:
            driving_frame: BGR webcam frame.
            output_size: (width, height) of output, defaults to driving frame size.

        Returns:
            Animated BGR image.
        """
        if self._source_tensor is None or self._source_kp is None:
            raise RuntimeError("Source image not set. Call set_source() first.")

        h, w = driving_frame.shape[:2]
        out_size = output_size or (w, h)

        driving_tensor = self._preprocess(driving_frame)
        kp_driving = self.model.kp_detector(driving_tensor)

        generated = self.model.animate(
            self._source_tensor, self._source_kp, kp_driving
        )
        result = self._postprocess(generated["prediction"], out_size)
        return result

    @torch.no_grad()
    def animate_from_expression(
        self,
        expression_state: dict,
        output_size: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Animate using expression parameters when no driving frame is available.
        Maps expression coefficients to keypoint deltas for parametric control.

        Args:
            expression_state: Dict with blink_left, blink_right, mouth_open,
                              smile, head_yaw, head_pitch, head_roll.
            output_size: (width, height) of output.

        Returns:
            Animated BGR image.
        """
        if self._source_tensor is None or self._source_kp is None:
            raise RuntimeError("Source image not set. Call set_source() first.")

        out_size = output_size or (FOMM_RESOLUTION, FOMM_RESOLUTION)

        # Create driving keypoints by perturbing source keypoints
        kp_driving = {
            "value": self._source_kp["value"].clone(),
        }
        if "jacobian" in self._source_kp:
            kp_driving["jacobian"] = self._source_kp["jacobian"].clone()

        num_kp = kp_driving["value"].shape[1]
        yaw = expression_state.get("head_yaw", 0.0) / 90.0
        pitch = expression_state.get("head_pitch", 0.0) / 90.0
        roll_rad = np.radians(expression_state.get("head_roll", 0.0))
        mouth_open = expression_state.get("mouth_open", 0.0)
        blink_l = expression_state.get("blink_left", 0.0)
        blink_r = expression_state.get("blink_right", 0.0)

        # Global head motion: shift all keypoints
        shift = torch.tensor(
            [[yaw * 0.15, pitch * 0.15]], device=self.device
        )
        kp_driving["value"] = kp_driving["value"] + shift.unsqueeze(0)

        # Apply roll via 2D rotation on keypoints
        if abs(roll_rad) > 0.01:
            cos_r = np.cos(roll_rad)
            sin_r = np.sin(roll_rad)
            rot = torch.tensor(
                [[cos_r, -sin_r], [sin_r, cos_r]],
                device=self.device, dtype=torch.float32,
            )
            center = kp_driving["value"].mean(dim=1, keepdim=True)
            kp_driving["value"] = torch.matmul(
                kp_driving["value"] - center, rot.T
            ) + center

        # Mouth: move bottom keypoints down (heuristic: last 2 KPs often mouth)
        if mouth_open > 0.05 and num_kp >= 4:
            kp_driving["value"][0, -1, 1] += mouth_open * 0.06
            kp_driving["value"][0, -2, 1] += mouth_open * 0.04

        # Blink: move eye keypoints closer vertically (heuristic: KPs 2-5)
        if num_kp >= 6:
            if blink_l > 0.3:
                kp_driving["value"][0, 2, 1] += blink_l * 0.02
            if blink_r > 0.3:
                kp_driving["value"][0, 3, 1] += blink_r * 0.02

        generated = self.model.animate(
            self._source_tensor, self._source_kp, kp_driving
        )
        return self._postprocess(generated["prediction"], out_size)
