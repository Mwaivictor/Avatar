"""
Face Animation Inference Server — First Order Motion Model (FOMM).

Real neural motion transfer: detects abstract keypoints in source and
driving images, computes dense optical flow, and generates animated frames
via an occlusion-aware generator network.

Supports two animation modes:
  1. Motion transfer: provide `driving_frame` (webcam image) — FOMM extracts
     keypoints from both source and driving to animate.
  2. Parametric: provide `expression_state` only — keypoints are perturbed
     directly from expression coefficients.

Set FOMM_CHECKPOINT to the path of a pretrained checkpoint (e.g. vox-cpk.pth.tar).
"""

import base64
import os
import logging
import hashlib
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from inference import FOMMInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = os.getenv("FOMM_CHECKPOINT", "/app/checkpoints/vox-cpk.pth.tar")

engine: Optional[FOMMInference] = None
# Cache to avoid re-setting the same source image on every request
_source_hash: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initializing FOMM inference engine...")
    engine = FOMMInference(checkpoint_path=CHECKPOINT_PATH)
    logger.info("FOMM engine ready (model_loaded=%s)", engine.model_loaded)
    yield
    logger.info("Shutting down FOMM engine")


app = FastAPI(title="Face Animation Service (FOMM)", lifespan=lifespan)


# ─── Request / Response models ────────────────────────────────────

class AnimateRequest(BaseModel):
    avatar_image: str                            # base64-encoded PNG/JPG
    driving_frame: Optional[str] = None          # base64-encoded webcam frame
    facial_landmarks: Optional[List[List[float]]] = None
    expression_state: Optional[dict] = None


class AnimateResponse(BaseModel):
    animated_frame: str  # base64-encoded JPEG


# ─── Helpers ──────────────────────────────────────────────────────

def _decode_image(b64: str) -> Optional[np.ndarray]:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(img: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def _image_hash(b64: str) -> str:
    return hashlib.md5(b64[:2048].encode()).hexdigest()


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "face_animation",
        "model": "FOMM",
        "model_loaded": engine.model_loaded if engine else False,
    }


@app.post("/animate_face", response_model=AnimateResponse)
async def animate_face(req: AnimateRequest):
    """
    Animate the avatar image using FOMM neural motion transfer.

    Priority: driving_frame > expression_state.
    If driving_frame is provided, full FOMM motion transfer is used.
    Otherwise falls back to parametric animation from expression_state.
    """
    global _source_hash

    avatar = _decode_image(req.avatar_image)
    if avatar is None:
        return AnimateResponse(animated_frame=req.avatar_image)

    # Update source only if the avatar image changed
    current_hash = _image_hash(req.avatar_image)
    if current_hash != _source_hash:
        engine.set_source(avatar)
        _source_hash = current_hash
        logger.info("Source avatar updated")

    # Mode 1: full motion transfer from driving frame
    if req.driving_frame is not None:
        driving = _decode_image(req.driving_frame)
        if driving is not None:
            animated = engine.animate(
                driving, output_size=(avatar.shape[1], avatar.shape[0])
            )
            return AnimateResponse(animated_frame=_encode_image(animated))

    # Mode 2: parametric animation from expression state
    if req.expression_state is not None:
        animated = engine.animate_from_expression(
            req.expression_state,
            output_size=(avatar.shape[1], avatar.shape[0]),
        )
        return AnimateResponse(animated_frame=_encode_image(animated))

    # No driving data — return source unchanged
    return AnimateResponse(animated_frame=_encode_image(avatar))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
