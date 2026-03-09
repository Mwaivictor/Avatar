"""
Lip Synchronization Inference Server — Wav2Lip.

Neural lip sync: takes an avatar frame and audio, generates lip movements
that match the speech using the Wav2Lip (GAN) model.

Set WAV2LIP_CHECKPOINT to the path of a pretrained Wav2Lip GAN checkpoint
(e.g. wav2lip_gan.pth).
"""

import base64
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from inference import Wav2LipInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = os.getenv("WAV2LIP_CHECKPOINT", "/app/checkpoints/wav2lip_gan.pth")

engine: Optional[Wav2LipInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initializing Wav2Lip inference engine...")
    engine = Wav2LipInference(checkpoint_path=CHECKPOINT_PATH)
    logger.info("Wav2Lip engine ready (model_loaded=%s)", engine.model_loaded)
    yield
    logger.info("Shutting down Wav2Lip engine")


app = FastAPI(title="Lip Synchronization Service (Wav2Lip)", lifespan=lifespan)


# ─── Request / Response models ────────────────────────────────────

class SyncRequest(BaseModel):
    avatar_frame: str  # base64-encoded JPEG
    audio_data: str  # base64-encoded int16 PCM
    sample_rate: int = 16000


class SyncResponse(BaseModel):
    synced_frame: str  # base64-encoded JPEG


# ─── Helpers ──────────────────────────────────────────────────────

def _decode_image(b64: str) -> Optional[np.ndarray]:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(img: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ─── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "lip_sync",
        "model": "Wav2Lip",
        "model_loaded": engine.model_loaded if engine else False,
    }


@app.post("/sync_lips", response_model=SyncResponse)
async def sync_lips(req: SyncRequest):
    """
    Adjust avatar mouth movement to match speech audio using Wav2Lip.
    """
    frame = _decode_image(req.avatar_frame)
    if frame is None:
        return SyncResponse(synced_frame=req.avatar_frame)

    # Decode audio from int16 PCM → float32
    raw_bytes = base64.b64decode(req.audio_data)
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) == 0:
        return SyncResponse(synced_frame=_encode_image(frame))

    # Run Wav2Lip inference
    synced = engine.sync(frame, audio, sample_rate=req.sample_rate)
    return SyncResponse(synced_frame=_encode_image(synced))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
