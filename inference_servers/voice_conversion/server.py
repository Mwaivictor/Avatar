"""
Voice Conversion Inference Server — HuBERT + WORLD Vocoder.

Real voice conversion using:
  - HuBERT (pretrained, auto-downloaded from HuggingFace) for content extraction
  - WORLD vocoder for pitch analysis and resynthesis
  - Log-domain F0 normalization for pitch conversion
  - Formant shifting and spectral tilt for timbre transformation

Requires no custom trained weights — HuBERT downloads from HuggingFace Hub
and WORLD is pure signal processing.
"""

import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from models.pipeline import VoiceConversionPipeline, SpeakerProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_HUBERT = os.getenv("VC_USE_HUBERT", "true").lower() == "true"
pipeline: Optional[VoiceConversionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initializing voice conversion pipeline (hubert=%s)...", USE_HUBERT)
    pipeline = VoiceConversionPipeline(
        sample_rate=16000,
        use_hubert=USE_HUBERT,
    )
    # Pre-load HuBERT if enabled (so first request isn't slow)
    if USE_HUBERT and pipeline.content_encoder is not None:
        try:
            pipeline.content_encoder.load()
        except Exception:
            logger.exception("Failed to load HuBERT — continuing without it")
    logger.info("Voice conversion pipeline ready")
    yield
    logger.info("Shutting down voice conversion pipeline")


app = FastAPI(title="Voice Conversion Service (HuBERT + WORLD)", lifespan=lifespan)


class ConvertRequest(BaseModel):
    audio_data: str           # base64-encoded int16 PCM
    sample_rate: int = 16000
    speaker_id: str = "default"


class ConvertResponse(BaseModel):
    converted_audio: str      # base64-encoded int16 PCM
    sample_rate: int


class ProfileRequest(BaseModel):
    speaker_id: str
    f0_mean: float
    f0_std: float = 25.0
    spectral_tilt: float = 1.0
    formant_shift: float = 1.0
    gain: float = 1.0
    breathiness: float = 0.0


class AnalyzeRequest(BaseModel):
    audio_data: str           # base64-encoded int16 PCM
    sample_rate: int = 16000
    speaker_id: str           # ID to register the profile under


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "voice_conversion",
        "model": "HuBERT + WORLD",
        "hubert_enabled": USE_HUBERT,
    }


@app.post("/convert_voice", response_model=ConvertResponse)
async def convert_voice(req: ConvertRequest):
    """Convert input speech to the target speaker voice profile."""
    # Decode audio
    raw = base64.b64decode(req.audio_data)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Run conversion
    converted = pipeline.convert(
        audio, speaker_id=req.speaker_id, sample_rate=req.sample_rate
    )

    # Encode result
    int_audio = (converted * 32768).astype(np.int16)
    b64 = base64.b64encode(int_audio.tobytes()).decode("utf-8")

    return ConvertResponse(converted_audio=b64, sample_rate=req.sample_rate)


@app.get("/speakers")
async def list_speakers():
    """List available speaker profiles."""
    profiles = {}
    for sid, p in pipeline.profiles.items():
        profiles[sid] = {
            "name": p.name,
            "f0_mean": p.f0_mean,
            "formant_shift": p.formant_shift,
        }
    return {"speakers": profiles}


@app.post("/speakers/add")
async def add_speaker(req: ProfileRequest):
    """Register a custom speaker voice profile."""
    profile = SpeakerProfile(
        name=req.speaker_id,
        f0_mean=req.f0_mean,
        f0_std=req.f0_std,
        spectral_tilt=req.spectral_tilt,
        formant_shift=req.formant_shift,
        gain=req.gain,
        breathiness=req.breathiness,
    )
    pipeline.add_profile(req.speaker_id, profile)
    return {"message": f"Speaker '{req.speaker_id}' added"}


@app.post("/speakers/analyze")
async def analyze_voice(req: AnalyzeRequest):
    """
    Analyze an uploaded voice sample and create a speaker profile from it.

    The system extracts F0 (pitch) statistics and spectral characteristics
    from the audio and registers a new speaker profile that can be selected
    for voice conversion.
    """
    raw = base64.b64decode(req.audio_data)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) < pipeline.sample_rate:  # Need at least 1 second
        return {"error": "Audio too short — need at least 1 second of speech"}

    # Analyze with WORLD vocoder
    params = pipeline.vocoder.analyze(audio)
    f0 = params["f0"]
    sp = params["sp"]
    ap = params["ap"]

    # Extract statistics from voiced frames only
    voiced = f0 > 0
    if not np.any(voiced):
        return {"error": "No voiced speech detected — please upload a sample with clear speech"}

    f0_voiced = f0[voiced]
    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))

    # Estimate spectral tilt: ratio of high-freq to low-freq energy
    mid = sp.shape[1] // 2
    low_energy = np.mean(sp[voiced, :mid])
    high_energy = np.mean(sp[voiced, mid:])
    spectral_tilt = float(high_energy / (low_energy + 1e-8))
    spectral_tilt = np.clip(spectral_tilt, 0.5, 2.0)

    # Estimate formant shift relative to average speaker (~150 Hz mean)
    avg_f0 = 150.0
    formant_shift = float(np.clip(f0_mean / avg_f0, 0.6, 1.8))

    # Estimate breathiness from mean aperiodicity of voiced frames
    # Aperiodicity values: 0 = fully periodic (resonant), 1 = fully noisy
    mean_ap = float(np.mean(ap[voiced]))
    # Map to a -0.3 to +0.3 scale relative to typical value (~0.1)
    breathiness = float(np.clip((mean_ap - 0.1) * 3.0, -0.3, 0.3))

    profile = SpeakerProfile(
        name=req.speaker_id,
        f0_mean=f0_mean,
        f0_std=f0_std,
        spectral_tilt=spectral_tilt,
        formant_shift=formant_shift,
        gain=1.0,
        breathiness=breathiness,
    )
    pipeline.add_profile(req.speaker_id, profile)

    return {
        "message": f"Voice profile '{req.speaker_id}' created from audio",
        "speaker_id": req.speaker_id,
        "profile": {
            "f0_mean": round(f0_mean, 1),
            "f0_std": round(f0_std, 1),
            "spectral_tilt": round(spectral_tilt, 3),
            "formant_shift": round(formant_shift, 3),
            "breathiness": round(breathiness, 3),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
