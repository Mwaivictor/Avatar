"""FastAPI server — REST API and web interface for the avatar system."""

import asyncio
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import AppConfig
from app.controller import AvatarController
from app.output.virtual_camera import detect_virtual_camera
from app.output.virtual_microphone import detect_virtual_microphone

logger = logging.getLogger(__name__)

_controller: Optional[AvatarController] = None
_config: Optional[AppConfig] = None
_custom_profiles: dict = {}

VOICE_PROFILES_DIR = Path("voice_profiles")


def _load_saved_profiles() -> dict:
    """Load all saved voice profiles from disk."""
    profiles = {}
    if not VOICE_PROFILES_DIR.exists():
        return profiles
    for meta_file in VOICE_PROFILES_DIR.glob("*/profile.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            sid = data.get("speaker_id", meta_file.parent.name)
            profiles[sid] = data
        except Exception:
            logger.warning("Failed to load profile: %s", meta_file)
    return profiles


def _save_profile(speaker_id: str, profile_data: dict, wav_bytes: bytes = None) -> None:
    """Save a voice profile (metadata + optional WAV) to disk."""
    safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in speaker_id)
    profile_dir = VOICE_PROFILES_DIR / safe_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    meta = {"speaker_id": speaker_id, **profile_data}
    (profile_dir / "profile.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    if wav_bytes:
        (profile_dir / "sample.wav").write_bytes(wav_bytes)
    logger.info("Voice profile '%s' saved to %s", speaker_id, profile_dir)


def get_controller() -> AvatarController:
    if _controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    return _controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global _controller, _config, _custom_profiles
    _config = AppConfig()
    _controller = AvatarController(_config)

    # Load default avatar if available
    if os.path.exists(_config.avatar_image_path):
        _controller.load_avatar(_config.avatar_image_path)

    # Restore saved voice profiles from disk
    _custom_profiles.update(_load_saved_profiles())
    if _custom_profiles:
        logger.info("Loaded %d saved voice profile(s)", len(_custom_profiles))

    logger.info("Avatar system initialized")
    yield

    # Shutdown
    if _controller and _controller.is_running:
        await _controller.stop()
    logger.info("Avatar system shut down")


app = FastAPI(
    title="Avatar Transformation System",
    description="Real-time avatar and voice transformation API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files
os.makedirs("static/avatars", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Health & Status ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status")
async def get_status():
    ctrl = get_controller()
    return {
        "running": ctrl.is_running,
        "mode": ctrl.mode,
        "stats": ctrl.stats.to_dict(),
    }


@app.get("/api/services/health")
async def check_services():
    ctrl = get_controller()
    results = await ctrl.check_services()
    return {"services": results}


# ─── Pipeline Control ─────────────────────────────────────────────

@app.post("/api/pipeline/start")
async def start_pipeline(
    enable_virtual_cam: bool = True,
    enable_virtual_mic: bool = True,
    mode: str = "full",
):
    ctrl = get_controller()
    if ctrl.is_running:
        return {"message": "Pipeline already running"}

    if mode not in ("full", "audio"):
        raise HTTPException(status_code=400, detail="Mode must be 'full' or 'audio'")

    ctrl.start(
        enable_virtual_cam=enable_virtual_cam and mode == "full",
        enable_virtual_mic=enable_virtual_mic,
        mode=mode,
    )
    return {
        "message": f"Pipeline started ({mode} mode)",
        "mode": mode,
        "virtual_camera": enable_virtual_cam and mode == "full",
        "virtual_microphone": enable_virtual_mic,
    }


@app.post("/api/pipeline/stop")
async def stop_pipeline():
    ctrl = get_controller()
    if not ctrl.is_running:
        return {"message": "Pipeline not running"}
    await ctrl.stop()
    return {"message": "Pipeline stopped"}


# ─── Avatar Management ────────────────────────────────────────────

@app.post("/api/avatar/upload")
async def upload_avatar(file: UploadFile = File(...)):
    ctrl = get_controller()
    contents = await file.read()

    # Validate image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Save to disk
    filename = "avatar_current.png"
    save_path = os.path.join("static", "avatars", filename)
    cv2.imwrite(save_path, img)

    ctrl.set_avatar_from_array(img)
    return {"message": "Avatar uploaded", "path": save_path}


@app.post("/api/voice/speaker")
async def set_speaker(speaker_id: str):
    ctrl = get_controller()
    ctrl.set_speaker(speaker_id)
    return {"message": f"Speaker set to {speaker_id}"}


@app.get("/api/voice/speakers")
async def list_speakers():
    """List all available voice profiles (built-in + custom)."""
    ctrl = get_controller()
    fallback = {
        "default": {"name": "default", "f0_mean": 0, "formant_shift": 1.0},
        "male_1": {"name": "male_deep", "f0_mean": 95.0, "formant_shift": 0.88},
        "male_2": {"name": "male_mid", "f0_mean": 120.0, "formant_shift": 0.94},
        "female_1": {"name": "female_bright", "f0_mean": 220.0, "formant_shift": 1.18},
        "female_2": {"name": "female_warm", "f0_mean": 195.0, "formant_shift": 1.12},
    }
    try:
        speakers = await ctrl.list_speakers()
        if not speakers:
            speakers = dict(fallback)
    except Exception:
        speakers = dict(fallback)
    # Merge in-memory custom profiles
    speakers.update(_custom_profiles)
    # Merge saved profiles from disk
    speakers.update(_load_saved_profiles())
    return {"speakers": speakers}


@app.post("/api/voice/upload")
async def upload_voice_sample(file: UploadFile = File(...), speaker_id: str = ""):
    """
    Upload a voice sample (.wav) to create a custom speaker profile.

    The system analyzes the pitch and timbre of the uploaded voice and
    creates a new speaker profile you can select from the dropdown.
    Requires at least 1 second of clear speech.
    """
    import base64

    # Require a profile name
    speaker_id = speaker_id.strip()
    if not speaker_id or speaker_id == "custom":
        raise HTTPException(status_code=400, detail="Profile name is required")

    ctrl = get_controller()

    contents = await file.read()
    if len(contents) < 1000:
        raise HTTPException(status_code=400, detail="File too small — need at least 1 second of speech")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large — max 10 MB")

    # Decode WAV to raw PCM
    try:
        import wave
        wav_io = io.BytesIO(contents)
        with wave.open(wav_io, "rb") as wf:
            if wf.getnchannels() > 2:
                raise HTTPException(status_code=400, detail="Only mono or stereo WAV supported")
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_pcm = wf.readframes(n_frames)
            sample_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {e}")

    # Convert to int16 mono
    audio = np.frombuffer(raw_pcm, dtype=np.int16 if sample_width == 2 else np.uint8)
    if sample_width == 1:
        audio = ((audio.astype(np.int16) - 128) * 256)
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import scipy.signal
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = scipy.signal.resample(audio.astype(np.float32), num_samples).astype(np.int16)
        sample_rate = 16000

    # Forward to voice conversion service for analysis
    audio_b64 = base64.b64encode(audio.tobytes()).decode("utf-8")
    result = await ctrl.analyze_voice(audio_b64, sample_rate, speaker_id)

    if result is None:
        raise HTTPException(status_code=502, detail="Voice conversion service unavailable")
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Auto-select the new profile
    ctrl.set_speaker(speaker_id)
    logger.info("Voice profile '%s' created and auto-selected", speaker_id)

    # Track custom profile locally so it survives Docker timeouts
    if "profile" in result:
        profile_data = {"name": speaker_id, **result["profile"]}
        _custom_profiles[speaker_id] = profile_data
        # Persist to disk for reuse across restarts
        _save_profile(speaker_id, profile_data, wav_bytes=contents)

    return result


# ─── Video Streaming ──────────────────────────────────────────────

def _generate_mjpeg():
    """Generator that yields MJPEG frames for streaming."""
    ctrl = get_controller()
    while ctrl.is_running:
        frame = ctrl.get_latest_frame()
        if frame is None:
            frame = ctrl.get_preview_frame()
        if frame is None:
            # Generate blank frame
            frame = np.zeros(
                (
                    _config.rendering.output_height,
                    _config.rendering.output_width,
                    3,
                ),
                dtype=np.uint8,
            )

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


@app.get("/api/stream/video")
async def video_stream():
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/preview")
async def preview_frame():
    ctrl = get_controller()
    frame = ctrl.get_preview_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available")
    _, buffer = cv2.imencode(".jpg", frame)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()), media_type="image/jpeg"
    )


def _generate_webcam_mjpeg():
    """Generator that yields MJPEG frames from the raw webcam input."""
    ctrl = get_controller()
    while ctrl.is_running:
        frame = ctrl.get_preview_frame()
        if frame is None:
            time.sleep(0.03)
            continue
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.get("/api/stream/webcam")
async def webcam_stream():
    """MJPEG stream of raw webcam input with face-tracking overlay."""
    return StreamingResponse(
        _generate_webcam_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─── Virtual Devices ──────────────────────────────────────────────

@app.get("/api/devices")
async def get_virtual_devices():
    """Detect available virtual camera and microphone devices.

    Returns the device names as they appear in video call apps so the
    user knows exactly which device to select in Zoom, Meet, Teams, etc.
    """
    cam_info = detect_virtual_camera()
    mic_info = detect_virtual_microphone()

    ctrl = get_controller()
    # If pipeline is running, override with actual active device names
    if ctrl.is_running:
        if ctrl._virtual_cam and ctrl._virtual_cam.device_name:
            cam_info["device_name"] = ctrl._virtual_cam.device_name
            cam_info["available"] = True

    return {
        "virtual_camera": cam_info,
        "virtual_microphone": mic_info,
    }


# ─── Web Interface ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the full dashboard from static/dashboard.html."""
    dashboard_path = os.path.join("static", "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    return HTMLResponse("<h1>Dashboard not found</h1><p>Place dashboard.html in static/</p>")
