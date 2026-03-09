"""
FastAPI server providing REST API and web interface for the avatar system.
Includes endpoints for control, streaming, configuration, health monitoring,
app detection, and permission management.
"""

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig
from app.controller import AvatarController
from app.permissions import PermissionManager, KNOWN_APPS
from app.app_detector import detect_running_apps

logger = logging.getLogger(__name__)

_controller: Optional[AvatarController] = None
_config: Optional[AppConfig] = None
_permissions: Optional[PermissionManager] = None


def get_controller() -> AvatarController:
    if _controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    return _controller


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global _controller, _config, _permissions
    _config = AppConfig()
    _controller = AvatarController(_config)
    _permissions = PermissionManager()

    # Load default avatar if available
    if os.path.exists(_config.avatar_image_path):
        _controller.load_avatar(_config.avatar_image_path)

    logger.info("Avatar system initialized")
    yield

    # Shutdown
    if _controller and _controller.is_running:
        _controller.stop()
    if _permissions:
        _permissions.revoke_all()
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
):
    ctrl = get_controller()
    if ctrl.is_running:
        return {"message": "Pipeline already running"}

    # Permission gate: require at least one granted permission
    if not _permissions.any_granted:
        raise HTTPException(
            status_code=403,
            detail="No permissions granted. Grant access to at least one application before starting.",
        )

    # Only enable virtual devices that have permission
    cam_allowed = enable_virtual_cam and _permissions.is_camera_allowed()
    mic_allowed = enable_virtual_mic and _permissions.is_microphone_allowed()

    ctrl.start(
        enable_virtual_cam=cam_allowed,
        enable_virtual_mic=mic_allowed,
    )
    return {
        "message": "Pipeline started",
        "virtual_camera": cam_allowed,
        "virtual_microphone": mic_allowed,
    }


@app.post("/api/pipeline/stop")
async def stop_pipeline():
    ctrl = get_controller()
    if not ctrl.is_running:
        return {"message": "Pipeline not running"}
    ctrl.stop()
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
    try:
        speakers = await ctrl.list_speakers()
        return {"speakers": speakers}
    except Exception:
        # Fallback: return the built-in names
        return {"speakers": {
            "default": {"name": "default", "f0_mean": 0, "formant_shift": 1.0},
            "male_1": {"name": "male_deep", "f0_mean": 95.0, "formant_shift": 0.88},
            "male_2": {"name": "male_mid", "f0_mean": 120.0, "formant_shift": 0.94},
            "female_1": {"name": "female_bright", "f0_mean": 220.0, "formant_shift": 1.18},
            "female_2": {"name": "female_warm", "f0_mean": 195.0, "formant_shift": 1.12},
        }}


@app.post("/api/voice/upload")
async def upload_voice_sample(file: UploadFile = File(...), speaker_id: str = "custom"):
    """
    Upload a voice sample (.wav) to create a custom speaker profile.

    The system analyzes the pitch and timbre of the uploaded voice and
    creates a new speaker profile you can select from the dropdown.
    Requires at least 1 second of clear speech.
    """
    import base64

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


# ─── Permission & App Detection ───────────────────────────────────

class GrantRequest(BaseModel):
    app_id: str
    virtual_camera: bool = True
    virtual_microphone: bool = True


@app.get("/api/apps/detect")
async def detect_apps():
    """Scan for running video call / streaming applications."""
    apps = detect_running_apps()
    # Also return known apps as hints even if not running
    known = [
        {"app_id": k, "display_name": v["display_name"], "icon": v["icon"]}
        for k, v in KNOWN_APPS.items()
        if k != "other"
    ]
    return {"apps": apps, "known_apps": known}


@app.get("/api/permissions")
async def list_permissions():
    """List all current permission records."""
    return {"permissions": _permissions.get_all_permissions()}


@app.get("/api/permissions/status")
async def permission_status():
    """Quick summary: is anything granted?"""
    return _permissions.get_status_summary()


@app.post("/api/permissions/grant")
async def grant_permission(req: GrantRequest):
    """Grant virtual device access for a specific app."""
    record = _permissions.grant_permission(
        req.app_id,
        virtual_camera=req.virtual_camera,
        virtual_microphone=req.virtual_microphone,
    )
    return {"message": f"Permission granted for {record.app_name}", "record": record.to_dict()}


@app.post("/api/permissions/revoke")
async def revoke_permission(app_id: str):
    """Revoke permission for a specific app."""
    record = _permissions.revoke_permission(app_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No permission found for that app")
    return {"message": f"Permission revoked for {record.app_name}", "record": record.to_dict()}


@app.post("/api/permissions/revoke-all")
async def revoke_all():
    """Revoke all granted permissions and stop pipeline if running."""
    _permissions.revoke_all()
    ctrl = get_controller()
    if ctrl.is_running:
        ctrl.stop()
    return {"message": "All permissions revoked, pipeline stopped"}


# ─── Web Interface ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve the full dashboard from static/dashboard.html."""
    dashboard_path = os.path.join("static", "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    return HTMLResponse("<h1>Dashboard not found</h1><p>Place dashboard.html in static/</p>")
