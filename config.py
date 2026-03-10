"""
Central configuration for the Avatar Transformation System.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field


@dataclass
class VideoConfig:
    camera_index: int = int(os.getenv("AVATAR_CAMERA_INDEX", "0"))
    frame_width: int = int(os.getenv("AVATAR_FRAME_WIDTH", "640"))
    frame_height: int = int(os.getenv("AVATAR_FRAME_HEIGHT", "480"))
    target_fps: int = int(os.getenv("AVATAR_TARGET_FPS", "30"))


@dataclass
class AudioConfig:
    sample_rate: int = int(os.getenv("AVATAR_SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("AVATAR_AUDIO_CHANNELS", "1"))
    chunk_size: int = int(os.getenv("AVATAR_AUDIO_CHUNK", "1024"))
    format_width: int = 2  # 16-bit audio


@dataclass
class ServiceEndpoints:
    face_animation_url: str = os.getenv(
        "AVATAR_FACE_ANIMATION_URL", "http://localhost:8001"
    )
    voice_conversion_url: str = os.getenv(
        "AVATAR_VOICE_CONVERSION_URL", "http://localhost:8002"
    )
    lip_sync_url: str = os.getenv(
        "AVATAR_LIP_SYNC_URL", "http://localhost:8003"
    )
    request_timeout: float = float(os.getenv("AVATAR_SERVICE_TIMEOUT", "10"))


@dataclass
class RenderingConfig:
    output_width: int = int(os.getenv("AVATAR_OUTPUT_WIDTH", "640"))
    output_height: int = int(os.getenv("AVATAR_OUTPUT_HEIGHT", "480"))
    output_fps: int = int(os.getenv("AVATAR_OUTPUT_FPS", "30"))
    video_codec: str = "H264"
    audio_codec: str = "PCM"


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    services: ServiceEndpoints = field(default_factory=ServiceEndpoints)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    avatar_image_path: str = os.getenv("AVATAR_IMAGE_PATH", "static/avatars/default.png")
    virtual_camera_name: str = os.getenv("AVATAR_VIRTUAL_CAMERA", "")
    virtual_mic_name: str = os.getenv("AVATAR_VIRTUAL_MIC", "")
    debug: bool = os.getenv("AVATAR_DEBUG", "false").lower() == "true"
    host: str = os.getenv("AVATAR_HOST", "127.0.0.1")
    port: int = int(os.getenv("AVATAR_PORT", "8000"))
