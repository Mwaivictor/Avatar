"""
Application entry point.
Starts the FastAPI server with the avatar transformation system.
"""

import logging
import os
import sys

import uvicorn

from config import AppConfig


def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    config = AppConfig()
    setup_logging(config.debug)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  Avatar Transformation System")
    logger.info("=" * 60)
    logger.info("  Web UI:    http://%s:%d", config.host, config.port)
    logger.info("  Services:")
    logger.info("    Face Animation:    %s", config.services.face_animation_url)
    logger.info("    Voice Conversion:  %s", config.services.voice_conversion_url)
    logger.info("    Lip Sync:          %s", config.services.lip_sync_url)
    logger.info("=" * 60)

    os.makedirs("static/avatars", exist_ok=True)

    uvicorn.run(
        "app.api.server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info",
    )


if __name__ == "__main__":
    main()
