"""
Permission and consent management for the avatar transformation system.

Before the virtual camera/microphone can override the real devices in any
target application (Zoom, Google Meet, Teams, WhatsApp, etc.), the user
MUST grant explicit consent through the web UI.

Permissions are stored in-memory per session and are never persisted to disk.
Every restart requires fresh consent.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PermissionStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"


@dataclass
class PermissionRecord:
    """A single permission grant for a target application."""
    app_name: str
    status: PermissionStatus = PermissionStatus.NOT_REQUESTED
    virtual_camera: bool = False
    virtual_microphone: bool = False
    granted_at: Optional[float] = None
    revoked_at: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "app_name": self.app_name,
            "status": self.status.value,
            "virtual_camera": self.virtual_camera,
            "virtual_microphone": self.virtual_microphone,
            "granted_at": self.granted_at,
            "revoked_at": self.revoked_at,
            "reason": self.reason,
        }


# Well-known video call / streaming applications
KNOWN_APPS = {
    "zoom": {
        "display_name": "Zoom",
        "process_names": ["Zoom.exe", "zoom.us", "zoom"],
        "icon": "📹",
    },
    "google_meet": {
        "display_name": "Google Meet",
        "process_names": [],  # Browser-based — detected via window title
        "browser_titles": ["meet.google.com"],
        "icon": "🟢",
    },
    "teams": {
        "display_name": "Microsoft Teams",
        "process_names": ["Teams.exe", "ms-teams.exe", "msteams"],
        "icon": "🟣",
    },
    "whatsapp": {
        "display_name": "WhatsApp",
        "process_names": ["WhatsApp.exe", "whatsapp"],
        "icon": "💬",
    },
    "discord": {
        "display_name": "Discord",
        "process_names": ["Discord.exe", "discord"],
        "icon": "🎮",
    },
    "obs": {
        "display_name": "OBS Studio",
        "process_names": ["obs64.exe", "obs32.exe", "obs"],
        "icon": "🎥",
    },
    "skype": {
        "display_name": "Skype",
        "process_names": ["Skype.exe", "skypeforlinux"],
        "icon": "🔵",
    },
    "webex": {
        "display_name": "Cisco Webex",
        "process_names": ["CiscoCollabHost.exe", "webex"],
        "icon": "🌐",
    },
    "slack": {
        "display_name": "Slack",
        "process_names": ["slack.exe", "slack"],
        "icon": "💼",
    },
    "whatsapp_web": {
        "display_name": "WhatsApp Web",
        "process_names": [],
        "icon": "💬",
    },
    "discord_web": {
        "display_name": "Discord (Web)",
        "process_names": [],
        "icon": "🎮",
    },
    "slack_web": {
        "display_name": "Slack (Web)",
        "process_names": [],
        "icon": "💼",
    },
    "webex_web": {
        "display_name": "Webex (Web)",
        "process_names": [],
        "icon": "🌐",
    },
    "facetime": {
        "display_name": "FaceTime",
        "process_names": ["FaceTime"],
        "icon": "📗",
    },
    "jitsi": {
        "display_name": "Jitsi Meet",
        "process_names": [],
        "icon": "🟧",
    },
    "whereby": {
        "display_name": "Whereby",
        "process_names": [],
        "icon": "🏠",
    },
    "around": {
        "display_name": "Around",
        "process_names": [],
        "icon": "🔵",
    },
    "streamyard": {
        "display_name": "StreamYard",
        "process_names": [],
        "icon": "🎬",
    },
    "unknown_video_call": {
        "display_name": "Unknown Video Call",
        "process_names": [],
        "icon": "🌍",
    },
    "other": {
        "display_name": "Other Application",
        "process_names": [],
        "icon": "📱",
    },
}


class PermissionManager:
    """
    Manages user consent for virtual device usage per target application.

    Flow:
      1. User sees detected running apps in the UI
      2. User clicks "Allow" on a specific app → permission = GRANTED
      3. Pipeline can only feed virtual cam/mic when at least one permission is GRANTED
      4. User can revoke at any time → pipeline stops feeding that app
    """

    def __init__(self):
        self._permissions: Dict[str, PermissionRecord] = {}
        self._lock = threading.Lock()
        self._global_enabled = False

    def request_permission(
        self,
        app_id: str,
        virtual_camera: bool = True,
        virtual_microphone: bool = True,
        reason: str = "",
    ) -> PermissionRecord:
        """Create a pending permission request for a target app."""
        with self._lock:
            app_info = KNOWN_APPS.get(app_id, KNOWN_APPS["other"])
            record = PermissionRecord(
                app_name=app_info["display_name"],
                status=PermissionStatus.PENDING,
                virtual_camera=virtual_camera,
                virtual_microphone=virtual_microphone,
                reason=reason or f"Avatar overlay for {app_info['display_name']}",
            )
            self._permissions[app_id] = record
            logger.info("Permission requested for %s", app_info["display_name"])
            return record

    def grant_permission(
        self,
        app_id: str,
        virtual_camera: bool = True,
        virtual_microphone: bool = True,
    ) -> PermissionRecord:
        """Grant consent for virtual devices on a specific app."""
        with self._lock:
            app_info = KNOWN_APPS.get(app_id, KNOWN_APPS["other"])
            record = self._permissions.get(app_id)
            if record is None:
                record = PermissionRecord(app_name=app_info["display_name"])
                self._permissions[app_id] = record

            record.status = PermissionStatus.GRANTED
            record.virtual_camera = virtual_camera
            record.virtual_microphone = virtual_microphone
            record.granted_at = time.time()
            record.revoked_at = None

            self._global_enabled = True
            logger.info(
                "Permission GRANTED for %s (camera=%s, mic=%s)",
                record.app_name, virtual_camera, virtual_microphone,
            )
            return record

    def revoke_permission(self, app_id: str) -> Optional[PermissionRecord]:
        """Revoke a previously granted permission."""
        with self._lock:
            record = self._permissions.get(app_id)
            if record is None:
                return None

            record.status = PermissionStatus.REVOKED
            record.revoked_at = time.time()

            # Check if any permissions are still active
            self._global_enabled = any(
                r.status == PermissionStatus.GRANTED
                for r in self._permissions.values()
            )

            logger.info("Permission REVOKED for %s", record.app_name)
            return record

    def revoke_all(self) -> None:
        """Revoke all granted permissions."""
        with self._lock:
            now = time.time()
            for record in self._permissions.values():
                if record.status == PermissionStatus.GRANTED:
                    record.status = PermissionStatus.REVOKED
                    record.revoked_at = now
            self._global_enabled = False
            logger.info("All permissions revoked")

    def is_camera_allowed(self) -> bool:
        """Check if virtual camera output is allowed for any app."""
        with self._lock:
            return any(
                r.status == PermissionStatus.GRANTED and r.virtual_camera
                for r in self._permissions.values()
            )

    def is_microphone_allowed(self) -> bool:
        """Check if virtual microphone output is allowed for any app."""
        with self._lock:
            return any(
                r.status == PermissionStatus.GRANTED and r.virtual_microphone
                for r in self._permissions.values()
            )

    @property
    def any_granted(self) -> bool:
        return self._global_enabled

    def get_permission(self, app_id: str) -> Optional[PermissionRecord]:
        with self._lock:
            return self._permissions.get(app_id)

    def get_all_permissions(self) -> List[dict]:
        with self._lock:
            return [
                {"app_id": app_id, **record.to_dict()}
                for app_id, record in self._permissions.items()
            ]

    def get_status_summary(self) -> dict:
        with self._lock:
            granted_apps = [
                r.app_name
                for r in self._permissions.values()
                if r.status == PermissionStatus.GRANTED
            ]
            return {
                "any_granted": self._global_enabled,
                "camera_allowed": self.is_camera_allowed(),
                "microphone_allowed": self.is_microphone_allowed(),
                "granted_apps": granted_apps,
                "total_permissions": len(self._permissions),
            }
