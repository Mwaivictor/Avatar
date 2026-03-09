"""
Detect running video-conferencing and streaming applications.

IMPORTANT: The avatar system does NOT inject into or connect to any specific
app. It writes to a system-wide virtual camera / virtual microphone device.
ANY application that selects that virtual device as its input will receive
the avatar feed — detection here is purely an informational consent UI.

Detection methods:
  1. Process scan (psutil): matches known executable names for native apps.
  2. Window title scan (Windows ctypes): reads ALL visible window titles
     across every browser (Chrome, Edge, Firefox, Brave, etc.) and matches
     against known AND generic video-call keywords.
  3. Manual add: the user can add any app name from the dashboard, which
     just creates a permission record — no technical link is required.

Works the same regardless of which browser the dashboard is open in.
The dashboard (localhost:8000) and the video call app are completely
independent — they don't need to be in the same browser or even on the
same screen.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


# Process name → app_id mapping (lowercase)
_PROCESS_MAP: Dict[str, str] = {
    "zoom.exe": "zoom",
    "zoom.us": "zoom",
    "zoom": "zoom",
    "teams.exe": "teams",
    "ms-teams.exe": "teams",
    "msteams": "teams",
    "whatsapp.exe": "whatsapp",
    "whatsapp": "whatsapp",
    "discord.exe": "discord",
    "discord": "discord",
    "obs64.exe": "obs",
    "obs32.exe": "obs",
    "obs": "obs",
    "skype.exe": "skype",
    "skypeforlinux": "skype",
    "ciscocollabhost.exe": "webex",
    "webex": "webex",
    "slack.exe": "slack",
    "slack": "slack",
}

# ── Known browser title patterns (specific sites) ─────────────────
_BROWSER_TITLE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "google_meet": [
        re.compile(r"meet\.google\.com", re.IGNORECASE),
        re.compile(r"Google Meet", re.IGNORECASE),
    ],
    "teams": [
        re.compile(r"Microsoft Teams", re.IGNORECASE),
        re.compile(r"teams\.microsoft\.com", re.IGNORECASE),
        re.compile(r"teams\.live\.com", re.IGNORECASE),
    ],
    "zoom": [
        re.compile(r"Zoom Meeting", re.IGNORECASE),
        re.compile(r"zoom\.us/j/", re.IGNORECASE),
    ],
    "whatsapp_web": [
        re.compile(r"WhatsApp", re.IGNORECASE),
        re.compile(r"web\.whatsapp\.com", re.IGNORECASE),
    ],
    "discord_web": [
        re.compile(r"discord\.com/(channels|app)", re.IGNORECASE),
        re.compile(r"Discord \|", re.IGNORECASE),
    ],
    "slack_web": [
        re.compile(r"app\.slack\.com", re.IGNORECASE),
        re.compile(r"Slack \|", re.IGNORECASE),
    ],
    "webex_web": [
        re.compile(r"webex\.com", re.IGNORECASE),
    ],
    "facetime_web": [
        re.compile(r"facetime\.apple\.com", re.IGNORECASE),
    ],
    "jitsi": [
        re.compile(r"meet\.jit\.si", re.IGNORECASE),
        re.compile(r"Jitsi Meet", re.IGNORECASE),
    ],
    "whereby": [
        re.compile(r"whereby\.com", re.IGNORECASE),
    ],
    "around": [
        re.compile(r"around\.co", re.IGNORECASE),
    ],
    "streamyard": [
        re.compile(r"streamyard\.com", re.IGNORECASE),
    ],
}

# ── Generic keywords that suggest ANY video/voice call in a browser ──
# If a window title matches these but didn't match a known app above,
# it's flagged as an "unknown_video_call" so the user can grant permission.
_GENERIC_VIDEO_CALL_KEYWORDS = re.compile(
    r"video\s*call|voice\s*call|meeting|conference|webinar|live\s*stream"
    r"|camera|microphone|screen\s*share|join .* call|calling",
    re.IGNORECASE,
)

# Display info for browser-detected apps not in KNOWN_APPS
_BROWSER_APP_META: Dict[str, Dict] = {
    "whatsapp_web": {"display_name": "WhatsApp Web", "icon": "💬"},
    "discord_web": {"display_name": "Discord (Browser)", "icon": "🎮"},
    "slack_web": {"display_name": "Slack (Browser)", "icon": "💼"},
    "webex_web": {"display_name": "Webex (Browser)", "icon": "🌐"},
    "facetime_web": {"display_name": "FaceTime Link", "icon": "📱"},
    "jitsi": {"display_name": "Jitsi Meet", "icon": "📞"},
    "whereby": {"display_name": "Whereby", "icon": "🔵"},
    "around": {"display_name": "Around", "icon": "⭕"},
    "streamyard": {"display_name": "StreamYard", "icon": "🎬"},
    "unknown_video_call": {"display_name": "Video Call (Browser)", "icon": "🌍"},
}


def detect_running_apps() -> List[Dict]:
    """
    Scan running processes and window titles for video call apps.

    Returns detected apps from three sources:
      1. Native processes (Zoom.exe, Teams.exe, etc.)
      2. Known browser-based apps (meet.google.com, web.whatsapp.com, etc.)
      3. Unknown browser pages with generic video-call keywords

    Returns:
        List of dicts with keys: app_id, display_name, icon, pid, process_name, source
    """
    detected = {}

    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed — app detection unavailable")
        return []

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = proc.info["name"].lower()
            app_id = _PROCESS_MAP.get(pname)
            if app_id and app_id not in detected:
                from app.permissions import KNOWN_APPS
                app_info = KNOWN_APPS.get(app_id, {})
                detected[app_id] = {
                    "app_id": app_id,
                    "display_name": app_info.get("display_name", app_id),
                    "icon": app_info.get("icon", "📱"),
                    "pid": proc.info["pid"],
                    "process_name": proc.info["name"],
                    "source": "process",
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Try to detect browser-based apps via window titles (Windows only)
    try:
        _detect_browser_apps(detected)
    except Exception:
        pass  # Non-critical — skip silently on non-Windows or if ctypes fails

    return list(detected.values())


def _get_all_window_titles() -> List[str]:
    """Read all visible window titles on Windows using EnumWindows."""
    import ctypes
    import ctypes.wintypes

    EnumWindows = ctypes.windll.user32.EnumWindows
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )

    titles = []

    def enum_callback(hwnd, _lparam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buf, length + 1)
                titles.append(buf.value)
        return True

    EnumWindows(WNDENUMPROC(enum_callback), 0)
    return titles


def _detect_browser_apps(detected: dict) -> None:
    """
    Detect browser-based video call apps via window titles.

    Scans ALL open windows across ALL browsers (Chrome, Edge, Firefox,
    Brave, Opera, Arc, etc.) — the browser type doesn't matter because
    we're reading the window title which typically contains the page title
    and site URL.

    Two-pass approach:
      Pass 1: Match known site patterns (meet.google.com, etc.)
      Pass 2: Match generic video-call keywords for unknown sites
    """
    titles = _get_all_window_titles()

    from app.permissions import KNOWN_APPS

    # Pass 1: Known browser-based apps
    for title in titles:
        for app_id, patterns in _BROWSER_TITLE_PATTERNS.items():
            if app_id in detected:
                continue
            for pattern in patterns:
                if pattern.search(title):
                    # Check KNOWN_APPS first, then _BROWSER_APP_META
                    app_info = KNOWN_APPS.get(app_id, _BROWSER_APP_META.get(app_id, {}))
                    detected[app_id] = {
                        "app_id": app_id,
                        "display_name": app_info.get("display_name", app_id),
                        "icon": app_info.get("icon", "🌍"),
                        "pid": None,
                        "process_name": f"Browser ({title[:50]})",
                        "source": "browser_title",
                    }
                    break

    # Pass 2: Generic keyword scan for unrecognized video call sites
    # Skip if we already found a generic match, and skip titles that
    # belong to our own dashboard or to already-detected apps.
    if "unknown_video_call" not in detected:
        own_titles = {"Avatar", "localhost:8000", "127.0.0.1:8000"}
        for title in titles:
            # Skip our own dashboard
            if any(own in title for own in own_titles):
                continue
            # Skip titles already matched to a known app
            already_matched = any(
                title[:50] in d.get("process_name", "")
                for d in detected.values()
            )
            if already_matched:
                continue
            # Check for generic video call keywords
            if _GENERIC_VIDEO_CALL_KEYWORDS.search(title):
                meta = _BROWSER_APP_META["unknown_video_call"]
                detected["unknown_video_call"] = {
                    "app_id": "unknown_video_call",
                    "display_name": f"{meta['display_name']}: {title[:40]}",
                    "icon": meta["icon"],
                    "pid": None,
                    "process_name": f"Browser ({title[:50]})",
                    "source": "browser_keyword",
                }
                break  # Only report the first unknown match
