"""
start.py — One-command launcher for the Avatar Transformation System.

Handles EVERYTHING from a clean checkout:
  1. Checks prerequisites (Python, Docker, Docker Compose)
  2. Creates the checkpoints/ directory
  3. Installs Python dependencies (pip install -r requirements.txt)
  4. Copies .env.example → .env if .env doesn't exist
  5. Builds and starts Docker containers (AI inference servers)
  6. Waits for all three inference services to become healthy
  7. Launches the main FastAPI application (dashboard + pipeline)

Usage:
  First run:   python start.py
  Later runs:  python start.py          (skips pip install if deps exist)
  Rebuild:     python start.py --build  (forces Docker image rebuild)
  Stop:        python start.py --stop   (stops Docker containers)
  Status:      python start.py --status (shows container health)
"""

import argparse
import http.client
import os
import platform
import shutil
import subprocess
import sys
import time

# ── Constants ────────────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")
ENV_FILE = os.path.join(ROOT_DIR, ".env")
ENV_EXAMPLE = os.path.join(ROOT_DIR, ".env.example")
REQUIREMENTS = os.path.join(ROOT_DIR, "requirements.txt")
STATIC_AVATARS = os.path.join(ROOT_DIR, "static", "avatars")

SERVICES = {
    "face-animation":   {"port": 8001, "health": "/health", "checkpoint": "vox-cpk.pth.tar"},
    "voice-conversion": {"port": 8002, "health": "/health", "checkpoint": None},
    "lip-sync":         {"port": 8003, "health": "/health", "checkpoint": "wav2lip_gan.pth"},
}

HEALTH_TIMEOUT = 120          # Max seconds to wait for Docker services
HEALTH_POLL_INTERVAL = 3      # Seconds between health checks
DASHBOARD_PORT = 8000


# ── Helpers ──────────────────────────────────────────────────────

def banner(msg: str) -> None:
    width = max(len(msg) + 4, 60)
    print(f"\n{'=' * width}")
    print(f"  {msg}")
    print(f"{'=' * width}\n")


def info(msg: str) -> None:
    print(f"  [✓] {msg}")


def warn(msg: str) -> None:
    print(f"  [!] {msg}")


def fail(msg: str) -> None:
    print(f"\n  [✗] {msg}\n")
    sys.exit(1)


def run(cmd: list[str], cwd: str = ROOT_DIR, check: bool = True, capture: bool = False):
    """Run a subprocess; fail on error unless check=False."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check,
            capture_output=capture, text=True,
        )
        return result
    except FileNotFoundError:
        fail(f"Command not found: {cmd[0]}")
    except subprocess.CalledProcessError as e:
        if capture:
            print(e.stdout or "")
            print(e.stderr or "")
        fail(f"Command failed: {' '.join(cmd)}")


def check_port(port: int) -> bool:
    """Return True if a service responds on localhost:port."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


# ── Step functions ───────────────────────────────────────────────

DOCKER_DESKTOP_PATHS = [
    os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Docker", "Docker", "Docker Desktop.exe"),
    os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Docker", "Docker", "Docker Desktop.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Docker", "Docker Desktop.exe"),
]
DOCKER_STARTUP_TIMEOUT = 90   # Max seconds to wait for Docker daemon after launching


def _start_docker_desktop() -> None:
    """Attempt to launch Docker Desktop and wait for the daemon to respond."""
    warn("Docker daemon is not running — attempting to start Docker Desktop...")

    # Find the Docker Desktop executable
    exe_path = None
    for path in DOCKER_DESKTOP_PATHS:
        if os.path.isfile(path):
            exe_path = path
            break

    if exe_path is None:
        # Try the Windows registry as a fallback
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Docker Inc.\Docker Desktop",
            )
            install_path, _ = winreg.QueryValueEx(key, "InstallPath")
            winreg.CloseKey(key)
            candidate = os.path.join(install_path, "Docker Desktop.exe")
            if os.path.isfile(candidate):
                exe_path = candidate
        except Exception:
            pass

    if exe_path is None:
        fail(
            "Could not find Docker Desktop executable.\n"
            "         Please start Docker Desktop manually, then run start.py again."
        )

    info(f"Launching: {exe_path}")
    # Start Docker Desktop as a detached process (non-blocking)
    subprocess.Popen(
        [exe_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
    )

    # Poll until the daemon responds
    start_time = time.time()
    while (time.time() - start_time) < DOCKER_STARTUP_TIMEOUT:
        elapsed = int(time.time() - start_time)
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            info(f"Docker daemon is running (started in {elapsed}s)")
            return
        print(f"    ... waiting for Docker daemon ({elapsed}s / {DOCKER_STARTUP_TIMEOUT}s)")
        time.sleep(3)

    fail(
        f"Docker daemon did not start within {DOCKER_STARTUP_TIMEOUT}s.\n"
        "         Open Docker Desktop manually and try again."
    )

def check_prerequisites() -> dict:
    """Verify Python, Docker, and Docker Compose are available."""
    banner("Checking prerequisites")
    results = {}

    # Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ required (found {py_ver})")
    info(f"Python {py_ver}")
    results["python"] = py_ver

    # Docker
    docker = shutil.which("docker")
    if not docker:
        fail("Docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/")
    r = run(["docker", "--version"], capture=True)
    info(f"Docker: {r.stdout.strip()}")
    results["docker"] = r.stdout.strip()

    # Docker daemon running? If not, start Docker Desktop automatically.
    r = run(["docker", "info"], capture=True, check=False)
    if r.returncode != 0:
        _start_docker_desktop()
    else:
        info("Docker daemon is running")

    # Docker Compose (v2 plugin or standalone)
    compose_cmd = _find_compose_cmd()
    if compose_cmd is None:
        fail("Docker Compose not found. Install: https://docs.docker.com/compose/install/")
    r = run([*compose_cmd, "version"], capture=True)
    info(f"Compose: {r.stdout.strip()}")
    results["compose_cmd"] = compose_cmd

    return results


def _find_compose_cmd() -> list[str] | None:
    """Find 'docker compose' (v2 plugin) or 'docker-compose' (standalone)."""
    # Try v2 plugin first
    try:
        r = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True, text=True, check=True,
        )
        return ["docker", "compose"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    # Try standalone
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return None


def create_directories() -> None:
    """Create required directories if they don't exist."""
    banner("Creating directories")
    for d in [CHECKPOINTS_DIR, STATIC_AVATARS]:
        os.makedirs(d, exist_ok=True)
        info(f"{os.path.relpath(d, ROOT_DIR)}/")

    # Check for model checkpoints
    missing_ckpts = []
    for svc_name, svc_info in SERVICES.items():
        ckpt = svc_info["checkpoint"]
        if ckpt and not os.path.exists(os.path.join(CHECKPOINTS_DIR, ckpt)):
            missing_ckpts.append(ckpt)
    if missing_ckpts:
        warn("Missing model checkpoints (services will start but use random weights):")
        for c in missing_ckpts:
            print(f"         checkpoints/{c}")
        print()
        print("    Download them and place in the checkpoints/ folder:")
        print("      vox-cpk.pth.tar  → https://github.com/AliaksandrSiarohin/first-order-model")
        print("      wav2lip_gan.pth  → https://github.com/Rudrabha/Wav2Lip")
        print()


def setup_env_file() -> None:
    """Copy .env.example → .env if .env doesn't already exist."""
    if not os.path.exists(ENV_FILE):
        if os.path.exists(ENV_EXAMPLE):
            shutil.copy2(ENV_EXAMPLE, ENV_FILE)
            info("Created .env from .env.example")
        else:
            warn(".env.example not found — using built-in defaults")
    else:
        info(".env already exists — keeping existing settings")


def install_python_deps() -> None:
    """Install Python dependencies via pip."""
    banner("Installing Python dependencies")

    # Quick check: if fastapi is already importable, skip
    try:
        import fastapi  # noqa: F401
        import cv2      # noqa: F401
        import mediapipe # noqa: F401
        info("Core dependencies already installed — skipping pip install")
        return
    except ImportError:
        pass

    info("Running: pip install -r requirements.txt")
    run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS])
    info("Python dependencies installed")


def docker_build_and_start(compose_cmd: list[str], force_build: bool = False) -> None:
    """Build Docker images and start inference containers."""
    banner("Starting AI inference services (Docker)")

    cmd = [*compose_cmd, "up", "-d"]
    if force_build:
        cmd.append("--build")
        info("Forcing Docker image rebuild")

    info(f"Running: {' '.join(cmd)}")
    run(cmd)
    info("Docker containers started")


def wait_for_services() -> None:
    """Poll health endpoints until all services are up or timeout."""
    banner("Waiting for AI services to become healthy")

    start = time.time()
    pending = {name: info for name, info in SERVICES.items()}

    while pending and (time.time() - start) < HEALTH_TIMEOUT:
        for name in list(pending.keys()):
            port = pending[name]["port"]
            if check_port(port):
                info(f"{name} (port {port}) — healthy")
                del pending[name]
        if pending:
            remaining = list(pending.keys())
            elapsed = int(time.time() - start)
            print(f"    ... waiting for {', '.join(remaining)} ({elapsed}s / {HEALTH_TIMEOUT}s)")
            time.sleep(HEALTH_POLL_INTERVAL)

    if pending:
        warn("These services did not become healthy in time:")
        for name, svc in pending.items():
            print(f"         {name} (port {svc['port']})")
        print()
        warn("The dashboard will still start — you can check Docker logs with:")
        print("         docker compose logs -f")
        print()
    else:
        info("All AI services are healthy!")


def start_dashboard() -> None:
    """Launch the main FastAPI application."""
    banner("Starting Avatar Dashboard")
    info(f"Dashboard: http://127.0.0.1:{DASHBOARD_PORT}")
    info("Press Ctrl+C to stop\n")

    # Import and run main — this blocks until Ctrl+C
    try:
        # Add root to path so imports work
        sys.path.insert(0, ROOT_DIR)
        os.chdir(ROOT_DIR)
        from main import main
        main()
    except KeyboardInterrupt:
        print("\n")
        info("Dashboard stopped")


def docker_stop(compose_cmd: list[str]) -> None:
    """Stop all Docker containers."""
    banner("Stopping AI inference services")
    run([*compose_cmd, "down"])
    info("All containers stopped")


def docker_status(compose_cmd: list[str]) -> None:
    """Show status of Docker containers."""
    banner("Docker container status")
    run([*compose_cmd, "ps"])
    print()
    for name, svc in SERVICES.items():
        healthy = check_port(svc["port"])
        status = "healthy" if healthy else "not responding"
        symbol = "✓" if healthy else "✗"
        print(f"  [{symbol}] {name} (port {svc['port']}) — {status}")
    print()


# ── Main ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python start.py",
        description="""
============================================================
  Avatar Transformation System — Launcher
============================================================

  One command to set up and run the entire system.
  Installs dependencies, builds Docker containers for the
  AI models, and launches the web dashboard.

  Everything runs locally on CPU — no GPU required.
  Educational use only.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
─── Commands ───────────────────────────────────────────────

  python start.py                Full startup (first run)
                                   • Checks Python, Docker, Compose
                                   • Creates directories
                                   • Copies .env.example → .env
                                   • pip install -r requirements.txt
                                   • docker compose up -d --build
                                   • Waits for services to be healthy
                                   • Opens dashboard at localhost:8000

  python start.py --build        Force rebuild Docker images
                                   (use after editing Dockerfiles
                                    or inference server code)

  python start.py --stop         Stop all Docker containers
                                   (docker compose down)

  python start.py --status       Show container health status
                                   (checks ports 8001, 8002, 8003)

  python start.py --skip-docker  Start dashboard only
                                   (assumes Docker containers are
                                    already running from a previous
                                    session)

  python start.py --skip-pip     Skip pip install step
                                   (use when deps are already
                                    installed and you want a
                                    faster startup)

─── Services ───────────────────────────────────────────────

  Port 8000  Dashboard + API     (Python / FastAPI)
  Port 8001  Face Animation      (FOMM — Docker)
  Port 8002  Voice Conversion    (HuBERT + WORLD — Docker)
  Port 8003  Lip Sync            (Wav2Lip — Docker)

─── Prerequisites ──────────────────────────────────────────

  • Python 3.10+          python.org/downloads
  • Docker Desktop        docs.docker.com/get-docker
  • OBS Studio            obsproject.com (virtual camera)
  • VB-Audio Cable        vb-audio.com/Cable (virtual mic)

─── Model Checkpoints ─────────────────────────────────────

  Place in checkpoints/ before first run:
    checkpoints/vox-cpk.pth.tar   (FOMM face animation)
    checkpoints/wav2lip_gan.pth   (Wav2Lip lip sync)
  HuBERT auto-downloads from HuggingFace on first start.
""",
    )
    p.add_argument(
        "--build", action="store_true",
        help="Force rebuild of Docker images before starting",
    )
    p.add_argument(
        "--stop", action="store_true",
        help="Stop all Docker containers and exit",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Show Docker container health and exit",
    )
    p.add_argument(
        "--skip-docker", action="store_true",
        help="Skip Docker build/start (containers already running)",
    )
    p.add_argument(
        "--skip-pip", action="store_true",
        help="Skip Python dependency installation",
    )
    return p.parse_args()


def main_entry() -> None:
    args = parse_args()

    banner("Avatar Transformation System")
    print("  Educational use only — not for production or impersonation.\n")

    # Prerequisites (always needed)
    prereqs = check_prerequisites()
    compose_cmd = prereqs["compose_cmd"]

    # --stop
    if args.stop:
        docker_stop(compose_cmd)
        return

    # --status
    if args.status:
        docker_status(compose_cmd)
        return

    # Full startup flow
    create_directories()
    setup_env_file()

    if not args.skip_pip:
        install_python_deps()

    if not args.skip_docker:
        docker_build_and_start(compose_cmd, force_build=args.build)
        wait_for_services()
    else:
        info("Skipping Docker — assuming containers are already running")

    start_dashboard()


if __name__ == "__main__":
    main_entry()
