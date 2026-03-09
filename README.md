# Avatar Transformation System

> **Disclaimer**: This project is for **educational purposes only** and should not be reused in any production environment, commercial application, identity fraud, impersonation, deepfake generation, or any context that violates applicable laws or the rights of others. The authors assume no liability for misuse.

Real-time avatar and voice transformation platform. Captures your webcam and microphone, runs the feed through AI models (face animation, voice conversion, lip sync), and outputs the result through virtual camera/microphone devices that any video call app can use.

Everything runs locally on your machine — the AI models run in Docker containers on CPU (no GPU needed).

## Architecture

```
 You (webcam + mic)
        │
    ┌───┴───┐
    │ start.py │ ← run this, it does everything
    └───┬───┘
        │
  ┌─────┴─────┐
  │  Dashboard │  http://127.0.0.1:8000
  │  (FastAPI) │  permission gate + controls
  └─────┬─────┘
        │
  ┌─────┴──────────────────────────────────┐
  │        Docker Containers (CPU)          │
  │  ┌──────────────┐ ┌─────────────────┐  │
  │  │ Face Anim.   │ │ Voice Conversion│  │
  │  │ FOMM :8001   │ │ HuBERT+WORLD   │  │
  │  │              │ │ :8002           │  │
  │  └──────────────┘ └─────────────────┘  │
  │         ┌──────────────┐               │
  │         │ Lip Sync     │               │
  │         │ Wav2Lip :8003│               │
  │         └──────────────┘               │
  └────────────────────────────────────────┘
        │
  Virtual Camera + Virtual Mic
        │
  Zoom / Meet / Teams / Discord / WhatsApp / ...
```

## AI Models

| Service | Model | What It Does | Weights |
|---------|-------|-------------|---------|
| Face Animation | [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) | Transfers your head motion to the avatar image | `vox-cpk.pth.tar` (~700 MB) |
| Voice Conversion | HuBERT + WORLD Vocoder | Changes your voice pitch/timbre to a target profile | Auto-downloaded from HuggingFace |
| Lip Sync | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Moves the avatar's lips to match converted audio | `wav2lip_gan.pth` (~400 MB) |

## Prerequisites

You need these installed before running:

| Requirement | What | Install Link |
|---|---|---|
| **Python 3.10+** | Runs the dashboard + pipeline | https://www.python.org/downloads/ |
| **Docker Desktop** | Runs the AI models in containers | https://docs.docker.com/get-docker/ |
| **Virtual Camera** | OBS Virtual Camera (comes with OBS) | https://obsproject.com/ |
| **Virtual Mic** *(optional)* | VB-Audio Virtual Cable (Windows) | https://vb-audio.com/Cable/ |
| **Webcam + Microphone** | Your physical devices | — |

Docker Compose is included with Docker Desktop. That's it — `start.py` handles the rest.

## Quick Start

### 1. Download model checkpoints

Create a `checkpoints/` folder and download the weights:

```
Avatar/
  checkpoints/
    vox-cpk.pth.tar     ← from FOMM repo (https://github.com/AliaksandrSiarohin/first-order-model)
    wav2lip_gan.pth      ← from Wav2Lip repo (https://github.com/Rudrabha/Wav2Lip)
```

> HuBERT weights don't need manual download — they auto-download from HuggingFace on first start.

### 2. Run start.py

```bash
python start.py
```

That's it. `start.py` will:

1. ✓ Check that Python, Docker, and Docker Compose are installed
2. ✓ Create `checkpoints/` and `static/avatars/` directories
3. ✓ Copy `.env.example` → `.env` (if `.env` doesn't exist)
4. ✓ Install Python dependencies (`pip install -r requirements.txt`)
5. ✓ Build and start 3 Docker containers (face, voice, lip sync)
6. ✓ Wait for all services to become healthy
7. ✓ Launch the dashboard at **http://127.0.0.1:8000**

### 3. Use the dashboard

Open **http://127.0.0.1:8000** in your browser.

1. **Grant permissions** — the dashboard auto-detects running video call apps. Select the ones you want and click Grant Access.
2. **Upload an avatar** — any face image (PNG/JPG). This is the face that will appear on camera.
3. **Pick a voice** — select a preset (Male Deep, Female Bright, etc.) or upload a `.wav` sample to create a custom voice profile.
4. **Start Pipeline** — your webcam + mic feed is now transformed in real-time.
5. **In your video call app** — select "OBS Virtual Camera" as your camera and the virtual audio cable as your microphone.

### start.py commands

| Command | What it does |
|---------|-------------|
| `python start.py` | Full startup (install + build + run) |
| `python start.py --build` | Force rebuild Docker images |
| `python start.py --stop` | Stop Docker containers |
| `python start.py --status` | Check if services are healthy |
| `python start.py --skip-docker` | Start dashboard only (containers already running) |
| `python start.py --skip-pip` | Skip Python dependency install |

## Voice System

Two ways to choose a voice:

### Preset profiles (built-in)

Select from the dropdown in the dashboard:

| Profile | Pitch | Character |
|---------|-------|-----------|
| Default | No change | Your natural voice |
| Male Deep | ~95 Hz | Low, resonant |
| Male Medium | ~120 Hz | Natural male range |
| Female Bright | ~220 Hz | Higher, clear |
| Female Warm | ~195 Hz | Mid-high, rounded |

### Upload your own voice

1. Record a `.wav` file (1–10 seconds of clear speech)
2. Enter a profile name in the dashboard (e.g. "narrator")
3. Click **Upload Voice Sample**
4. The system analyzes the pitch, spectral tilt, and formant structure
5. A new voice profile is created and auto-selected

This is voice **conversion** (pitch/timbre shifting), not voice **cloning**. Your words and rhythm stay the same — the tonal characteristics change.

## App Detection

The system scans for running video call apps in three ways:

| Method | How | Example |
|--------|-----|---------|
| **Process scan** | Checks running `.exe` processes via `psutil` | Zoom.exe, Teams.exe, Discord.exe |
| **Browser title (known)** | Reads window titles and matches known URL patterns | meet.google.com, web.whatsapp.com, discord.com |
| **Browser title (generic)** | Catches any browser tab with video-call keywords | "My Company Meeting Room", "Conference Call" |

Works across **all browsers** (Chrome, Firefox, Edge, Brave, Opera, Arc, etc.) — it reads the window title, not the browser binary.

The dashboard and the video call don't need to be in the same browser.

## Permission System

The pipeline will **not start** until you explicitly grant permission:

- Permissions are per-app (Zoom, Meet, Teams, etc.)
- Camera and microphone can be enabled/disabled independently
- Revoke at any time — pipeline stops immediately
- All permissions reset on restart (nothing persisted to disk)

## Project Structure

```
Avatar/
├── start.py                         # ← Run this. Handles everything.
├── main.py                          # FastAPI launcher (called by start.py)
├── config.py                        # Settings from .env
├── docker-compose.yml               # Docker container definitions
├── requirements.txt                 # Python dependencies
├── .env.example                     # Default environment variables
│
├── checkpoints/                     # Model weights (you download these)
│   ├── vox-cpk.pth.tar
│   └── wav2lip_gan.pth
│
├── app/
│   ├── controller.py                # Pipeline orchestrator
│   ├── permissions.py               # Consent gate (per-app, per-device)
│   ├── app_detector.py              # Process + window title scanner
│   ├── capture/
│   │   ├── video_capture.py         # Threaded webcam capture
│   │   └── audio_capture.py         # Threaded mic capture
│   ├── tracking/
│   │   ├── face_tracker.py          # MediaPipe Face Mesh (468 landmarks)
│   │   └── expression_analyzer.py   # Blink, mouth, smile, head pose
│   ├── services/
│   │   ├── base_client.py           # Async HTTP client base
│   │   ├── face_animation_client.py
│   │   ├── voice_conversion_client.py
│   │   └── lip_sync_client.py
│   ├── rendering/
│   │   ├── renderer.py              # Frame compositing
│   │   └── synchronizer.py          # A/V sync buffer
│   ├── output/
│   │   ├── virtual_camera.py        # pyvirtualcam output
│   │   └── virtual_microphone.py    # Virtual audio cable output
│   └── api/
│       └── server.py                # REST API + serves dashboard
│
├── inference_servers/
│   ├── face_animation/              # FOMM Docker service (port 8001)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── server.py
│   │   ├── inference.py
│   │   └── models/fomm.py
│   ├── voice_conversion/            # HuBERT+WORLD Docker service (port 8002)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── server.py
│   │   └── models/pipeline.py
│   └── lip_sync/                    # Wav2Lip Docker service (port 8003)
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── server.py
│       ├── inference.py
│       └── models/
│           ├── wav2lip.py
│           └── audio.py
│
├── static/
│   └── dashboard.html               # Web UI (dark theme, single page)
│
└── tests/
    ├── test_capture.py
    ├── test_rendering.py
    └── test_services.py
```

## Configuration

All settings come from environment variables. `start.py` copies `.env.example` to `.env` on first run.

| Variable | Default | What |
|----------|---------|------|
| `AVATAR_CAMERA_INDEX` | `0` | Webcam device index |
| `AVATAR_FRAME_WIDTH` | `640` | Capture width (px) |
| `AVATAR_FRAME_HEIGHT` | `480` | Capture height (px) |
| `AVATAR_TARGET_FPS` | `30` | Target frame rate |
| `AVATAR_SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |
| `AVATAR_FACE_ANIMATION_URL` | `http://localhost:8001` | Face animation endpoint |
| `AVATAR_VOICE_CONVERSION_URL` | `http://localhost:8002` | Voice conversion endpoint |
| `AVATAR_LIP_SYNC_URL` | `http://localhost:8003` | Lip sync endpoint |
| `AVATAR_SERVICE_TIMEOUT` | `0.5` | Inference call timeout (sec) |
| `AVATAR_DEBUG` | `false` | Debug logging + auto-reload |

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | Pipeline status + metrics |
| `GET` | `/api/services/health` | AI service health |
| `POST` | `/api/pipeline/start` | Start pipeline (needs permission) |
| `POST` | `/api/pipeline/stop` | Stop pipeline |
| `POST` | `/api/avatar/upload` | Upload avatar image |
| `POST` | `/api/voice/speaker?speaker_id=` | Select voice profile |
| `GET` | `/api/voice/speakers` | List all voice profiles |
| `POST` | `/api/voice/upload?speaker_id=` | Upload .wav to create voice profile |
| `GET` | `/api/stream/video` | MJPEG video stream |
| `GET` | `/api/stream/preview` | Single preview frame |
| `GET` | `/api/apps/detect` | Scan for running video apps |
| `GET` | `/api/permissions` | List all permission records |
| `GET` | `/api/permissions/status` | Quick permission summary |
| `POST` | `/api/permissions/grant` | Grant access for an app |
| `POST` | `/api/permissions/revoke?app_id=` | Revoke access for an app |
| `POST` | `/api/permissions/revoke-all` | Revoke all + stop pipeline |

## Performance

All AI inference runs on **CPU** inside Docker. No GPU or CUDA required.

| Stage | How |
|-------|-----|
| Video capture | Threaded OpenCV, ring buffer, zero-copy latest frame |
| Audio capture | Threaded PyAudio, chunk batching |
| Face tracking | MediaPipe Face Mesh (468 landmarks per frame) |
| AI inference | Async HTTP to localhost Docker containers |
| A/V sync | Timestamp-based synchronizer with drift correction |
| Output | pyvirtualcam (30 FPS) + PyAudio stream |

Expect **10–20 FPS** on a modern CPU. The pipeline gracefully holds the previous frame if inference takes too long.

Docker images use CPU-only PyTorch (~800 MB instead of ~2.3 GB for CUDA builds).

## Stopping

```bash
# Stop everything (Ctrl+C stops the dashboard, then):
python start.py --stop

# Or just stop Docker containers:
docker compose down
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Docker daemon is not running` | Open Docker Desktop and wait for it to start |
| Services stuck at "waiting for health" | Check logs: `docker compose logs -f` |
| No virtual camera in Zoom/Meet | Install OBS Studio, open it once to register the virtual camera |
| Audio not transforming | Install VB-Audio Virtual Cable, select it as mic in your call app |
| Checkpoints missing warning | Download model weights into `checkpoints/` (see Quick Start) |
| `pip install` fails on PyAudio | Windows: `pip install pipwin && pipwin install pyaudio` |
| Port 8000 already in use | Set `AVATAR_PORT=9000` in `.env` |

## Technology Stack

| Layer | Technology |
|-------|------------|
| Launcher | Python (`start.py`) |
| Backend | FastAPI, uvicorn |
| Computer Vision | OpenCV, MediaPipe |
| AI Inference | Docker, CPU-only PyTorch, REST APIs |
| Audio | PyAudio, pyworld, HuBERT (HuggingFace) |
| Virtual Devices | pyvirtualcam, virtual audio cable |
| Frontend | Single-page HTML/JS dashboard |
