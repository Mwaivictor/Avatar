<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-28%2B-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Status">
</p>

# Avatar — Real-Time Avatar & Voice Transformation

Transform your appearance and voice in real time for video calls. Avatar replaces your webcam feed with an AI-generated avatar that mirrors your facial expressions and converts your voice — all routed through virtual devices that work with **any** video call app (Google Meet, Zoom, Teams, Discord, etc.).

> **[!] Educational & Research Project** — This system is built for learning about real-time AI pipelines, computer vision, and audio processing. Use responsibly and ethically.

---

## Features

- **Real-time face animation** — Your expressions drive an avatar face using First Order Motion Model (FOMM)
- **Voice conversion** — Transform your voice in real time using HuBERT + WORLD vocoder
- **Lip sync** — Keep avatar mouth movements matched to your speech with Wav2Lip
- **Audio-only mode** — Voice-only transformation for phone/voice calls without the camera pipeline
- **Virtual camera & microphone** — Output appears as standard system devices (OBS Virtual Camera + VB-Audio)
- **Works with any app** — Google Meet, Zoom, Teams, Discord, WhatsApp — anything that uses a camera/mic
- **Web dashboard** — Simple browser UI to control everything, preview feeds, and upload avatars
- **Persistent voice profiles** — Upload a `.wav` sample to clone a voice; profiles are saved to disk and reloaded across restarts

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────┐     ┌───────────────┐
│   Webcam +   │────▶│              Avatar System               │────▶│ OBS Virtual   │
│   Microphone │     │                                          │     │ Camera        │
└──────────────┘     │  ┌────────┐  ┌────────┐  ┌──────────┐  │     ├───────────────┤
                     │  │  Face  │  │ Voice  │  │   Lip    │  │     │ VB-Audio      │
                     │  │Tracker │  │Convert │  │   Sync   │  │────▶│ Virtual Cable │
                     │  └───┬────┘  └───┬────┘  └────┬─────┘  │     └───────┬───────┘
                     │      │           │            │         │             │
                     │      ▼           ▼            ▼         │             ▼
                     │  ┌────────────────────────────────────┐ │     ┌───────────────┐
                     │  │     AI Inference (Docker)          │ │     │  Google Meet   │
                     │  │  :8001 FOMM  :8002 VC  :8003 W2L  │ │     │  Zoom / Teams  │
                     │  └────────────────────────────────────┘ │     │  Discord / etc │
                     └──────────────────────────────────────────┘     └───────────────┘
```

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.11 or higher |
| **Docker Desktop** | For running AI inference servers |
| **OBS Studio** | Provides OBS Virtual Camera driver |
| **VB-Audio Virtual Cable** | Provides virtual microphone driver |
| **GPU (optional)** | NVIDIA GPU with CUDA speeds up inference significantly |

### Install Virtual Device Drivers

**Windows** (run as Administrator):
```powershell
cd C:\Users\Admin\Desktop\Avatar
powershell -ExecutionPolicy Bypass -File install_virtual_devices.ps1
```

Or install manually:
- **OBS Studio**: https://obsproject.com/download
- **VB-Audio Virtual Cable**: https://vb-audio.com/Cable/

> Restart your computer after installing the drivers.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Mwaivictor/Avatar.git
cd Avatar

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start everything (Docker containers + web server)
python start.py

# 4. Open the dashboard
#    http://127.0.0.1:8000
```

`start.py` will:
1. Build and start the 3 Docker inference containers
2. Wait for health checks to pass
3. Launch the FastAPI web server on port 8000

### Using with Video Calls

1. Select mode: **Video + Audio** for video calls, or **Audio Only** for voice calls
2. Click **Start Avatar** in the dashboard
3. In your video call app (Meet, Zoom, Teams), go to **Settings**:
   - **Camera** → select **OBS Virtual Camera** (full mode only)
   - **Microphone** → select **CABLE Output (VB-Audio Virtual Cable)**
4. Others will see your avatar and hear your converted voice

---

## Dashboard

The web dashboard at `http://127.0.0.1:8000` provides:

- **Mode selection** — Choose between **Video + Audio** (full avatar) or **Audio Only** (voice calls)
- **Live preview** — See your webcam input and avatar output side by side (full mode)
- **Audio visualizer** — Animated waveform display when running in audio-only mode
- **One-click start/stop** — No complex setup, just click Start
- **Avatar upload** — Upload any face image as your avatar
- **Voice selection** — Choose from built-in voice profiles or upload custom `.wav` samples
- **Persistent voice profiles** — Uploaded voice profiles are saved to disk and available across restarts
- **Service health** — Monitor the 3 AI inference containers
- **Performance stats** — FPS, A/V drift, frame/audio counters

---

## Project Structure

```
avatar/
├── start.py                    # Launcher — builds Docker, starts server
├── main.py                     # Uvicorn entry point
├── config.py                   # Configuration (env vars / dataclasses)
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # AI inference containers
├── install_virtual_devices.ps1 # Windows driver installer
│
├── app/
│   ├── controller.py           # Pipeline orchestrator (full + audio-only modes)
│   ├── api/
│   │   └── server.py           # FastAPI REST API + web UI
│   ├── capture/
│   │   ├── video_capture.py    # Webcam capture (OpenCV)
│   │   └── audio_capture.py    # Microphone capture (sounddevice)
│   ├── tracking/
│   │   ├── face_tracker.py     # MediaPipe face landmark detection
│   │   └── expression_analyzer.py
│   ├── services/
│   │   ├── base_client.py      # Async HTTP client base
│   │   ├── face_animation_client.py
│   │   ├── voice_conversion_client.py
│   │   └── lip_sync_client.py
│   ├── rendering/
│   │   ├── renderer.py         # Frame compositing
│   │   └── synchronizer.py     # A/V sync
│   └── output/
│       ├── virtual_camera.py   # OBS Virtual Camera output
│       └── virtual_microphone.py # VB-Audio output
│
├── inference_servers/
│   ├── face_animation/         # FOMM — First Order Motion Model (:8001)
│   ├── voice_conversion/       # HuBERT + WORLD vocoder (:8002)
│   └── lip_sync/               # Wav2Lip (:8003)
│
├── static/
│   └── dashboard.html          # Web UI
├── voice_profiles/             # Saved custom voice profiles (auto-created)
├── checkpoints/                # Model weights (not in repo — see below)
└── tests/
```

---

## Configuration

All settings are configurable via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `AVATAR_CAMERA_INDEX` | `0` | Webcam device index |
| `AVATAR_FRAME_WIDTH` | `640` | Capture width |
| `AVATAR_FRAME_HEIGHT` | `480` | Capture height |
| `AVATAR_TARGET_FPS` | `30` | Target frame rate |
| `AVATAR_SAMPLE_RATE` | `16000` | Audio sample rate |
| `AVATAR_SERVICE_TIMEOUT` | `10` | AI service request timeout (seconds) |
| `AVATAR_OUTPUT_WIDTH` | `640` | Output resolution width |
| `AVATAR_OUTPUT_HEIGHT` | `480` | Output resolution height |
| `AVATAR_OUTPUT_FPS` | `30` | Output frame rate |
| `AVATAR_VIRTUAL_CAMERA` | *(auto)* | Override virtual camera device name |
| `AVATAR_VIRTUAL_MIC` | *(auto)* | Override virtual mic device name |

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard |
| `GET` | `/health` | Server health check |
| `GET` | `/api/status` | Pipeline status + stats |
| `GET` | `/api/services/health` | AI service health |
| `POST` | `/api/pipeline/start` | Start the pipeline (`?mode=full` or `?mode=audio`) |
| `POST` | `/api/pipeline/stop` | Stop the pipeline |
| `POST` | `/api/avatar/upload` | Upload avatar image |
| `GET` | `/api/voice/speakers` | List voice profiles |
| `POST` | `/api/voice/speaker` | Set active voice |
| `POST` | `/api/voice/upload` | Upload voice sample |
| `GET` | `/api/stream/video` | MJPEG avatar output stream |
| `GET` | `/api/stream/webcam` | MJPEG webcam input stream |
| `GET` | `/api/devices` | Virtual device info |

---

## AI Models

| Service | Model | Port | Purpose |
|---|---|---|---|
| Face Animation | First Order Motion Model (FOMM) | 8001 | Animate avatar with your expressions |
| Voice Conversion | HuBERT + WORLD vocoder | 8002 | Real-time voice transformation |
| Lip Sync | Wav2Lip | 8003 | Match avatar lips to speech |

### Model Checkpoints

Model weights are **not included** in the repository (they're large binary files). The Docker containers will download them automatically on first build.

If running without Docker, place checkpoints in `checkpoints/`:
- `vox-cpk.pth.tar` — FOMM weights
- `wav2lip_gan.pth` — Wav2Lip weights

---

## Development

### Running without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start just the web server (no AI inference)
python main.py
```

### Running tests

```bash
python -m pytest tests/ -v
```

### Code structure

- **Capture** → raw webcam frames + audio chunks
- **Tracking** → MediaPipe face landmarks + expression analysis
- **Services** → async HTTP clients to Docker inference servers
- **Rendering** → frame compositing + A/V synchronization
- **Output** → virtual camera (pyvirtualcam) + virtual mic (sounddevice)

---

## Roadmap

Planned features and research directions for upcoming releases:

### Near-Term
- [ ] **Multi-avatar switching** — Hot-swap between different avatar faces mid-call
- [ ] **Emotion intensity slider** — Control how much expression transfers (subtle to exaggerated)
- [x] **Audio-only mode** — Voice conversion without the camera pipeline for lower resource usage
- [x] **Persistent voice profiles** — Upload voice samples that are saved to disk and reloaded across restarts
- [ ] **Session recording** — Record avatar output locally for review or demo purposes
- [ ] **Linux and macOS support** — PulseAudio/PipeWire virtual mic, v4l2loopback virtual camera

### Mid-Term
- [ ] **3D avatar support** — Drive a 3D mesh head (e.g. Ready Player Me, MetaHuman) instead of 2D image
- [ ] **Background replacement** — Virtual backgrounds composited behind the avatar
- [ ] **Multi-language voice cloning** — Cross-lingual voice conversion preserving accent and prosody
- [ ] **Real-time style transfer** — Apply artistic styles (cartoon, anime, sketch) to the avatar
- [ ] **Mobile companion app** — Control the avatar system from your phone during a call

### Long-Term Research
- [ ] **Zero-shot avatar generation** — Generate a completely new face from a text description
- [ ] **Gesture synthesis** — Generate upper-body movement and hand gestures from speech
- [ ] **Emotion-aware voice modulation** — Automatically adjust voice tone based on detected facial emotion
- [ ] **Edge deployment** — Optimized models for running inference directly on-device without Docker
- [ ] **Live translation overlay** — Real-time speech translation with lip-synced output in another language

---

## Areas of Collaboration

We're actively looking for contributors and collaborators in these areas:

### Research Collaboration
| Area | What We Need | Relevant Skills |
|---|---|---|
| **Generative models** | Better face animation with fewer artifacts | GANs, diffusion models, PyTorch |
| **Voice synthesis** | Lower-latency real-time voice conversion | Speech processing, DSP, vocoder design |
| **Lip sync accuracy** | Improved mouth shape prediction from audio | Audio-visual ML, phoneme mapping |
| **Model compression** | Smaller models for CPU/edge inference | Quantization, pruning, knowledge distillation |
| **Evaluation metrics** | Perceptual quality benchmarks for avatar output | Image quality assessment, user studies |

### Engineering Collaboration
| Area | What We Need | Relevant Skills |
|---|---|---|
| **Cross-platform** | Linux virtual devices, macOS CoreAudio integration | Systems programming, OS-level APIs |
| **WebRTC integration** | Direct browser-to-browser avatar streaming | WebRTC, media servers |
| **GPU optimization** | CUDA kernels, TensorRT model export | CUDA, ONNX, TensorRT |
| **CI/CD pipeline** | Automated testing, Docker image publishing | GitHub Actions, container registries |
| **Security audit** | Review virtual device isolation and data flow | Application security, threat modeling |

### Community Collaboration
- **Documentation** — Tutorials, video guides, translation to other languages
- **UI/UX design** — Dashboard improvements, accessibility, mobile-responsive layouts
- **Testing** — Cross-platform testing, edge case discovery, performance benchmarking
- **Use case exploration** — Accessibility tools, privacy-preserving video calls, creative applications

### How to Get Involved
1. Check the [Issues](https://github.com/Mwaivictor/Avatar/issues) tab for open tasks
2. Join a discussion in [Discussions](https://github.com/Mwaivictor/Avatar/discussions)
3. Propose a new feature by opening an issue with the `enhancement` label
4. Reach out for research collaboration by opening an issue with the `research` label

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Ethical Use

This software is intended for **educational and research purposes**. Users are responsible for ensuring their use complies with:

- Local laws regarding identity and impersonation
- Platform terms of service (Zoom, Meet, Teams, etc.)
- Consent of people being communicated with
- Organizational policies on video call modifications

**Do not** use this system to deceive, defraud, or impersonate others without their knowledge and consent.

---

## Acknowledgments

- [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) — Aliaksandr Siarohin et al.
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) — Rudrabha Mukhopadhyay et al.
- [MediaPipe](https://github.com/google/mediapipe) — Google
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) — Maik Riechert
- [OBS Studio](https://obsproject.com/) — OBS Project
- [VB-Audio](https://vb-audio.com/) — VB-Audio Software
