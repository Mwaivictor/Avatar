# Contributing to Avatar

Thank you for your interest in contributing! This document provides guidelines for contributing to the Avatar project.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/Mwaivictor/Avatar.git
   cd avatar
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

### Making Changes

1. Make your changes in your feature branch
2. Test your changes locally:
   ```bash
   python -m pytest tests/ -v
   ```
3. Ensure the system still starts correctly:
   ```bash
   python start.py
   ```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add new voice profile selection UI
fix: resolve camera capture timeout on Windows
docs: update API endpoint documentation
refactor: simplify face tracker initialization
```

Prefixes:
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `refactor:` — Code refactoring (no behavior change)
- `test:` — Adding or updating tests
- `chore:` — Build, CI, or tooling changes

### Pull Requests

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a Pull Request against the `main` branch
3. Describe what your PR does and why
4. Link any related issues

## What to Contribute

### Good First Issues

- Improving error messages and logging
- Adding tests for existing modules
- Documentation improvements
- UI/UX improvements to the dashboard

### Feature Ideas

- Additional AI model backends
- New voice conversion models
- Mobile browser support for the dashboard
- Recording/playback functionality
- Additional virtual device backends (Linux PulseAudio, macOS CoreAudio)

### Bug Reports

When reporting bugs, include:
- Python version (`python --version`)
- Operating system and version
- Docker version (`docker --version`)
- Steps to reproduce
- Error messages / stack traces
- Expected vs actual behavior

## Code Style

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Keep functions focused and small
- Add docstrings to public classes and methods
- Use `logging` instead of `print()` for output

## Project Structure

- `app/capture/` — Input capture (webcam, microphone)
- `app/tracking/` — Face detection and expression analysis
- `app/services/` — AI inference service clients
- `app/rendering/` — Frame compositing and A/V sync
- `app/output/` — Virtual device output
- `app/api/` — REST API and web interface
- `inference_servers/` — Docker-based AI model servers
- `static/` — Web dashboard files
- `tests/` — Test suite

## AI Model Contributions

If contributing new AI models:
1. Create a new directory under `inference_servers/`
2. Include a `Dockerfile`, `server.py`, and `requirements.txt`
3. Follow the existing health-check endpoint pattern (`GET /health`)
4. Document model requirements and expected input/output formats

## Questions?

Open an issue on GitHub with the `question` label, or start a discussion in the Discussions tab.

---

Thank you for helping make Avatar better!
