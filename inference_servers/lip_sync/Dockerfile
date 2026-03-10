FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Checkpoint directory (mount or copy weights here)
RUN mkdir -p /app/checkpoints

# Install CPU-only PyTorch (no CUDA — keeps image ~1.5 GB smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY inference.py .
COPY server.py .

# Model checkpoint path (override via docker-compose or -e)
ENV WAV2LIP_CHECKPOINT=/app/checkpoints/wav2lip_gan.pth

EXPOSE 8003
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
