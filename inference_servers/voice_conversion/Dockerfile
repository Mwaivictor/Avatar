FROM python:3.11-slim

WORKDIR /app

# System dependencies for pyworld and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (no CUDA — keeps image ~1.5 GB smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY server.py .

# HuBERT model will auto-download from HuggingFace on first load.
# Set to "false" to disable HuBERT and use only WORLD vocoder.
ENV VC_USE_HUBERT=true

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
