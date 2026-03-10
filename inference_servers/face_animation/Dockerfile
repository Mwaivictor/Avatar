FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (no CUDA — keeps image ~1.5 GB smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and server code
COPY models/ models/
COPY inference.py .
COPY server.py .

# Create checkpoints directory (mount or copy weights here)
RUN mkdir -p /app/checkpoints

ENV FOMM_CHECKPOINT=/app/checkpoints/vox-cpk.pth.tar

EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
