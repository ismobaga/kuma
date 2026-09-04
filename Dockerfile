FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Install base dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    opencv-python==4.8.1.78 \
    yt-dlp==2026.8.19 \
    pandas==2.1.3 \
    pyarrow==14.0.0 \
    numpy==1.24.3 \
    python-dotenv==1.0.0 \
    huggingface-hub==0.19.4

# Install ASR (optional, separate layer for caching)
# Uncomment the next line to include ASR in build
# RUN pip install --no-cache-dir transformers torch
# For CPU only:
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# For GPU (nvidia):
# RUN pip install --no-cache-dir torch

# Copy application code
COPY app.py .
COPY index.html .

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run app
CMD ["python", "app.py"]