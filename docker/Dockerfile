# Base image: Python 3.12-slim (compatible with Python 3.9+ code)
# Override with: docker build --build-arg PYTHON_VERSION=3.11 .
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# Control ONNX Runtime variant: onnxruntime (CPU) or onnxruntime-gpu (GPU)
ARG INSTALL_GPU=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for OpenCV, ONNX, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements-docker.txt /app/requirements-docker.txt

# Install base and PyTorch, optionally GPU variant
RUN pip install --no-cache-dir -r /app/requirements-docker.txt && \
    if [ "$INSTALL_GPU" -eq 1 ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
            'torch>=2.0,<3.0' 'torchvision>=0.15,<1.0'; \
    else \
        pip install --no-cache-dir 'torch>=2.0,<3.0' 'torchvision>=0.15,<1.0'; \
    fi

COPY . /app

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:create_app()"]
