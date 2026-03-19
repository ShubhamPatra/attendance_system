# Production-ready Dockerfile for AutoAttendance
FROM python:3.11-slim

# Install system dependencies for dlib, OpenCV, and CMake
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libboost-all-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create required directories
RUN mkdir -p uploads unknown_faces backups logs models

# Expose Flask port
EXPOSE 5000

# Environment variables with defaults
ENV MONGO_URI=""
ENV SECRET_KEY="change-me-in-production"
ENV CELERY_BROKER_URL="redis://redis:6379/0"
ENV CELERY_RESULT_BACKEND="redis://redis:6379/0"

# Run with gunicorn in production
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:create_app"]
