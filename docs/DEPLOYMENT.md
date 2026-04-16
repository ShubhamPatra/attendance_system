# Deployment & Operations

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Environment Configuration](#environment-configuration)
4. [Production Setup](#production-setup)
5. [Scaling & Performance Tuning](#scaling--performance-tuning)
6. [Monitoring & Logging](#monitoring--logging)
7. [Backup & Disaster Recovery](#backup--disaster-recovery)
8. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- Python 3.9+ (tested on 3.12)
- MongoDB Atlas account or local MongoDB server
- Git
- Webcam or IP camera

### Step 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system

# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use editor of choice
```

**Essential Settings**:

```bash
# MongoDB
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/attendance_system

# Flask
FLASK_ENV=development
SECRET_KEY=dev-secret-key-change-in-production

# Admin app
ADMIN_APP_HOST=0.0.0.0
ADMIN_APP_PORT=5000

# Student app
STUDENT_APP_HOST=0.0.0.0
STUDENT_APP_PORT=5001

# ML settings
EMBEDDING_BACKEND=arcface
RECOGNITION_THRESHOLD=0.38
ENABLE_GPU_PROVIDERS=0  # Set to 1 for GPU

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/attendance.log
```

### Step 3: Download Models

```bash
# Verify Python environment
python scripts/verify_versions.py

# Download required models
python scripts/download_models.py
```

**Expected Output**:

```
✓ YuNet model: models/face_detection_yunet_2023mar.onnx
✓ Anti-spoofing model: models/anti_spoofing/
✓ All models verified
```

### Step 4: Bootstrap Admin User

```bash
# Create admin account
python scripts/bootstrap_admin.py \
  --username admin \
  --password secure-password-here
```

### Step 5: Run Development Servers

```bash
# Option 1: Both apps together
python run.py

# Option 2: Separate terminals
# Terminal 1: Admin app
python run_admin.py

# Terminal 2: Student app  
python run_student.py

# Terminal 3: Nginx (optional)
nginx -c $(pwd)/deploy/nginx/nginx.conf
```

**Access**:

- Admin: http://localhost:5000
- Student: http://localhost:5001
- Nginx: http://localhost

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (optional, for GPU)

### CPU Build

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# View logs
docker compose logs -f web student-web nginx

# Stop services
docker compose down
```

**Service Status**:

```bash
docker compose ps

# Output:
# NAME         STATUS              PORTS
# web          Up 2 minutes        5000/tcp
# student-web  Up 2 minutes        5001/tcp
# nginx        Up 2 minutes        80/tcp
```

### GPU Build

```bash
# Set GPU flag and build
INSTALL_GPU=1 docker compose build

# Start with GPU
docker compose up -d

# Verify GPU
docker compose exec web python -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
"
```

### Docker Compose Configuration

**File**: [docker-compose.yml](../docker-compose.yml)

```yaml
version: '3.9'

services:
  web:
    build:
      context: .
      dockerfile: docker/Dockerfile
      args:
        INSTALL_GPU: ${INSTALL_GPU:-0}
        PYTHON_VERSION: ${PYTHON_VERSION:-3.12}
    ports:
      - "5000:5000"
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
      - ./unknown_faces:/app/unknown_faces
    environment:
      - FLASK_ENV=production
      - MONGO_URI=${MONGO_URI}
      - ENABLE_GPU_PROVIDERS=${ENABLE_GPU_PROVIDERS:-0}
    restart: unless-stopped

  student-web:
    build:
      context: .
      dockerfile: docker/Dockerfile.student
      args:
        PYTHON_VERSION: ${PYTHON_VERSION:-3.12}
    ports:
      - "5001:5001"
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    environment:
      - FLASK_ENV=production
      - MONGO_URI=${MONGO_URI}
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - web
      - student-web
    restart: unless-stopped
```

---

## Environment Configuration

### Configuration Precedence

```
Command-line arguments (highest)
       ↓
Environment variables (.env file)
       ↓
config.py defaults (lowest)
```

### Critical Parameters

**Database**

```bash
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/db_name
MONGO_MAX_POOL_SIZE=50                    # Connection pool size
MONGO_CIRCUIT_BREAKER_THRESHOLD=5         # Failures before circuit opens
```

**ML Pipeline**

```bash
EMBEDDING_BACKEND=arcface                # arcface or dlib
RECOGNITION_THRESHOLD=0.38               # Cosine similarity threshold
RECOGNITION_MIN_CONFIDENCE=0.46          # Minimum match score
BLUR_THRESHOLD=6.0                       # Laplacian variance minimum
RECOGNITION_CONFIRM_FRAMES=2             # Multi-frame voting threshold
LIVENESS_HISTORY_SIZE=5                  # Rolling liveness buffer
LIVENESS_CONFIDENCE_THRESHOLD=0.55       # Liveness gate
```

**Performance**

```bash
FRAME_PROCESS_WIDTH=512                  # Detection frame width
DETECTION_INTERVAL=6                     # Frames between detections
PERF_MAX_FACES=5                        # Max simultaneous tracks
PERF_USE_KCF_TRACKER=0                   # Use KCF (faster, less accurate)
FRAME_RESIZE_FACTOR=0.25                 # Global frame resize
```

**GPU Support**

```bash
ENABLE_GPU_PROVIDERS=1                   # Enable CUDA/GPU
ONNXRT_PROVIDER_PRIORITY=CUDAExecutionProvider,CPUExecutionProvider
```

**Logging**

```bash
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
LOG_FILE=logs/attendance.log
MAX_LOG_SIZE_MB=50
LOG_BACKUP_COUNT=5
```

### Example .env Files

**Development**

```bash
# .env.dev
FLASK_ENV=development
DEBUG=1
MONGO_URI=mongodb://localhost:27017/attendance_system
EMBEDDING_BACKEND=arcface
ENABLE_GPU_PROVIDERS=0
LOG_LEVEL=DEBUG
```

**Production (MongoDB Atlas)**

```bash
# .env.prod
FLASK_ENV=production
DEBUG=0
MONGO_URI=mongodb+srv://prod_user:$(cat /run/secrets/mongo_password)@cluster.mongodb.net/attendance_system
MONGO_MAX_POOL_SIZE=100
EMBEDDING_BACKEND=arcface
ENABLE_GPU_PROVIDERS=1
LOG_LEVEL=WARNING
SECRET_KEY=$(cat /run/secrets/flask_secret)
```

---

## Production Setup

### Recommended Architecture

```
                    ┌─────────────────┐
                    │   DNS / Load    │
                    │    Balancer     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
    │ Server1 │         │ Server2 │         │ Server3 │
    │ (Nginx) │         │ (Nginx) │         │ (Nginx) │
    └────┬────┘         └────┬────┘         └────┬────┘
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
    │ Gunicorn│         │ Gunicorn│         │ Gunicorn│
    │ (4 wkr) │         │ (4 wkr) │         │ (4 wkr) │
    └────┬────┘         └────┬────┘         └────┬────┘
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  MongoDB Atlas  │
                    │  (sharded)      │
                    └─────────────────┘
```

### Gunicorn Configuration

**File**: [gunicorn.conf.py](../gunicorn.conf.py)

```python
# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Workers
workers = 4  # (2 × CPU_cores) + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
graceful_timeout = 30
keepalive = 2

# Process naming
proc_name = "attendance-system"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# SSL (optional)
# keyfile = "/etc/ssl/private/key.pem"
# certfile = "/etc/ssl/certs/cert.pem"
# ssl_version = "TLSv1_2"
```

### Nginx Configuration

**File**: [docker/nginx/nginx.conf](../docker/nginx/nginx.conf)

```nginx
upstream admin_backend {
    server web:5000;
}

upstream student_backend {
    server student-web:5001;
}

server {
    listen 80;
    server_name _;
    client_max_body_size 10M;

    # Admin app
    location / {
        proxy_pass http://admin_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /socket.io {
        proxy_pass http://admin_backend/socket.io;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Student app
    location /student/ {
        proxy_pass http://student_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Static assets
    location /static/ {
        alias /app/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

### SSL/TLS Setup

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Or use Let's Encrypt (production)
certbot certonly --standalone -d yourdomain.com

# Configure Nginx for HTTPS
# Update docker/nginx/nginx.conf:
# listen 443 ssl http2;
# ssl_certificate /etc/nginx/ssl/cert.pem;
# ssl_certificate_key /etc/nginx/ssl/key.pem;
```

---

## Scaling & Performance Tuning

### Horizontal Scaling

Add more server instances behind load balancer:

```yaml
# docker-compose.yml (multi-server setup)
version: '3.9'

services:
  web-1:
    image: attendance-system:latest
    # ... config
  
  web-2:
    image: attendance-system:latest
    # ... config
  
  nginx:
    image: nginx:alpine
    volumes:
      - ./docker/nginx/nginx-lb.conf:/etc/nginx/nginx.conf
    # Distributes traffic to web-1, web-2, ...
```

### Vertical Scaling

Increase resources on single server:

```bash
# Monitor GPU utilization
nvidia-smi

# Monitor CPU/memory
top
htop

# Increase worker count (if CPU-bound)
# In gunicorn.conf.py: workers = 8
```

### Detection Interval Tuning

Adjust detection frequency for speed-accuracy trade-off:

```bash
# Faster (skip more frames)
DETECTION_INTERVAL=12  # Detect every 12 frames (3× speedup)

# Slower but more accurate
DETECTION_INTERVAL=3   # Detect every 3 frames (higher latency)
```

### Database Connection Pool

Tune MongoDB connection pooling:

```bash
# More connections for higher concurrency
MONGO_MAX_POOL_SIZE=100

# Fewer connections for resource-constrained environments
MONGO_MAX_POOL_SIZE=20
```

---

## Monitoring & Logging

### Log Levels

```bash
LOG_LEVEL=DEBUG    # Verbose; includes all events
LOG_LEVEL=INFO     # Standard; attendance marks, sessions, errors
LOG_LEVEL=WARNING  # Only warnings and errors
LOG_LEVEL=ERROR    # Only errors
```

### Log Files

```
logs/
├─ attendance.log          # Main application log
├─ access.log              # HTTP access log (Gunicorn)
├─ error.log               # HTTP error log (Gunicorn)
└─ camera_debug.log        # Camera loop debug trace
```

### Real-Time Monitoring

```bash
# Tail admin app logs
docker compose logs -f web

# Tail student app logs
docker compose logs -f student-web

# Tail Nginx logs
docker compose logs -f nginx
```

### Health Check Endpoint

```bash
# Check application status
curl http://localhost:5000/health

# Response (200 OK):
{
  "status": "healthy",
  "timestamp": "2024-09-15T09:15:23.123Z",
  "database": "connected",
  "models": "loaded",
  "camera": "ready"
}
```

---

## Backup & Disaster Recovery

### MongoDB Backup

```bash
# Automated backup (via MongoDB Atlas)
# Configure in Atlas console: Backup → Enable Continuous Backups

# Manual backup (mongodump)
mongodump --uri "mongodb+srv://user:pass@cluster/db" \
  --out ./backups/$(date +%Y-%m-%d-%H-%M-%S)

# Restore from backup
mongorestore --uri "mongodb+srv://user:pass@cluster/db" \
  ./backups/2024-09-15-09-15-23
```

### Application Data Backup

```bash
# Backup volumes
docker compose down
tar -czf attendance-system-backup-$(date +%Y-%m-%d).tar.gz \
  logs/ uploads/ unknown_faces/ models/

docker compose up -d
```

### Recovery Procedure

```bash
# 1. Stop services
docker compose down

# 2. Restore database
mongorestore --uri "mongodb+srv://user:pass@cluster/db" \
  ./backups/2024-09-15-09-15-23

# 3. Restore volumes (if corrupted)
tar -xzf attendance-system-backup-2024-09-15.tar.gz

# 4. Restart services
docker compose up -d

# 5. Verify
curl http://localhost:5000/health
```

---

## Troubleshooting

### Issue: Docker Build Fails

```
Error: "Could not find ONNX model..."
```

**Solution**:

```bash
# Ensure models directory exists
mkdir -p models/

# Download models manually
python scripts/download_models.py

# Rebuild Docker image
docker compose build --no-cache
```

### Issue: MongoDB Connection Error

```
pymongo.errors.ServerSelectionTimeoutError: No servers...
```

**Solution**:

```bash
# Check MONGO_URI in .env
cat .env | grep MONGO_URI

# Test connection
python -c "
from pymongo import MongoClient
import os
uri = os.getenv('MONGO_URI')
client = MongoClient(uri, serverSelectionTimeoutMS=5000)
client.admin.command('ping')
print('✓ Connection successful')
"
```

### Issue: GPU Not Detected

```
CUDA not available; falling back to CPU
```

**Solution**:

```bash
# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list

# Verify NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# Rebuild with GPU
INSTALL_GPU=1 docker compose build
docker compose up -d

# Verify GPU in container
docker compose exec web python -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Device: {torch.cuda.get_device_name(0)}')
"
```

### Issue: High Latency (Slow Face Detection)

```
YuNet detection taking > 100ms per frame
```

**Solution**:

```bash
# 1. Enable GPU acceleration
ENABLE_GPU_PROVIDERS=1
INSTALL_GPU=1 docker compose build

# 2. Increase detection interval (detect less frequently)
DETECTION_INTERVAL=12

# 3. Reduce frame size
FRAME_PROCESS_WIDTH=320

# 4. Use KCF tracker (faster, less accurate)
PERF_USE_KCF_TRACKER=1
```

---

## Summary

AutoAttendance deployment provides:

1. **Flexibility**: Local, Docker, and cloud-ready setups.
2. **Scalability**: Horizontal scaling via load balancers; vertical scaling via resource tuning.
3. **Reliability**: Automated backups, circuit breaker, graceful degradation.
4. **Observability**: Comprehensive logging and health endpoints.

Next steps:
- See [TESTING.md](TESTING.md) for validation procedures.
- See [RESEARCH.md](RESEARCH.md) for evaluation metrics.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
