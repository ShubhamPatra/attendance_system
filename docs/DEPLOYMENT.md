# Deployment & Operations

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Kubernetes Deployment (Phase 7)](#kubernetes-deployment-phase-7)
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
MONGO_URI=mongodb+srv://user:REDACTED/attendance_system

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
 YuNet model: models/face_detection_yunet_2023mar.onnx
 Anti-spoofing model: models/anti_spoofing/
 All models verified
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
# Start unified application (runs both admin and student apps)
python run.py
```

**Access**:

- Admin: http://localhost:5000
- Student: http://localhost:5001

---

## Kubernetes Deployment (Phase 7)

### Overview

Phase 7 provides production-ready Kubernetes deployment with:
- Health probes (startup, liveness, readiness)
- Horizontal Pod Autoscaling (HPA)
- Network policies and security
- RBAC configuration
- Ingress with TLS support

**For comprehensive guide, see [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md)**

### Quick Start

```bash
# 1. Create namespace
kubectl create namespace attendance-system

# 2. Create secrets (edit with actual values)
kubectl create secret generic attendance-secrets \
  -n attendance-system \
  --from-literal=MONGO_URI='mongodb+srv://user:pass@cluster.mongodb.net' \
  --from-literal=SECRET_KEY='your-secure-key'

# 3. Apply configurations
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml

# 4. Verify deployment
kubectl get pods -n attendance-system
kubectl logs -f -n attendance-system deployment/attendance-admin
```

### Health Probes

Three health checks ensure reliability:

| Probe | Endpoint | Interval | Purpose |
|-------|----------|----------|---------|
| **Startup** | `/api/health` | 5s | Gives app 35s to initialize |
| **Liveness** | `/api/health` | 10s | Restarts if unhealthy 30s+ |
| **Readiness** | `/api/readiness` | 5s | Removes from service if not ready |

### Scaling

- **Min replicas**: 2
- **Max replicas**: 5
- **Scale triggers**: CPU >70%, Memory >80%
- **Auto-scale disabled**: Set `maxReplicas: <minReplicas>` in HPA

### Configuration

Edit `deploy/k8s/configmap.yaml` and `deploy/k8s/deployment.yaml`:
- Update MongoDB connection string
- Update domain in Ingress rules
- Adjust resource requests/limits for your load
- Configure TLS certificates

### Troubleshooting

```bash
# Check pod status
kubectl describe pod -n attendance-system <pod-name>

# View app logs
kubectl logs -f -n attendance-system deployment/attendance-admin

# Test health endpoints (port-forward)
kubectl port-forward -n attendance-system svc/attendance-admin 8080:80
curl http://localhost:8080/api/health
```

**See [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md) for detailed troubleshooting.**

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
MONGO_URI=mongodb+srv://user:REDACTED/db_name
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

### Security First: Authentication Configuration

**⚠️ CRITICAL: Always enable authentication in production**

The `AUTH_REQUIRED` environment variable controls whether authentication is enforced. This defaults to `true` (enabled), which is the secure production setting.

#### For Production Deployments

```bash
# .env - Production settings
AUTH_REQUIRED=true          # ✅ REQUIRED - Always true in production
MONGO_URI=...               # Use MongoDB Atlas or secured cluster
SECRET_KEY=<long-random-string>  # Use strong, random secret
```

**Verification**: Check application startup logs for:
```
✅ Authentication is ENABLED (production mode)
```

#### For Local Development Only

```bash
# .env - Development settings (never commit to production!)
AUTH_REQUIRED=false         # ⚠️ DEVELOPMENT ONLY - Disables authentication
```

**Warning**: Check application startup logs for:
```
⚠️ Authentication is DISABLED (development mode only - set AUTH_REQUIRED=true for production)
```

Do not use `AUTH_REQUIRED=false` in production under any circumstances. All protected routes will be accessible without authentication.

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

---

## Scaling & Performance Tuning

### Horizontal Scaling

Add more server instances behind a load balancer (AWS ELB, GCP Load Balancer, Azure Load Balancer, or Nginx).

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
# Tail application logs
tail -f logs/attendance.log

# Tail access logs
tail -f logs/access.log

# Tail error logs
tail -f logs/error.log
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
# Stop application
sudo systemctl stop gunicorn

# Backup application data
tar -czf attendance-system-backup-$(date +%Y-%m-%d).tar.gz \
  logs/ uploads/ unknown_faces/ models/

# Backup MongoDB database
mongodump --uri "mongodb+srv://user:pass@cluster/db" \
  --out ./backups/$(date +%Y-%m-%d-%H-%M-%S)

# Restart application
sudo systemctl start gunicorn
```

### Recovery Procedure

```bash
# 1. Stop application
sudo systemctl stop gunicorn

# 2. Restore database
mongorestore --uri "mongodb+srv://user:pass@cluster/db" \
  ./backups/2024-09-15-09-15-23

# 3. Restore application data (if corrupted)
tar -xzf attendance-system-backup-2024-09-15.tar.gz

# 4. Restart application
sudo systemctl start gunicorn

# 5. Verify
curl http://localhost:5000/health
```

---

## Troubleshooting

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
print(' Connection successful')
"
```

### Issue: GPU Not Detected

```
CUDA not available; falling back to CPU
```

**Solution**:

```bash
# Verify NVIDIA GPU drivers installed
nvidia-smi

# Set GPU environment variable
export ENABLE_GPU_PROVIDERS=1

# Test GPU with Python
python -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
"

# Restart application with GPU enabled
sudo systemctl restart gunicorn
```

### Issue: High Latency (Slow Face Detection)

```
YuNet detection taking > 100ms per frame
```

**Solution**:

```bash
# 1. Enable GPU acceleration
export ENABLE_GPU_PROVIDERS=1

# 2. Increase detection interval (detect less frequently)
export DETECTION_INTERVAL=12

# 3. Reduce frame size
export FRAME_PROCESS_WIDTH=320

# 4. Use KCF tracker (faster, less accurate)
export PERF_USE_KCF_TRACKER=1

# 5. Restart application
sudo systemctl restart gunicorn
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

