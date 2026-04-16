# Build from Scratch Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Setup](#repository-setup)
3. [Environment Configuration](#environment-configuration)
4. [Model Download](#model-download)
5. [Database Setup](#database-setup)
6. [First Attendance Capture](#first-attendance-capture)
7. [Verification Checklist](#verification-checklist)

---

## Prerequisites

### System Requirements

**Minimum (CPU-Only)**:
- OS: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+
- CPU: 4 cores (8 recommended)
- RAM: 8 GB (16 GB recommended)
- Storage: 20 GB (includes models + logs)
- Python: 3.9, 3.10, 3.11, or 3.12

**Recommended (GPU Support)**:
- GPU: NVIDIA RTX 2080 or better (for real-time processing)
- CUDA Toolkit: 11.8 or 12.0
- cuDNN: 8.6+

### Software Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
  python3.12 python3.12-venv python3.12-dev \
  git curl wget \
  build-essential cmake \
  libopencv-dev

# macOS (via Homebrew)
brew install python@3.12 cmake opencv

# Windows: Download from official websites
# - Python 3.12: https://www.python.org/
# - Git: https://git-scm.com/
# - Build Tools: Visual Studio Community 2022
```

### MongoDB Setup (Choose One)

#### Option A: Local MongoDB (Development)

```bash
# Ubuntu/Debian
sudo apt-get install -y mongodb

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify
mongosh

> use attendance_system
> db.students.insertOne({ test: true })
> db.students.findOne()
# Should return: { _id: ObjectId(...), test: true }
```

#### Option B: MongoDB Atlas (Cloud, Recommended)

```bash
# 1. Create account at https://www.mongodb.com/cloud/atlas
# 2. Create free cluster
# 3. Create database user
# 4. Get connection string (looks like):
#    mongodb+srv://user:password@cluster0.mongodb.net/
# 5. Note this connection string for environment setup
```

---

## Repository Setup

### Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system

# Verify directory structure
ls -la
# Should see: README.md, docker-compose.yml, requirements.txt, etc.

# Verify critical directories
test -d admin_app && echo "✓ admin_app exists"
test -d student_app && echo "✓ student_app exists"
test -d core && echo "✓ core exists"
test -d vision && echo "✓ vision exists"
test -d models && echo "✓ models directory exists"
```

### Create Python Virtual Environment

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Verify activation
python --version
# Should show: Python 3.12.x

which python  # Linux/macOS
# Should show: /path/to/attendance_system/venv/bin/python
```

### Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
pip install -r requirements.txt

# Verify critical packages
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import pymongo; print(f'PyMongo: {pymongo.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"

# All should print versions without errors
```

---

## Environment Configuration

### Create .env File

```bash
# Create .env in project root
cat > .env << 'EOF'
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017  # Local
# MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/  # Atlas

MONGODB_DATABASE=attendance_system

# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here-generate-with-python-secrets  # See below

# Face Recognition Configuration
RECOGNITION_THRESHOLD=0.38
LIVENESS_THRESHOLD=0.55
DETECTION_INTERVAL=6

# GPU Configuration (set to 1 for NVIDIA GPU)
ATTENDANCE_ENABLE_GPU=0

# Logging
LOG_LEVEL=INFO

# Feature Flags
ENABLE_RBAC=0
ENABLE_NOTIFICATIONS=1
ENABLE_ANALYTICS=1
EOF
```

### Generate Secret Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Output: abc123def456...

# Add to .env
echo "SECRET_KEY=<output-from-above>" >> .env

# Verify .env loaded
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f'SECRET_KEY: {os.getenv(\"SECRET_KEY\")}')"
```

---

## Model Download

### Verify Model Directory

```bash
# Check if models directory exists and is writable
ls -la models/
# Should contain: face_detection_yunet_2023mar.onnx, anti_spoofing/ (directory)

mkdir -p models/anti_spoofing
chmod -R 755 models
```

### Download Models (Automated)

```bash
# Script: scripts/download_models.py (provided)
python scripts/download_models.py

# Monitoring output:
# Downloading YuNet face detection model...
# Downloading ArcFace embeddings model...
# Downloading Silent-Face anti-spoofing models...
# ✓ All models downloaded successfully
```

### Manual Download (If Script Fails)

#### YuNet Face Detection

```bash
cd models

# Download YuNet ONNX model
wget https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Verify
ls -lh face_detection_yunet_2023mar.onnx
# Should be ~6 MB

cd ..
```

#### ArcFace Embeddings

```python
# In Python console
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CPUProvider'])
app.prepare(ctx_id=-1, det_model='retinaface', rec_model='arcface_r100_v1')
# This auto-downloads to ~/.insightface/models/

print("Models downloaded to ~/.insightface/models/")
```

#### Silent-Face Anti-Spoofing

```bash
cd models

# Clone Silent-Face repository
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git anti_spoofing

cd anti_spoofing

# Download pre-trained models from releases
wget https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases/download/v1.0/model_lead.pth
wget https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases/download/v1.0/model_auxiliary.pth

cd ../..
```

### Verify Models

```bash
python scripts/verify_versions.py

# Output:
# ✓ OpenCV: 4.8.1
# ✓ PyTorch: 2.0.1
# ✓ ONNX Runtime: 1.17.0
# ✓ InsightFace: 0.7.3
# ✓ YuNet model: found (6.1 MB)
# ✓ ArcFace model: found (500.2 MB)
# ✓ Silent-Face model: found (1.2 MB)
# All systems ready!
```

---

## Database Setup

### Create Database & User

```bash
# Connect to MongoDB
mongosh  # or mongo (older versions)

# Switch to admin database
use admin

# Create admin user (if not already created)
db.createUser({
  user: "admin",
  pwd: "strong-password-here",
  roles: [ { role: "root", db: "admin" } ]
})

# Switch to attendance_system database
use attendance_system

# Create application user
db.createUser({
  user: "attendance_app",
  pwd: "app-password-here",
  roles: [
    { role: "readWrite", db: "attendance_system" }
  ]
})
```

### Initialize Collections & Indexes

```bash
# Run bootstrap script (creates collections, indexes, admin user)
python scripts/bootstrap_admin.py

# Prompts:
# > Enter admin email: admin@university.edu
# > Enter admin password: ••••••••
# > Enter admin name: Admin User

# Creates:
# ✓ MongoDB collections (students, attendance, sessions, users, notification_events)
# ✓ Indexes (unique on student_id+date, etc.)
# ✓ Admin user in database
```

### Verify Database Setup

```bash
# Connect and verify
mongosh "mongodb://localhost:27017/attendance_system"

# List collections
show collections

# Verify collections exist
db.students.countDocuments()      # Should be 0 (empty)
db.attendance.countDocuments()    # Should be 0 (empty)
db.users.countDocuments()         # Should be 1 (admin user)

# Verify indexes
db.attendance.getIndexes()
# Should show unique index on student_id, date
```

---

## First Attendance Capture

### Step 1: Seed Demo Data

```bash
python scripts/seed_demo_data.py

# This creates:
# - 5 test students (with pre-generated embeddings)
# - 1 test course (CS101)
# - 1 test attendance session
# 
# Output:
# Created 5 students: CS21001–CS21005
# Created course: CS101
# Enrollment complete
```

### Step 2: Start Admin App

```bash
# Terminal 1: Start admin app (port 5000)
python run_admin.py

# Output:
# * Running on http://localhost:5000
# * WARNING: This is a development server...
# * Press CTRL+C to quit
```

### Step 3: Start Student App

```bash
# Terminal 2: Start student app (port 5001)
python run_student.py

# Output:
# * Running on http://localhost:5001
```

### Step 4: Access Applications

```bash
# Open web browser

# Admin Dashboard
http://localhost:5000

# Login
# Email: admin@university.edu
# Password: (from bootstrap step)

# Student Portal
http://localhost:5001

# Self-register or login with demo student credentials:
# Reg No: CS21001
# Password: (from seed_demo_data output)
```

### Step 5: Test Face Detection

```bash
# Use camera to verify face detection works

# In Admin Dashboard:
# 1. Navigate to "Live Camera"
# 2. Start Session (course CS101)
# 3. Point camera at your face
# 4. Verify bounding box appears around face
# 5. After 2–3 frames, should mark "Present"

# Watch browser console for errors
# Open DevTools (F12) → Console tab
```

### Step 6: Test Student Self-Enrollment

```bash
# In Student Portal:
# 1. Click "Self-Enrollment"
# 2. Capture 5 different face samples (different angles)
# 3. System auto-approves if quality score ≥ 85
# 4. Student can now be recognized in future sessions
```

### Step 7: Verify Attendance Mark

```bash
# In Admin Dashboard:
# 1. Navigate to "Attendance Records"
# 2. Filter by date (today)
# 3. Verify your face appears in attendance list
# 4. Check confidence score (should be > 0.9)

# Or via MongoDB shell:
mongosh
> use attendance_system
> db.attendance.find({ date: "2024-09-15" })
# Should show one record with your name
```

---

## Verification Checklist

### ✓ Development Environment

- [ ] Python 3.12 installed: `python --version`
- [ ] Virtual environment activated: `which python` shows venv path
- [ ] All packages installed: `pip list | grep opencv-contrib-python`
- [ ] Models downloaded: `ls -la models/face_detection_yunet_2023mar.onnx`
- [ ] `.env` file created with correct `MONGODB_URI`

### ✓ Database Setup

- [ ] MongoDB running: `mongosh --eval "db.version()"`
- [ ] Collections created: `mongosh --eval "use attendance_system; show collections"`
- [ ] Indexes created: `mongosh --eval "use attendance_system; db.attendance.getIndexes()"`
- [ ] Admin user created: Login succeeds on admin dashboard

### ✓ Application Startup

- [ ] Admin app starts: `python run_admin.py` (http://localhost:5000 loads)
- [ ] Student app starts: `python run_student.py` (http://localhost:5001 loads)
- [ ] No Python exceptions in console
- [ ] Browser console (DevTools F12) has no JavaScript errors

### ✓ ML Pipeline

- [ ] YuNet model loads: No "model not found" errors in logs
- [ ] ArcFace model loads: First face detection triggers embedding generation
- [ ] Silent-Face model loads: Liveness check runs (no "graceful degradation" warning)
- [ ] Face detection works: Bounding box appears around faces in camera feed

### ✓ Database Operations

- [ ] Student can enroll: Self-enrollment form submits successfully
- [ ] Attendance mark recorded: After 2–3 frames, MongoDB shows attendance entry
- [ ] No duplicate marks: Multiple frames of same face only mark once per day
- [ ] Query works: `db.attendance.find()` returns records

### ✓ First Session Completion

Checklist:
```bash
# 1. Create session
curl -X POST http://localhost:5000/api/attendance/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "lab-1",
    "course_id": "CS101"
  }'
# Should return session_id

# 2. Verify attendance marked
curl http://localhost:5000/api/attendance \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-09-15"}'
# Should return attendance records

# 3. Close session
curl -X POST http://localhost:5000/api/attendance/sessions/<SESSION_ID>/end

# 4. Export report
curl http://localhost:5000/api/attendance/export \
  -d '{"date": "2024-09-15", "format": "csv"}' \
  -o attendance_report.csv
```

---

## Troubleshooting First Setup

### Issue: "YuNet model not found"

**Cause**: Models not downloaded.

**Fix**:
```bash
python scripts/download_models.py
# If fails, download manually:
# cd models && wget https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

### Issue: "MongoDB connection refused"

**Cause**: MongoDB not running or wrong URI.

**Fix**:
```bash
# Check if running (local)
sudo systemctl status mongod

# Start if not running
sudo systemctl start mongod

# For Atlas (cloud), verify connection string in .env:
# MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/attendance_system
```

### Issue: "Cannot open camera"

**Cause**: Camera not found or permission denied.

**Fix**:
```bash
# Linux: Check camera permissions
ls -la /dev/video0  # Should be readable by user

# macOS: Grant browser permission (Safari/Chrome settings)

# Windows: Allow app through firewall

# Or test with OpenCV directly
python -c "import cv2; cap = cv2.VideoCapture(0); print(f'Camera: {cap.isOpened()}')"
```

### Issue: "Low FPS (< 5 FPS)"

**Cause**: CPU-only processing, too many faces, or frame processing width too high.

**Fix**:
```bash
# Reduce detection interval
export ATTENDANCE_DETECTION_INTERVAL=8

# Reduce frame size
export ATTENDANCE_FRAME_PROCESS_WIDTH=384

# Enable GPU (if available)
export ATTENDANCE_ENABLE_GPU=1

# Restart app
python run_admin.py
```

---

## Next Steps

After successful setup:

1. **Customize Configuration**: Adjust [RECOGNITION_THRESHOLD](../core/config.py), detection interval per your environment.
2. **Batch Enroll Students**: Use [scripts/bootstrap_admin.py](../scripts/bootstrap_admin.py) to load student list from CSV.
3. **Deploy to Production**: Follow [DEPLOYMENT.md](DEPLOYMENT.md) for Docker + Gunicorn setup.
4. **Monitor & Debug**: Use [core/profiling.py](../core/profiling.py) for performance analysis.
5. **Read Architecture**: Review [ARCHITECTURE.md](ARCHITECTURE.md) for design patterns and module organization.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
