# Setup & Installation Guide: Complete Step-by-Step

Comprehensive installation instructions for AutoAttendance on Windows, Linux, and macOS with troubleshooting.

---

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 20 GB | 50 GB |
| **GPU** | Optional | NVIDIA CUDA 11.0+ |
| **Python** | 3.8 | 3.9-3.11 |
| **OS** | Windows 10, Ubuntu 20.04, macOS 10.14+ | Latest LTS |

### Install Python

**Windows**:
```
1. Download from https://www.python.org (3.9+)
2. Run installer
3. ✓ Check "Add Python to PATH"
4. Choose "Install for all users"
5. Click Install
```

**Linux**:
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
```

**macOS**:
```bash
brew install python@3.9
# Or download from https://www.python.org
```

**Verify installation**:
```bash
python --version    # Should show Python 3.9+
pip --version       # Should show pip 21.0+
```

### Install Git

**Windows**:
Download from https://git-scm.com

**Linux**:
```bash
sudo apt install git
```

**macOS**:
```bash
brew install git
```

---

## Part 1: Clone Repository

```bash
# Navigate to desired location
cd d:\Projects  # Windows
cd ~/Projects   # Linux/macOS

# Clone repository
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system

# Verify clone
ls -la  # Should show: recognition/, anti_spoofing/, models/, etc.
```

---

## Part 2: Create Python Virtual Environment

**Why virtual environment?**
- Isolates dependencies
- Prevents conflicts with system Python
- Makes reproduction easier

### Windows

```bash
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# You should see (venv) in command prompt
```

### Linux/macOS

```bash
# Create virtual environment
python3.9 -m venv venv

# Activate
source venv/bin/activate

# You should see (venv) in terminal
```

**Deactivate** (when done working):
```bash
deactivate
```

---

## Part 3: Install Dependencies

### Option A: CPU-Only (Easiest)

```bash
# Ensure virtual environment is activated
# (venv) should appear in terminal

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt contains**:
```
Flask==3.0.0
Flask-SocketIO==5.3.4
opencv-python==4.8.0
numpy==1.24.3
pandas==2.0.3
pymongo==4.4.1
onnxruntime==1.16.0
onnx==1.14.0
Pillow==10.0.0
python-socketio==5.9.0
```

**Installation time**: 2-5 minutes

### Option B: GPU Support (NVIDIA CUDA)

```bash
# Install CUDA-enabled versions
pip install -r requirements/gpu.txt
```

**gpu.txt contains**:
```
# Same as CPU, but:
onnxruntime-gpu==1.16.0     # GPU ONNX Runtime
torch==2.0.0                # PyTorch with CUDA
```

**Requires**:
- NVIDIA GPU (GeForce RTX, Tesla, etc.)
- NVIDIA CUDA 11.0+
- NVIDIA cuDNN 8.0+

**Installation time**: 10-15 minutes (downloads GPU drivers)

### Option C: Development (Includes Testing Tools)

```bash
pip install -r requirements/dev.txt
```

**dev.txt adds**:
```
pytest==7.4.0
pytest-cov==4.1.0
black==23.7.0
flake8==6.1.0
mypy==1.5.0
```

---

## Part 4: Download Pre-trained Models

Models are downloaded automatically on first run, but you can pre-download:

```bash
# Create models directory
mkdir models/

# Download YuNet face detection (230 KB)
curl -L https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o models/face_detection_yunet_2023mar.onnx

# Download ArcFace embedding (370 MB)
# Contact: shubham.patra@school.edu for access link
# OR train your own using: scripts/train_arcface.py

# Download Silent-Face anti-spoofing model (80 MB)
# Clone Silent-Face-Anti-Spoofing/
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
cp Silent-Face-Anti-Spoofing/resources/anti_spoofing_models/2o3_liveness_model_15_0.525500_state_dict.pth models/
```

**Models directory structure**:
```
models/
├── face_detection_yunet_2023mar.onnx      (230 KB)
├── arcface_resnet100.onnx                 (370 MB)
└── anti_spoofing/
    └── liveness_model.pth                 (80 MB)
```

**Verify downloads**:
```bash
ls -lh models/  # Check file sizes
```

---

## Part 5: Configure Environment Variables

### Create .env file

**Windows** (`d:\Projects\attendance_system\.env`):
```env
# Flask
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=your-secret-key-here-change-in-production

# MongoDB
MONGO_URI=mongodb://localhost:27017
DB_NAME=attendance_db

# Paths
MODELS_DIR=models/
UPLOADS_DIR=uploads/

# Face Recognition Thresholds
RECOGNITION_THRESHOLD=0.38
MIN_CONFIDENCE=0.42
LIVENESS_THRESHOLD=0.50

# Camera
CAMERA_ID=0           # 0 = default camera
FRAME_SKIP_INTERVAL=3

# Features
TRACKER_REUSE_ENABLED=true
BLINK_DETECTION_ENABLED=true
```

**Linux/macOS**: Same as Windows

### Load environment variables

**Python automatically loads from .env**:
```python
from dotenv import load_dotenv
load_dotenv()

import os
mongo_uri = os.getenv('MONGO_URI')
```

---

## Part 6: Setup MongoDB

### Option A: Local MongoDB (Development)

**Windows**:
```
1. Download: https://www.mongodb.com/try/download/community
2. Run installer
3. Select "Install MongoDB Community Server"
4. Accept defaults
5. MongoDB runs as Windows Service
```

**Linux**:
```bash
sudo apt install mongodb
sudo systemctl start mongodb
sudo systemctl status mongodb
```

**macOS**:
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Verify running**:
```bash
mongosh  # Connect to MongoDB shell
> db.version()
> exit
```

### Option B: MongoDB Atlas (Cloud, Recommended for Production)

```
1. Go to https://www.mongodb.com/cloud/atlas
2. Sign up for free account
3. Create cluster
4. Get connection string: mongodb+srv://user:password@cluster.mongodb.net/
5. Update .env: MONGO_URI=...
```

**Create database and collections**:
```javascript
// In mongosh or MongoDB Compass
use attendance_db

// Create collections with schema validation
db.createCollection("students", {
  validator: {
    $jsonSchema: {
      required: ["student_id", "name", "email"],
      properties: {
        student_id: { bsonType: "string" },
        name: { bsonType: "string" },
        face_embeddings: { bsonType: "array" }
      }
    }
  }
})

db.createCollection("attendance")
db.createCollection("courses")
db.createCollection("users")
db.createCollection("security_logs")
```

---

## Part 7: Initialize Database

```bash
# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

# Run bootstrap script
python scripts/bootstrap_admin.py

# Creates:
# - Admin user: admin@school.edu / password
# - Sample courses: CS101, CS102
# - Sample students: 10 demo students
```

**Output**:
```
✓ Created admin user
✓ Created 3 courses
✓ Created 10 demo students
✓ Database ready
```

---

## Part 8: Start Application

### Development Server

```bash
# Activate virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Run application
python app.py
```

**Output**:
```
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
 * SocketIO server started
 * Connected to MongoDB
```

**Visit in browser**: http://localhost:5000

### Production Server (Gunicorn)

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn (4 workers)
gunicorn -c gunicorn.conf.py app:app

# Output:
# [2025-01-20 10:30:00 +0000] [1234] [INFO] Listening at: http://127.0.0.1:8000
```

### Docker (Optional)

```bash
# Build image
docker build -f docker/Dockerfile -t attendance_system:latest .

# Run container
docker run -p 5000:5000 \
  -e MONGO_URI=mongodb://host.docker.internal:27017 \
  -v /path/to/models:/app/models \
  attendance_system:latest

# Access: http://localhost:5000
```

---

## Part 9: Test Installation

### 1. Check Python Environment

```bash
python -c "import cv2; print(cv2.__version__)"      # Should print: 4.8.0+
python -c "import onnxruntime; print(onnxruntime.get_device())"  # Should print: CPU/CUDA
python -c "from pymongo import MongoClient; print('MongoDB OK')"  # Should print: MongoDB OK
```

### 2. Test Face Detection

```bash
python scripts/smoke_test.py
```

**Output**:
```
Testing Face Detection...
✓ YuNet loaded: 230 KB
✓ Detected faces in test image: 1
✓ Detection confidence: 0.92

Testing Face Recognition...
✓ ArcFace loaded: 370 MB
✓ Generated embedding: (512,)
✓ Embedding norm: 1.00

Testing Liveness Detection...
✓ Silent-Face CNN loaded: 80 MB
✓ Liveness confidence: 0.87

All tests passed!
```

### 3. Test Web Server

```bash
# In another terminal
curl http://localhost:5000/health

# Output:
# {"status":"ok","models":"loaded","database":"connected"}
```

### 4. Test Enrollment (UI)

```
1. Open http://localhost:5000
2. Click "Student" → "Enroll"
3. Allow camera access
4. Capture 5+ face photos
5. Click "Submit"
6. Check: http://localhost:5000/api/students
```

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'cv2'"

**Cause**: OpenCV not installed

**Solution**:
```bash
# Ensure virtual environment is activated
pip install opencv-python --no-cache-dir

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue 2: "MongoDB connection refused"

**Cause**: MongoDB not running

**Solution**:
```bash
# Windows: Check Services
# Services → MongoDB Community Server → Start

# Linux:
sudo systemctl start mongodb

# macOS:
brew services start mongodb-community

# Verify:
mongosh
```

### Issue 3: "No module named 'onnxruntime'"

**Cause**: ONNX Runtime not installed

**Solution**:
```bash
pip install onnxruntime --no-cache-dir

# For GPU:
pip install onnxruntime-gpu
```

### Issue 4: "CUDA out of memory"

**Cause**: GPU memory insufficient

**Solution**:
```python
# Edit core/config.py
INFERENCE_BATCH_SIZE = 1  # Reduce from default
FRAME_SKIP_INTERVAL = 5   # Process fewer frames
```

### Issue 5: "Camera not found"

**Cause**: Camera not accessible or wrong device ID

**Solution**:
```bash
# List available cameras
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different ID
# In .env:
CAMERA_ID=1  # or 2, 3, etc.
```

### Issue 6: "Models not found"

**Cause**: Model files not downloaded

**Solution**:
```bash
# Download models manually
cd models/

# YuNet
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Verify
ls -lh
```

---

## Verification Checklist

```
✓ Python 3.9+ installed
✓ Virtual environment created
✓ Dependencies installed (pip list | grep Flask)
✓ Models downloaded (ls models/)
✓ MongoDB running (mongosh)
✓ .env file configured
✓ Database initialized (scripts/bootstrap_admin.py)
✓ App starts (python app.py)
✓ Web server responds (http://localhost:5000)
✓ Smoke tests pass (python scripts/smoke_test.py)
✓ Camera accessible
```

---

## Next Steps

### 1. Enroll Students

```bash
# Manually enroll via UI
# http://localhost:5000/enroll

# Or seed demo data
python scripts/seed_demo_data.py
```

### 2. Calibrate Thresholds

```bash
python scripts/calibrate_liveness_threshold.py
```

### 3. Test Attendance Marking

```bash
# Via UI
http://localhost:5000/mark-attendance

# Or API
curl -X POST http://localhost:5000/api/attendance \
  -H "Content-Type: application/json" \
  -d '{"student_id":"STU2025001","course_id":"CS101"}'
```

---

## Folder Structure After Setup

```
attendance_system/
├── venv/                  ← Virtual environment (created)
├── models/                ← Pre-trained models (downloaded)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── arcface_resnet100.onnx
│   └── anti_spoofing/...
├── .env                   ← Configuration (created)
├── app.py
├── requirements.txt       ← Dependencies (installed)
└── scripts/
    ├── bootstrap_admin.py ← Database init
    ├── smoke_test.py      ← Test script
    └── ...
```

---

## Support

**For issues**:
1. Check this guide → Troubleshooting section
2. Review logs: `logs/` directory
3. Check GitHub Issues: https://github.com/ShubhamPatra/attendance_system/issues
4. Contact: shubham.patra@school.edu

---

## References

1. Python Virtual Environments: https://docs.python.org/3/tutorial/venv.html
2. MongoDB Installation: https://docs.mongodb.com/manual/installation/
3. OpenCV Installation: https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html
4. ONNX Runtime: https://onnxruntime.ai/docs/install/
