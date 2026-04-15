# AutoAttendance - Smart Face Recognition Attendance System

**AutoAttendance** is an intelligent attendance system that uses facial recognition to automatically track attendance. It is designed for schools, colleges, and offices where manual attendance is time-consuming and error-prone.

The system provides:
- Automatic face recognition and identification
- Attendance recording with audit trails
- Spoofing detection (rejects photos, videos, and masks)
- Real-time dashboards and reporting
- Multi-camera support

## 🐍 Supported Versions

**Version-Independent Design**: AutoAttendance works on any machine without breaking the code.

| Component | Support |
|-----------|---------|
| **Python** | 3.9, 3.10, 3.11, 3.12, 3.13+ |
| **Operating Systems** | Windows 7+, macOS 10.13+, Ubuntu 18.04+, Debian 10+ |
| **Deployment** | Docker (CPU/GPU), Local Development, Kubernetes, Cloud |
| **Hardware** | CPU or NVIDIA GPU (CUDA 11.8+, CUDA 12.x) |

**Quick Start:**
```bash
# Local (any Python 3.9+)
pip install -e .
python scripts/verify_versions.py

# Docker (CPU)
docker compose up --build

# Docker (GPU)
INSTALL_GPU=1 docker compose up --build

# Specific Python Version
PYTHON_VERSION=3.11 docker compose up --build
```

📖 **[Full Installation Guide](docs/INSTALLATION.md)** - Detailed setup for each Python version and OS.

---

**Traditional attendance methods are slow:**
- Teachers spend time calling out names
- Students may not be physically present but marked present
- Manual entry creates errors
- No real-time visibility

**AutoAttendance solves this by:**
- Marking attendance automatically via face recognition
- Real-time reporting (see who's present right now)
- Anti-spoofing to prevent cheating (detects if someone shows a photo)
- Complete audit trail of attendance records
- Integration with admin dashboards and student portals

---

## How Does It Work? (The Simple Way)

### The Basic Flow

```
Camera capture → Face detection → Liveness check → 
Face recognition → Attendance recording
```

### Step-by-Step Process

1. **Camera Capture**
   - Captures video frames from webcam or IP camera
   - Runs continuously in real-time

2. **Face Detection (YuNet)**
   - Detects all faces in the frame
   - Processes only new faces not previously seen
   - Ignores faces that are too small or blurry

3. **Liveness Check (Anti-Spoofing)**
   - Verifies the person is real, not a photo or video
   - Uses deep learning to detect spoofing attempts
   - Collects multiple predictions over 8 frames

4. **Face Recognition (ArcFace)**
   - Extracts unique face features (embeddings)
   - Compares against stored student database
   - Matches if similarity score exceeds threshold

5. **Attendance Recording**
   - Records student ID, timestamp, and confidence score
   - Updates MongoDB database
   - Sends real-time update to dashboard

6. **Dashboard & Reporting**
   - Live dashboard displays attendance in real-time
   - Analytics show daily, weekly, and monthly trends
   - Reports available for download

---

## Technologies & Models Used

### Computer Vision Models

| Component | What It Does | Technology | Why? |
|-----------|-------------|-----------|------|
| **Face Detection** | Finds faces in video | YuNet ONNX | Fast (runs on CPU), accurate on different angles and lighting |
| **Face Recognition** | Identifies who the person is | ArcFace + InsightFace | Industry standard, very accurate, fast inference |
| **Face Alignment** | Rotates/aligns face to standard position | 5-point keypoints | Improves recognition accuracy |
| **Anti-Spoofing** | Detects real vs fake/photo | Silent-Face-Anti-Spoofing + PyTorch | Prevents attendance fraud with photos/videos |
| **Face Tracking** | Follows face across frames | CSRT (or KCF for speed) | Reduces processing load, smooth tracking |

### Backend Technologies

| Component | Purpose |
|-----------|---------|
| **Flask** | Web framework for dashboards and APIs |
| **MongoDB Atlas** | Cloud database for all data (students, attendance, etc.) |
| **OpenCV** | Image processing and computer vision |
| **SocketIO** | Real-time updates to browser (live feed) |
| **Docker** | Container for easy deployment |
| **Nginx** | Web server and load balancer |

---

## Understanding Anti-Spoofing (How We Stop Cheating)

Anti-spoofing is the system that prevents someone from showing a photo or video of another person to mark attendance.

### How Spoofing Attacks Work

```
Common attack methods:
- Show a printed photo of the student
- Display a video recording on a phone/tablet
- Use a 3D mask or mannequin head
- Hold up someone's ID photo
```

### How AutoAttendance Detects Spoofing

**Silent-Face-Anti-Spoofing Uses Deep Learning:**

1. **Texture Analysis**
   - Real faces have complex textures (skin pores, light reflection)
   - Photos have flat, repetitive patterns
   - Deep learning model learns to distinguish these

2. **Motion Analysis**
   - Real faces move naturally
   - Photos/videos have unnatural motion patterns
   - System analyzes movement over multiple frames

3. **3D Structure**
   - Real faces have 3D depth
   - Photos are 2D flat images
   - AI learns to detect the difference

4. **Voting System**
   - Collects 8 predictions over time
   - Requires majority vote before accepting
   - More confident = harder to fool

### Example Decision Logic

```
Frame 1: Confidence 95% -> REAL
Frame 2: Confidence 92% -> REAL
Frame 3: Confidence 88% -> REAL
Frame 4: Confidence 91% -> REAL
...
Vote Result: 7 out of 8 votes for REAL -> ACCEPT
```

---

## Key Features Explained

### 1. **Real-Time Face Recognition**
- Recognizes students instantly
- Works with multiple cameras simultaneously
- Handles different angles, lighting, and expressions

### 2. **Anti-Spoofing Protection**
- Detects and rejects spoofing attempts (photos, videos, masks)
- Multi-frame voting for high confidence
- Configurable strictness levels

### 3. **Live Dashboard**
- Real-time attendance display
- Live updates via WebSockets
- Attendance trend charts
- Heatmaps showing peak attendance periods

### 4. **Admin Panel**
- Manage student database
- Batch import students from CSV + images
- Edit attendance records
- Generate reports and exports

### 5. **Student Self-Service Portal**
- Students can create accounts
- Self-registration with photo upload
- Check personal attendance history
- Download attendance certificates

### 6. **Analytics & Reporting**
- Daily/weekly/monthly attendance reports
- Identify at-risk students (low attendance)
- Hourly attendance trends
- Export to CSV for further analysis

### 7. **Multi-Camera Support**
- Run multiple cameras simultaneously
- Per-camera diagnostics dashboard
- Independent tracking per camera
- No conflicts between cameras

### 8. **Quality Gate System**
- Rejects blurry face captures
- Rejects faces that are too small
- Rejects overexposed/underexposed images
- Ensures high-quality attendance records

---

## Architecture Overview

### System Diagram

```
                        ┌─────────────────────────────────┐
                        │     ADMIN & STUDENT BROWSERS    │
                        │  (Dashboard, Reports, Portal)   │
                        └──────────────┬──────────────────┘
                                       │
                        ┌──────────────▼──────────────────┐
                        │  FLASK WEB APPLICATION          │
                        │  (Routes, API, WebSocket)       │
                        ├─────────────────────────────────┤
                        │ SocketIO: Real-time updates    │
                        │ REST APIs: Data access         │
                        │ Auth: Login/RBAC (optional)    │
                        └──────────────┬──────────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
    ┌───────────▼──────────┐  ┌────────▼────────┐  ┌────────▼────────┐
    │  VISION PIPELINE     │  │    DATABASE     │  │  NOTIFICATIONS  │
    │  (Per Camera)        │  │   MongoDB       │  │  (Email alerts) │
    ├──────────────────────┤  │                 │  │                 │
    │ 1. Capture frames    │  │ Collections:    │  │ Dry-run events  │
    │ 2. Detect faces      │  │ - students      │  │ for absences    │
    │    (YuNet)           │  │ - attendance    │  │                 │
    │ 3. Track faces       │  │ - users         │  │                 │
    │    (CSRT/KCF)        │  │ - logs          │  │                 │
    │ 4. Check liveness    │  │                 │  │                 │
    │    (Anti-spoofing)   │  │                 │  │                 │
    │ 5. Extract features  │  │                 │  │                 │
    │    (ArcFace)         │  │                 │  │                 │
    │ 6. Match identity    │  │                 │  │                 │
    │ 7. Record attendance │  │                 │  │                 │
    └─────────────────────┘  └─────────────────┘  └──────────────────┘
         │ (Multiple)
         │
    ┌────▼──────────────────────────────┐
    │  CAMERAS (Webcams / IP Cameras)   │
    │  Camera 1, Camera 2, Camera 3...  │
    └───────────────────────────────────┘
```

### Per-Frame Processing Flow

```
Frame arrives
     │
     ├─→ Update existing trackers (lightweight, ~5ms)
     │
     ├─→ Detect motion?
     │   ├─ YES → Run YuNet detection
     │   └─ NO → Skip detection
     │
     ├─→ Associate detections to tracks
     │
     ├─→ For NEW tracks only:
     │   ├─→ Check face quality (blur, brightness, size)
     │   ├─→ Run anti-spoofing check (collect 8 votes)
     │   ├─→ Extract face embedding (ArcFace)
     │   ├─→ Find match in database
     │   └─→ Record attendance if match found
     │
     └─→ Draw overlays and send to browser
```

### Processing Time Budget

```
Per frame (30 fps = 33ms per frame):
├─ Tracker update: ~5ms
├─ Motion detection: ~2ms
├─ Face detection (YuNet): ~15ms (only every N frames)
├─ Anti-spoofing: ~50ms (only for new faces)
├─ Embedding extraction: ~20ms (only for new faces)
└─ Total: Targets ~100ms, but parallelized

Result: ~25-30 fps with 1-2 tracked faces
```

---



## Installation & Setup Guide

### Prerequisites

Before starting, ensure you have:
- Windows, Mac, or Linux computer
- Python 3.11 ([Download here](https://www.python.org/downloads/))
- A connected webcam or camera
- MongoDB Atlas account (free tier available at [mongodb.com/cloud](https://www.mongodb.com/cloud))
- At least 2GB free disk space
- At least 4GB RAM

### Step 1: Download the Project

```bash
# Clone the repository
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system
```

### Step 2: Create Python Virtual Environment

A virtual environment isolates this project's Python packages from your system.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- Flask (web framework)
- OpenCV (computer vision)
- PyMongo (database)
- PyTorch (for anti-spoofing)
- And 20+ others...

### Step 4: Setup MongoDB Atlas (Cloud Database)

**Why MongoDB?** It stores all your data - students, attendance records, user accounts, etc.

1. Go to [mongodb.com/cloud](https://www.mongodb.com/cloud)
2. Create a FREE account
3. Create a cluster (M0 free tier)
4. Create a database user
5. Get your connection string (looks like: `mongodb+srv://user:password@cluster.mongodb.net/...`)

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# On Windows
copy .env.example .env

# On Mac/Linux
cp .env.example .env
```

Edit the `.env` file and set these **essential variables**:

```env
# Database Connection (from Step 4)
MONGO_URI=mongodb+srv://YOUR_USERNAME:YOUR_PASSWORD@YOUR_CLUSTER.mongodb.net/attendance_system?retryWrites=true&w=majority

# Security (change this to something random!)
SECRET_KEY=my-super-secret-key-12345

# Basic settings
APP_HOST=0.0.0.0
APP_PORT=5000
APP_DEBUG=0
EMBEDDING_BACKEND=arcface

# Important for local development
ENABLE_RBAC=0
ENABLE_RESTX_API=0
STRICT_STARTUP_CHECKS=1
STARTUP_CAMERA_PROBE=0
```

**Optional configurations** (advanced):
- `LIVENESS_CONFIDENCE_THRESHOLD=0.55` - How strict anti-spoofing is (higher = stricter)
- `RECOGNITION_THRESHOLD=0.50` - How similar faces need to be (higher = stricter)
- `CAMERA_INDICES=0` - Which camera to use (0=default, 1=second camera, etc.)

### Step 6: Download AI Models

The AI models (YuNet for detection, ArcFace for recognition) are large files.

```bash
python scripts/download_models.py --skip-insightface
```

This downloads ~500MB of models. Takes 2-5 minutes depending on internet speed.

### Step 7: Verify Installation

```bash
python scripts/smoke_test.py --base-url http://localhost:5000
```

If all checks pass, the installation is complete and ready to use.

---

## Running the Application

### Quick Start (Recommended for Beginners)

This runs BOTH admin dashboard and student portal at once:

```bash
python run.py
```

Access in your browser:
- **Admin Dashboard:** http://localhost:5000
- **Student Portal:** http://localhost:5001

### Or Run Separately (Advanced)

**Terminal 1 - Admin Panel:**
```bash
python run_admin.py
```

**Terminal 2 - Student Portal:**
```bash
python run_student.py
```

**Terminal 3 - Separate Camera/Processing (if needed):**
```bash
python -m scripts.debug_pipeline
```

### Using Docker (Container Deployment)

If you have Docker installed:

```bash
# Start all services (web, student portal, nginx)
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

Access via `http://localhost` (Nginx routes to the apps).

---

## Initial Setup: Creating Your First Admin Account

Before marking attendance, you need an admin account.

### Create Admin User

```bash
python scripts/bootstrap_admin.py --username admin --role admin
```

When prompted, enter a strong password (8+ characters).

### Or: Use Demo Data (Testing Only)

To quickly test with sample data:

```bash
python scripts/seed_demo_data.py
```

This creates:
- 3 demo students (Alice, Bilal, Sara)
- Default admin: `admin` / `admin1234` (password)
- Sample attendance records

**Note:** Demo data is for testing only. Change the password before production use.

### Clear Database (Remove All Data)

To wipe the database and start fresh:

```bash
python -m scripts.clear_db
```

This drops all collections. Use with caution!

---

## How to Use the System

### For Teachers/Admins

1. **Login:**
   - Go to http://localhost:5000
   - Username/password (created with bootstrap_admin.py)

2. **Add Students:**
   - Click "Admin" → "Student Management"
   - Option A: Single student (name, registration, section, email)
   - Option B: Batch import (upload CSV + ZIP with photos)

3. **Start Recording Attendance:**
   - Click "Attendance" → "Live Camera"
   - Camera shows in real-time
   - Green boxes = recognized students
   - Attendance auto-recorded

4. **View Reports:**
   - Click "Report" to see attendance summary
   - Download CSV for Excel analysis
   - Click "Analytics" for trends

5. **Manage Attendance:**
   - Edit/delete records in admin panel
   - Mark absences manually if needed

### For Students (Self-Service Portal)

1. **Register:**
   - Go to http://localhost:5001/student/login
   - Click "Register"
   - Upload clear face photo
   - Provide email/details

2. **Check Attendance:**
   - Login with email
   - View personal attendance history
   - See daily/weekly stats

3. **Download Certificate:**
   - View attendance report
   - Download certificate showing attendance percentage

---

## Configuration Deep Dive

### Face Recognition Settings

```env
# Similarity threshold (0-1): How similar faces need to be
RECOGNITION_THRESHOLD=0.50         # Default: 50% similar = match

# Confidence score: How certain the algorithm is
RECOGNITION_MIN_CONFIDENCE=0.55    # Must be 55%+ confident

# Distance gap: Space between top matches
RECOGNITION_MIN_DISTANCE_GAP=0.04  # Larger gap = clearer winner

# How many frames to confirm
RECOGNITION_CONFIRM_FRAMES=2       # Confirm match in 2 consecutive frames
```

### Anti-Spoofing (Liveness) Settings

```env
# Voting system: Collect votes over frames
LIVENESS_HISTORY_SIZE=8            # Collect 8 predictions
LIVENESS_MIN_HISTORY=3             # Need at least 3 votes

# Confidence thresholds
LIVENESS_CONFIDENCE_THRESHOLD=0.55        # Base threshold
LIVENESS_REAL_FAST_CONFIDENCE=0.72        # Fast accept if very confident
LIVENESS_SPOOF_CONFIDENCE_MIN=0.6         # Spoof must be confident
LIVENESS_STRONG_SPOOF_CONFIDENCE=0.85     # Really strong spoof = reject

# Voting ratios
LIVENESS_REAL_VOTE_RATIO=0.7       # 70% must say REAL
LIVENESS_SPOOF_VOTE_RATIO=0.6      # 60% must say SPOOF
```

**Example - How Strict Are You?**

| Use Case | LIVENESS_CONFIDENCE_THRESHOLD | LIVENESS_REAL_VOTE_RATIO |
|----------|-------------------------------|--------------------------|
| Testing  | 0.40 (very lenient)          | 0.5 (any majority)        |
| School   | 0.55 (default)                | 0.7 (strict)              |
| Security | 0.75 (very strict)            | 0.9 (almost all agree)    |

### Performance Tuning

```env
# Frame scaling: Lower = faster, less accurate
PERF_FRAME_SCALE=1.0               # 1.0 = full resolution, 0.5 = half

# Max faces tracked simultaneously
PERF_MAX_FACES=5                   # Track up to 5 faces per frame

# Which tracker to use
PERF_USE_KCF_TRACKER=0             # 0=CSRT (accurate), 1=KCF (3x faster)

# JPEG compression for streaming
PERF_JPEG_QUALITY=80               # 0-100 (lower = faster)

# Detection intervals
DETECTION_INTERVAL_MIN=3           # Min frames between detections
DETECTION_INTERVAL_MAX=15          # Max frames between detections
TRACK_DETECTOR_MISS_TOLERANCE=2    # Frames before losing track
```

### Face Quality Requirements

```env
# Minimum face size (pixels)
MIN_FACE_SIZE_PIXELS=20             # Faces smaller than 20x20 rejected

# Minimum blurriness (higher = sharper required)
BLUR_THRESHOLD=100                  # Laplacian variance threshold

# Brightness range (0-255)
BRIGHTNESS_THRESHOLD=40             # Too dark
BRIGHTNESS_MAX=250                  # Too bright
```

### Advanced Settings

```env
# Incremental learning: Update embeddings automatically
INCREMENTAL_LEARNING_CONFIDENCE=0.92        # Only learn if very confident
MAX_ENCODINGS_PER_STUDENT=15                # Store up to 15 face embeddings

# Rate limiting (prevent API abuse)
API_RATE_LIMIT_WINDOW_SEC=60        # Per minute
API_RATE_LIMIT_MAX_REQUESTS=30      # Max 30 requests per minute

# Absence notifications
ABSENCE_THRESHOLD=75                # Alert if attendance < 75%

# Storage retention
BACKUP_RETENTION_DAYS=30            # Keep backups 30 days
UPLOAD_RETENTION_SECONDS=3600       # Keep upload cache 1 hour
```

---

## API Endpoints Reference

### Attendance APIs

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/attendance_activity` | GET | Hourly attendance counts |
| `/api/heatmap` | GET | Attendance by day/time |
| `/api/analytics/trends` | GET | Daily trend data |
| `/api/analytics/at_risk` | GET | Low-attendance students |

### Data APIs

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/registration_numbers` | GET | List all students |
| `/api/metrics` | GET | System performance metrics |
| `/api/events` | GET | Recent attendance events |
| `/api/logs` | GET | System logs |

### System APIs

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System health check |
| `/ready` | GET | Readiness check |
| `/api/cameras` | GET | Camera diagnostics |

### Usage Example

```bash
# Get attendance trends
curl http://localhost:5000/api/analytics/trends

# Get system metrics
curl http://localhost:5000/api/metrics

# Get at-risk students
curl http://localhost:5000/api/analytics/at_risk
```

---



## Understanding the Database

### What is MongoDB?

**Simple explanation:** MongoDB is like an Excel spreadsheet, but digital, cloud-hosted, and much more powerful.

- **Collections** = Sheets (one for students, one for attendance, etc.)
- **Documents** = Rows (each student is one document)
- **Fields** = Columns (name, email, etc.)

### Data Collections Explained

#### 1. **Students Collection**

Stores information about every student.

```
{
  "_id": "507f1f77bcf86cd799439011",      # Unique ID (auto-generated)
  "name": "Alice Khan",                    # Student name
  "semester": 3,                           # Current semester/year
  "registration_number": "REG-001",        # Unique registration ID
  "section": "A",                          # Class section
  "email": "alice@example.com",            # Student email
  "encodings": [                           # Multiple face embeddings (for robustness)
    [0.234, 0.567, 0.123, ...],           # Face embedding 1 (128 numbers)
    [0.245, 0.578, 0.134, ...],           # Face embedding 2
    ...                                    # Up to 15 embeddings per student
  ],
  "verification_status": "approved",       # pending, approved, or rejected
  "created_at": "2024-04-15T10:30:00Z",  # When record created
  "approved_at": "2024-04-15T11:00:00Z"  # When admin approved
}
```

#### 2. **Attendance Collection**

Records each attendance event.

```
{
  "_id": "507f1f77bcf86cd799439012",
  "student_id": "507f1f77bcf86cd799439011",    # Links to students table
  "date": "2024-04-15",                         # YYYY-MM-DD format
  "time": "09:30:45",                           # HH:MM:SS format
  "status": "Present",                          # Present or Absent
  "confidence_score": 0.92,                     # How confident (0-1): 92%
  "camera_index": 0,                            # Which camera recorded
  "recorded_at": "2024-04-15T09:30:45Z"       # When recorded
}
```

#### 3. **Users Collection**

Admin and teacher accounts.

```
{
  "_id": "507f1f77bcf86cd799439013",
  "username": "admin",                    # Login username
  "password_hash": "bcrypt:$2b$12...",   # Encrypted password (never stored plain)
  "role": "admin",                        # admin or teacher
  "email": "admin@example.com",           # Contact email
  "is_active": true,                      # Can this user login?
  "created_at": "2024-04-15T08:00:00Z",  # When account created
  "last_login": "2024-04-15T09:00:00Z"   # Last login time
}
```

#### 4. **Attendance Sessions Collection**

Tracks camera sessions and real-time attendance.

```
{
  "_id": "507f1f77bcf86cd799439014",
  "camera_index": 0,                      # Which camera
  "session_start": "2024-04-15T09:00:00Z",# When session started
  "session_end": "2024-04-15T12:00:00Z",  # When session ended
  "attendance_count": 45,                 # How many marked present
  "status": "active"                      # active or closed
}
```

#### 5. **System Logs Collection**

Tracks system events for debugging.

```
{
  "_id": "507f1f77bcf86cd799439015",
  "timestamp": "2024-04-15T09:30:45Z",
  "event_type": "attendance_recorded",    # Type of event
  "student_id": "507f1f77bcf86cd799439011",
  "message": "Alice Khan marked present",
  "level": "info"                          # info, warning, error
}
```

---

## Web Pages & Routes

### Admin Dashboard Routes

```
http://localhost:5000/
├── /               Landing/home page
├── /login          Staff login
├── /dashboard      Main dashboard with attendance overview
├── /attendance     Live camera feed for marking attendance
├── /register       Register new student (admin form)
├── /register/batch Batch import (CSV + ZIP photos)
├── /admin/students Student management (edit, delete, view)
├── /report         Attendance reports and export
├── /logs           System logs viewer
├── /metrics        Real-time performance metrics
├── /heatmap        Attendance heatmap (by time/date)
├── /attendance_activity  Hourly attendance chart
└── /logout         Logout
```

### Student Portal Routes

```
http://localhost:5001/
├── /student/login              Student login
├── /student/register           Student self-registration
├── /student/register/upload    Upload face photo
├── /student/dashboard          Personal attendance dashboard
├── /student/attendance         Personal attendance history
├── /student/certificate        Download attendance certificate
└── /student/logout             Logout
```

---

## Batch Import: Adding Multiple Students

### Use Case
You have 100 students and their photos. Instead of uploading one-by-one, batch import does it in seconds.

### How to Prepare Files

**Step 1: Create CSV file** (`students.csv`)

```csv
registration_number,name,semester,section,email
REG-001,Alice Khan,3,A,alice@example.com
REG-002,Bilal Ahmed,3,A,bilal@example.com
REG-003,Sara Iqbal,5,B,sara@example.com
REG-004,John Smith,4,A,john@example.com
```

**Required columns:**
- `registration_number` - Unique ID (no spaces/special chars)
- `name` - Full name
- `semester` - Year/Level (1, 2, 3, etc.)
- `section` - Class section (A, B, C, etc.)
- `email` - Student email

**Step 2: Collect photos**

Create a folder with photos named exactly as registration numbers:

```
photos/
├── REG-001.jpg
├── REG-002.jpg
├── REG-003.jpg
└── REG-004.jpg
```

**Photo requirements:**
- Format: JPG or PNG
- Minimum size: 100x100 pixels
- Clear face visible
- Good lighting (not too dark)
- Filename must match registration_number

**Step 3: Create ZIP file**

Create a ZIP with both files:

```
batch.zip
├── students.csv
└── photos/
    ├── REG-001.jpg
    ├── REG-002.jpg
    └── ...
```

**Step 4: Upload via Admin Dashboard**

To import the batch:
1. Login as admin
2. Go to `/register/batch`
3. Select `batch.zip`
4. Click "Import"
5. Complete

---

## Troubleshooting Guide

### Problem: "MONGO_URI not set" Error

**Error message:**
```
EnvironmentError: MONGO_URI environment variable is not set
```

**Solution:**
1. Check your `.env` file exists
2. Verify it has `MONGO_URI=mongodb+srv://...`
3. Restart the application
4. If still failing:
   ```bash
   # Clear any system-level MONGO_URI
   # On Windows (PowerShell):
   Remove-Item env:MONGO_URI
   ```

### Problem: Cannot Connect to MongoDB

**Error message:**
```
ConnectionFailure: connection refused
```

**Solution:**
1. Check MongoDB Atlas cluster is running
2. Verify connection string in `.env`
3. Check firewall allows outgoing connections
4. Test connection:
   ```bash
   # Install mongosh if needed
   pip install pymongo
   
   # Test connection
   python -c "from pymongo import MongoClient; MongoClient('YOUR_MONGO_URI')"
   ```

### Problem: Camera Not Detected

**Error message:**
```
OpenCV: VIDEOIO ERROR or Cannot open camera device
```

**Solution:**
1. Check camera is connected and working
2. Try default camera:
   ```env
   CAMERA_INDICES=0
   ```
3. List available cameras:
   ```bash
   python scripts/debug_pipeline.py
   ```
4. Check OS permissions:
   - Mac: System Preferences → Security → Allow camera access
   - Windows: Settings → Privacy → Webcam

### Problem: Face Recognition Not Working

**Error message:**
```
No face recognized or low confidence scores
```

**Solutions:**

1. **Check lighting:**
   - Move camera near window/light source
   - Avoid backlighting

2. **Check face size:**
   - Face should be 20-30% of frame
   - Move closer to camera

3. **Lower threshold (try first):**
   ```env
   RECOGNITION_THRESHOLD=0.45  # Lower = more lenient
   ```

4. **Increase enrollment encodings:**
   ```env
   MAX_ENCODINGS_PER_STUDENT=15
   ```
   Then re-register student with multiple angles.

5. **Calibrate liveness threshold:**
   ```bash
   python scripts/calibrate_liveness_threshold.py
   ```

### Problem: Anti-Spoofing Too Strict/Lenient

**Error: Getting rejected or accepted too easily**

**Solution:**

1. **Reduce strictness (too strict):**
   ```env
   LIVENESS_CONFIDENCE_THRESHOLD=0.45
   LIVENESS_REAL_VOTE_RATIO=0.5
   ```

2. **Increase strictness (too lenient):**
   ```env
   LIVENESS_CONFIDENCE_THRESHOLD=0.75
   LIVENESS_REAL_VOTE_RATIO=0.9
   ```

3. **Calibrate for your environment:**
   ```bash
   python scripts/calibrate_liveness_threshold.py
   ```

### Problem: Database Getting Slow

**Solution:**

1. **Clear old data:**
   ```bash
   # Clear everything
   python -m scripts.clear_db
   
   # Or remove old backups
   ```

2. **Add database indexes:**
   - Indexes speed up queries
   - Automatically created on first run

3. **Archive old attendance:**
   - Export to CSV monthly
   - Clear old records from database

### Problem: Docker Container Issues

**Error: Container won't start**

**Solution:**

```bash
# Check logs
docker compose logs web

# Rebuild from scratch
docker compose down -v
docker compose up -d --build

# Clean up
docker system prune -a
```

### Problem: Permission Denied (Uploads/Logs)

**Error: Cannot write to uploads or logs folder**

**Solution:**

```bash
# Fix folder permissions
chmod 777 uploads/
chmod 777 logs/
chmod 777 unknown_faces/
```

### Getting Help

If you can't solve it:

1. **Check logs:**
   ```bash
   # View system logs
   tail -f logs/*.log
   ```

2. **Run smoke test:**
   ```bash
   python scripts/smoke_test.py --base-url http://localhost:5000
   ```

3. **Enable debug mode:**
   ```env
   DEBUG_MODE=1
   APP_DEBUG=1
   ```

4. **Check database:**
   ```bash
   # View MongoDB collections
   # Use MongoDB Atlas web UI: https://cloud.mongodb.com
   ```

---

## Performance & Optimization

### Recommended Settings by Use Case

**School/College (50+ students):**
```env
PERF_FRAME_SCALE=0.8             # Slightly faster
PERF_MAX_FACES=10                # Handle multiple students
RECOGNITION_THRESHOLD=0.50       # Balanced
LIVENESS_CONFIDENCE_THRESHOLD=0.55
PERF_USE_KCF_TRACKER=0           # Use CSRT (accurate)
```

**Office (5-20 employees, fast throughput):**
```env
PERF_FRAME_SCALE=1.0             # Full quality
PERF_MAX_FACES=5                 # Fewer people
RECOGNITION_THRESHOLD=0.45       # More lenient
LIVENESS_CONFIDENCE_THRESHOLD=0.45
PERF_USE_KCF_TRACKER=0           # High accuracy needed
```

**Security/High-Security (prevent spoofing at all costs):**
```env
PERF_FRAME_SCALE=1.0             # Full resolution
PERF_MAX_FACES=3                 # Single entry point
RECOGNITION_THRESHOLD=0.60       # Stricter
LIVENESS_CONFIDENCE_THRESHOLD=0.75 # Very strict
LIVENESS_REAL_VOTE_RATIO=0.9     # Need 90% agreement
```

**Fast Throughput (speed over accuracy):**
```env
PERF_FRAME_SCALE=0.5             # Half resolution
PERF_MAX_FACES=20                # Track many faces
RECOGNITION_THRESHOLD=0.40       # Very lenient
LIVENESS_CONFIDENCE_THRESHOLD=0.40
PERF_USE_KCF_TRACKER=1           # KCF tracker (3x faster)
PERF_JPEG_QUALITY=60             # Lower quality stream
```

---

## Making It Production-Ready

### Before Deploying to Production

- [ ] **Security:**
  - [ ] Set `ENABLE_RBAC=1` to require login
  - [ ] Change `SECRET_KEY` to random strong key
  - [ ] Use HTTPS (not HTTP)
  - [ ] Set proper firewall rules

- [ ] **Performance:**
  - [ ] Run on dedicated server (not laptop)
  - [ ] Use GPU acceleration if available
  - [ ] Configure `PERF_*` settings appropriately
  - [ ] Monitor CPU/memory usage

- [ ] **Data Backup:**
  - [ ] Enable MongoDB backups
  - [ ] Regular exports to CSV
  - [ ] Test restore procedure

- [ ] **Monitoring:**
  - [ ] Set up error alerts
  - [ ] Check logs regularly
  - [ ] Monitor camera feeds
  - [ ] Track database size growth

---

## Next Steps

1. **Run it locally first** - Get comfortable with the system
2. **Test with your photos** - Import a few students and test
3. **Deploy to server** - See [DEPLOYMENT.md](DEPLOYMENT.md)
4. **Train your staff** - Show admins how to use it
---

## Project Layout

The codebase is organized into functional modules:

**Core Components:**
- `core/` - Shared utilities (auth, database, config, models)
- `vision/` - AI and computer vision pipeline
- `camera/` - Camera capture and frame processing
- `recognition/` - Face detection, alignment, and matching

**Applications:**
- `admin_app/` - Administrator dashboard and management
- `student_app/` - Student self-service portal
- `web/` - Web routes and REST APIs

**Supporting:**
- `scripts/` - Setup, testing, and maintenance utilities
- `templates/` - HTML page templates
- `static/` - CSS, JavaScript, images
- `models/` - Pre-trained AI models (large files)
- `tests/` - Unit and integration tests
- `docker/` - Container configurations
- `deploy/` - Deployment manifests (Nginx, Kubernetes)

---

## Deployment

### Docker (Recommended)

```bash
# Build and start all services
docker compose up -d --build

# Access at http://localhost
```

### Linux Server

1. Install dependencies and Python 3.11
2. Clone repository and create virtual environment
3. Install requirements: `pip install -r requirements.txt`
4. Configure `.env` with MongoDB URI and settings
5. Run with Gunicorn: `gunicorn --config gunicorn.conf.py "app:create_app()"`
6. Configure Nginx as reverse proxy

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Kubernetes (Large Scale)

For production deployments with multiple instances, see [deploy/k8s/README.md](deploy/k8s/README.md).

---

## Testing

### Run All Tests

```bash
pytest -v
```

### Run Specific Tests

```bash
# Database tests
pytest tests/test_database.py -v

# Face recognition tests
pytest tests/test_recognition.py -v

# Anti-spoofing tests
pytest tests/test_anti_spoofing.py -v
```

### Quick Health Check

```bash
python scripts/smoke_test.py --base-url http://localhost:5000
```

---

## Frequently Asked Questions

**Q: Can I use a USB camera instead of the built-in webcam?**

Set `CAMERA_INDICES=0` for default camera, `1` for first USB, `2` for second USB, etc.

**Q: How many students can the system handle?**

- Per camera: 20-50 simultaneous tracked faces
- Total database: 10,000+ students
- Throughput: 1-2 students per second

**Q: What happens if the internet drops?**

The admin app requires MongoDB, so it goes offline. The camera continues capturing and stores data locally, syncing when connection returns.

**Q: Can students cheat with a photo?**

Difficult due to anti-spoofing. For maximum security, use stricter settings:

```env
LIVENESS_CONFIDENCE_THRESHOLD=0.75
LIVENESS_REAL_VOTE_RATIO=0.9
```

**Q: How do I export attendance to Excel?**

Use the "Export" button on the admin dashboard, or call the API:
```
GET /api/analytics/trends?format=csv
```

**Q: How do I customize the UI (logo, colors)?**

- Logo: Replace `static/images/logo.png`
- Colors: Edit `static/css/style.css`
- Layout: Edit HTML templates in `templates/`

**Q: How often should I backup data?**

MongoDB Atlas auto-backs daily. Also manually export monthly:
```bash
python scripts/migrate_encodings.py
```

---

## Performance Benchmarks

### Hardware Requirements

| Use Case | CPU | RAM | Storage |
|----------|-----|-----|---------|
| Small (1 camera, 50 students) | 2-core | 2GB | 20GB |
| Medium (2-3 cameras, 500 students) | 4-core | 4GB | 50GB |
| Large (4+ cameras, 2000+ students) | 8-core | 8GB | 200GB |

### Processing Speed

- YuNet face detection: ~15ms per frame
- CSRT tracking: ~5ms per frame
- Anti-spoofing: ~50ms (new faces only)
- ArcFace embedding: ~20ms per face
- Database write: ~10ms per record

### Accuracy Metrics

| Metric | Performance |
|--------|-------------|
| Face Detection | 98%+ accuracy |
| Face Recognition | 95%+ accuracy (proper lighting) |
| Anti-Spoofing | 99%+ detection rate |
| False Positive Rate | <1% with default settings |

---

## Roadmap & Future Features

Planned improvements:
- GPU acceleration (CUDA support)
- Multi-site federation
- Mobile app for students
- Biometric fingerprint integration
- Advanced analytics and pattern detection
- Email notifications for absences
- ERP system integrations

---

## License

This project is open-source under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support & Contributing

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/ShubhamPatra/attendance_system).

**Key Technologies Used:**
- YuNet (face detection)
- ArcFace (face recognition)
- Silent-Face-Anti-Spoofing (liveness detection)
- Flask (web framework)
- MongoDB Atlas (database)
- OpenCV (computer vision)
- Docker (containerization)


# #   P r o j e c t   L a y o u t   ( W h a t   G o e s   W h e r e ? ) 
 
 
 
 ` ` ` 
 
 a t t e n d a n c e _ s y s t e m / 
 
 �   
 
 �  S�  � �  �   � x    a p p . p y                                                     M a i n   F l a s k   a p p   ( a d m i n   d a s h b o a r d ) 
 
 �  S�  � �  �   � x    r u n . p y                                                     R u n   b o t h   a p p s   a t   o n c e 
 
 �  S�  � �  �   � x    r u n _ a d m i n . p y                                         R u n   a d m i n   o n l y 
 
 �  S�  � �  �   � x    r u n _ s t u d e n t . p y                                     R u n   s t u d e n t   p o r t a l   o n l y 
 
 �  S�  � �  �   � x    r e q u i r e m e n t s . t x t                                 P y t h o n   p a c k a g e s   t o   i n s t a l l 
 
 �  S�  � �  �   � x    . e n v . e x a m p l e                                         T e m p l a t e   f o r   s e t t i n g s 
 
 �  S�  � �  �   � x    . e n v                                                         Y o u r   a c t u a l   s e t t i n g s   ( d o n ' t   s h a r e ! ) 
 
 �  S�  � �  �   � x    M a k e f i l e                                                 S h o r t c u t s   ( m a k e   r u n ,   m a k e   t e s t ,   e t c ) 
 
 �  S�  � �  �   � x    d o c k e r - c o m p o s e . y m l                             D o c k e r   s e t u p 
 
 �  S�  � �  �   � x    g u n i c o r n . c o n f . p y                                 P r o d u c t i o n   w e b   s e r v e r   c o n f i g 
 
 �  S�  � �  �   � x    D E P L O Y M E N T . m d                                       H o w   t o   d e p l o y   t o   p r o d u c t i o n 
 
 �   
 
 �  S�  � �  �   � x �   c o r e /                                                       S h a r e d   c o d e 
 
 �         �  S�  � �  �   a u t h . p y                                                 L o g i n / p a s s w o r d   h a n d l i n g 
 
 �         �  S�  � �  �   c o n f i g . p y                                             S e t t i n g s   l o a d i n g 
 
 �         �  S�  � �  �   d a t a b a s e . p y                                         M o n g o D B   c o n n e c t i o n 
 
 �         �  S�  � �  �   m o d e l s . p y                                             D a t a b a s e   c o l l e c t i o n s   s c h e m a 
 
 �         �  S�  � �  �   u t i l s . p y                                               H e l p e r   f u n c t i o n s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   v i s i o n /                                                   A I / C o m p u t e r   V i s i o n 
 
 �         �  S�  � �  �   p i p e l i n e . p y                                         Y u N e t   f a c e   d e t e c t i o n ,   t r a c k i n g 
 
 �         �  S�  � �  �   r e c o g n i t i o n . p y                                   F a c e   a l i g n m e n t ,   m a t c h i n g 
 
 �         �  S�  � �  �   a n t i _ s p o o f i n g . p y                             L i v e n e s s   d e t e c t i o n 
 
 �         �  S�  � �  �   f a c e _ e n g i n e . p y                                   E m b e d d i n g   g e n e r a t i o n   ( A r c F a c e ) 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   c a m e r a /                                                   C a m e r a   &   R e a l - t i m e 
 
 �         �  S�  � �  �   c a m e r a . p y                                             C a m e r a   c a p t u r e   a n d   p r o c e s s i n g 
 
 �         �   �  � �  �   _ _ i n i t _ _ . p y 
 
 �   
 
 �  S�  � �  �   � x �   a d m i n _ a p p /                                             A d m i n   D a s h b o a r d   A p p 
 
 �         �  S�  � �  �   a p p . p y                                                   A d m i n   F l a s k   a p p 
 
 �         �  S�  � �  �   f o r m s . p y                                               A d m i n   f o r m   h a n d l i n g 
 
 �         �  S�  � �  �   r o u t e s /                                                 A d m i n   w e b   p a g e s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   s t u d e n t _ a p p /                                         S t u d e n t   P o r t a l   A p p 
 
 �         �  S�  � �  �   a p p . p y                                                   S t u d e n t   F l a s k   a p p 
 
 �         �  S�  � �  �   r o u t e s . p y                                             S t u d e n t   w e b   p a g e s 
 
 �         �  S�  � �  �   a u t h . p y                                                 S t u d e n t   l o g i n 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   w e b /                                                         W e b   R o u t e s   &   A P I s 
 
 �         �  S�  � �  �   a t t e n d a n c e _ r o u t e s . p y                       A t t e n d a n c e   r e c o r d i n g   r o u t e s 
 
 �         �  S�  � �  �   r e p o r t _ r o u t e s . p y                               R e p o r t s   a n d   a n a l y t i c s 
 
 �         �  S�  � �  �   a p i _ r o u t e s . p y                                     R E S T   A P I s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   s c r i p t s /                                                 H e l p e r   S c r i p t s 
 
 �         �  S�  � �  �   b o o t s t r a p _ a d m i n . p y                           C r e a t e   f i r s t   a d m i n   a c c o u n t 
 
 �         �  S�  � �  �   s e e d _ d e m o _ d a t a . p y                             C r e a t e   d e m o   s t u d e n t s 
 
 �         �  S�  � �  �   c l e a r _ d b . p y                                         D e l e t e   a l l   d a t a b a s e   d a t a 
 
 �         �  S�  � �  �   d e b u g _ p i p e l i n e . p y                             T e s t   c a m e r a / r e c o g n i t i o n 
 
 �         �  S�  � �  �   c a l i b r a t e _ l i v e n e s s _ t h r e s h o l d . p y   T u n e   a n t i - s p o o f i n g 
 
 �         �  S�  � �  �   s m o k e _ t e s t . p y                                     C h e c k   i f   s y s t e m   w o r k s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   t e m p l a t e s /                                             H T M L   P a g e s 
 
 �         �  S�  � �  �   b a s e . h t m l                                             B a s e   l a y o u t   t e m p l a t e 
 
 �         �  S�  � �  �   d a s h b o a r d . h t m l                                   A d m i n   d a s h b o a r d 
 
 �         �  S�  � �  �   a t t e n d a n c e . h t m l                                 L i v e   c a m e r a   p a g e 
 
 �         �  S�  � �  �   r e g i s t e r . h t m l                                     R e g i s t r a t i o n   f o r m 
 
 �         �  S�  � �  �   s t u d e n t _ l o g i n . h t m l                           S t u d e n t   p o r t a l   l o g i n 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   s t a t i c /                                                   S t a t i c   F i l e s   ( C S S ,   J S ,   I m a g e s ) 
 
 �         �  S�  � �  �   c s s /                                                       S t y l e s h e e t s 
 
 �         �  S�  � �  �   j s /                                                         J a v a S c r i p t   f i l e s 
 
 �         �   �  � �  �   i m a g e s /                                                 L o g o ,   i c o n s 
 
 �   
 
 �  S�  � �  �   � x �   m o d e l s /                                                   A I   M o d e l s   ( L a r g e   F i l e s ) 
 
 �         �  S�  � �  �   f a c e _ d e t e c t i o n _ y u n e t _ 2 0 2 3 m a r . o n n x             Y u N e t   m o d e l 
 
 �         �  S�  � �  �   a n t i _ s p o o f i n g /                                                   A n t i - s p o o f i n g   m o d e l s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   d o c k e r /                                                   D o c k e r   C o n f i g u r a t i o n 
 
 �         �  S�  � �  �   D o c k e r f i l e                                           M a i n   a p p   c o n t a i n e r 
 
 �         �  S�  � �  �   D o c k e r f i l e . a d m i n                               A d m i n   a p p   c o n t a i n e r 
 
 �         �   �  � �  �   D o c k e r f i l e . s t u d e n t                           S t u d e n t   a p p   c o n t a i n e r 
 
 �   
 
 �  S�  � �  �   � x �   d e p l o y /                                                   D e p l o y m e n t   F i l e s 
 
 �         �  S�  � �  �   n g i n x /                                                   N g i n x   w e b   s e r v e r   c o n f i g 
 
 �         �  S�  � �  �   k 8 s /                                                       K u b e r n e t e s   c o n f i g s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   t e s t s /                                                     T e s t   F i l e s 
 
 �         �  S�  � �  �   t e s t _ d a t a b a s e . p y                               T e s t   d a t a b a s e   f u n c t i o n s 
 
 �         �  S�  � �  �   t e s t _ c a m e r a _ p i p e l i n e . p y                 T e s t   f a c e   d e t e c t i o n 
 
 �         �  S�  � �  �   t e s t _ r e c o g n i t i o n . p y                         T e s t   f a c e   m a t c h i n g 
 
 �         �  S�  � �  �   t e s t _ a n t i _ s p o o f i n g . p y                     T e s t   l i v e n e s s   d e t e c t i o n 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   u p l o a d s /                                                 U p l o a d e d   F i l e s 
 
 �         �  S�  � �  �   s t u d e n t _ a p p /                                       S t u d e n t   p h o t o s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �  S�  � �  �   � x �   l o g s /                                                       A p p l i c a t i o n   L o g s 
 
 �         �   �  � �  �   * . l o g                                                     D e b u g / e r r o r   l o g s 
 
 �   
 
 �  S�  � �  �   � x �   b a c k u p s /                                                 D a t a b a s e   B a c k u p s 
 
 �         �   �  � �  �   . . . 
 
 �   
 
 �   �  � �  �   � x �   c e l e r y _ d a t a /                                         C e l e r y   Q u e u e   ( i f   e n a b l e d ) 
 
         �   �  � �  �   . . . 
 
 ` ` ` 
 
 
 
 * * K e y   F o l d e r s   t o   K n o w : * * 
 
 
 
 -   * * ` c o r e / ` * *   -   D o n ' t   c h a n g e   u n l e s s   y o u   k n o w   w h a t   y o u ' r e   d o i n g 
 
 -   * * ` v i s i o n / ` * *   -   A I   m o d e l s   a n d   a l g o r i t h m s 
 
 -   * * ` a d m i n _ a p p / ` ,   ` s t u d e n t _ a p p / ` * *   -   T h e   t w o   a p p s   ( m o d i f y   H T M L / C S S   h e r e ) 
 
 -   * * ` s c r i p t s / ` * *   -   U t i l i t i e s   f o r   s e t u p   a n d   t e s t i n g 
 
 -   * * ` t e m p l a t e s / ` * *   -   E d i t   t h e s e   t o   c h a n g e   w e b   p a g e   l a y o u t / d e s i g n 
 
 -   * * ` s t a t i c / ` * *   -   W e b s i t e   s t y l i n g   a n d   i m a g e s 
 
 
 
 - - - 
 
 
 
 # #   D e p l o y m e n t   ( M o v i n g   t o   P r o d u c t i o n ) 
 
 
 
 # # #   O p t i o n   1 :   D o c k e r   ( E a s i e s t ) 
 
 
 
 ` ` ` b a s h 
 
 #   B u i l d   a n d   s t a r t 
 
 d o c k e r   c o m p o s e   u p   - d   - - b u i l d 
 
 
 
 #   A c c e s s   v i a   h t t p : / / l o c a l h o s t 
 
 ` ` ` 
 
 
 
 S e e   [ D E P L O Y M E N T . m d ] ( D E P L O Y M E N T . m d )   f o r   D o c k e r   b e s t   p r a c t i c e s . 
 
 
 
 # # #   O p t i o n   2 :   L i n u x   S e r v e r   ( M o s t   C o m m o n ) 
 
 
 
 1 .   * * S e t u p   s e r v e r : * * 
 
       ` ` ` b a s h 
 
       #   S S H   i n t o   y o u r   L i n u x   s e r v e r 
 
       s s h   u s e r @ y o u r - s e r v e r . c o m 
 
       
 
       #   I n s t a l l   d e p e n d e n c i e s 
 
       s u d o   a p t   u p d a t e 
 
       s u d o   a p t   i n s t a l l   p y t h o n 3 . 1 1   p y t h o n 3 - p i p   n g i n x 
 
       ` ` ` 
 
 
 
 2 .   * * D e p l o y   c o d e : * * 
 
       ` ` ` b a s h 
 
       g i t   c l o n e   h t t p s : / / g i t h u b . c o m / S h u b h a m P a t r a / a t t e n d a n c e _ s y s t e m . g i t 
 
       c d   a t t e n d a n c e _ s y s t e m 
 
       p y t h o n 3   - m   v e n v   v e n v 
 
       s o u r c e   v e n v / b i n / a c t i v a t e 
 
       p i p   i n s t a l l   - r   r e q u i r e m e n t s . t x t 
 
       ` ` ` 
 
 
 
 3 .   * * S e t u p   G u n i c o r n   ( p r o d u c t i o n   w e b   s e r v e r ) : * * 
 
       ` ` ` b a s h 
 
       g u n i c o r n   - - c o n f i g   g u n i c o r n . c o n f . p y   " a p p : c r e a t e _ a p p ( ) " 
 
       ` ` ` 
 
 
 
 4 .   * * S e t u p   N g i n x   ( r e v e r s e   p r o x y ) : * * 
 
       ` ` ` b a s h 
 
       s u d o   c p   d e p l o y / n g i n x / d e f a u l t . c o n f   / e t c / n g i n x / s i t e s - a v a i l a b l e / a u t o a t t e n d a n c e 
 
       s u d o   l n   - s   / e t c / n g i n x / s i t e s - a v a i l a b l e / a u t o a t t e n d a n c e   / e t c / n g i n x / s i t e s - e n a b l e d / 
 
       s u d o   n g i n x   - t 
 
       s u d o   s y s t e m c t l   r e s t a r t   n g i n x 
 
       ` ` ` 
 
 
 
 5 .   * * S t a r t   w i t h   s y s t e m d   ( a u t o - r e s t a r t ) : * * 
 
       ` ` ` b a s h 
 
       s u d o   s y s t e m c t l   e n a b l e   a u t o a t t e n d a n c e 
 
       s u d o   s y s t e m c t l   s t a r t   a u t o a t t e n d a n c e 
 
       ` ` ` 
 
 
 
 S e e   [ D E P L O Y M E N T . m d ] ( D E P L O Y M E N T . m d )   f o r   f u l l   g u i d e . 
 
 
 
 # # #   O p t i o n   3 :   K u b e r n e t e s   ( L a r g e   S c a l e ) 
 
 
 
 F o r   s c h o o l s / o f f i c e s   w i t h   1 0 0 0 +   s t u d e n t s : 
 
 
 
 S e e   [ d e p l o y / k 8 s / R E A D M E . m d ] ( d e p l o y / k 8 s / R E A D M E . m d ) 
 
 
 
 - - - 
 
 
 
 # #   T e s t i n g 
 
 
 
 # # #   R u n   A l l   T e s t s 
 
 
 
 ` ` ` b a s h 
 
 p y t e s t   - v 
 
 ` ` ` 
 
 
 
 # # #   R u n   S p e c i f i c   T e s t 
 
 
 
 ` ` ` b a s h 
 
 #   T e s t   d a t a b a s e   f u n c t i o n s 
 
 p y t e s t   t e s t s / t e s t _ d a t a b a s e . p y   - v 
 
 
 
 #   T e s t   f a c e   r e c o g n i t i o n 
 
 p y t e s t   t e s t s / t e s t _ r e c o g n i t i o n . p y   - v 
 
 
 
 #   T e s t   a n t i - s p o o f i n g 
 
 p y t e s t   t e s t s / t e s t _ a n t i _ s p o o f i n g . p y   - v 
 
 ` ` ` 
 
 
 
 # # #   Q u i c k   S m o k e   T e s t 
 
 
 
 ` ` ` b a s h 
 
 p y t h o n   s c r i p t s / s m o k e _ t e s t . p y   - - b a s e - u r l   h t t p : / / l o c a l h o s t : 5 0 0 0   - - c h e c k - v i d e o 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 # #   C o n t r i b u t i n g   &   D e v e l o p m e n t 
 
 
 
 # # #   C o d e   S t r u c t u r e   B e s t   P r a c t i c e s 
 
 
 
 1 .   * * K e e p   v i s i o n   c o d e   i n   ` v i s i o n / ` * *   -   A l l   A I / C V   l o g i c 
 
 2 .   * * K e e p   r o u t e s   i n   a p p r o p r i a t e   f o l d e r s * *   -   A d m i n   i n   ` a d m i n _ a p p / ` ,   A P I   i n   ` w e b / ` 
 
 3 .   * * U s e   u t i l i t y   f u n c t i o n s * *   -   ` c o r e / u t i l s . p y `   f o r   c o m m o n   h e l p e r s 
 
 4 .   * * W r i t e   t e s t s * *   -   E v e r y   f e a t u r e   s h o u l d   h a v e   t e s t s   i n   ` t e s t s / ` 
 
 
 
 # # #   R u n n i n g   i n   D e b u g   M o d e 
 
 
 
 ` ` ` e n v 
 
 D E B U G _ M O D E = 1 
 
 A P P _ D E B U G = 1 
 
 B Y P A S S _ A N T I S P O O F = 1                     #   S k i p   a n t i - s p o o f i n g   f o r   t e s t i n g 
 
 B Y P A S S _ M O T I O N _ D E T E C T I O N = 1       #   D e t e c t   e v e r y   f r a m e   ( s l o w e r   b u t   f i n d s   e v e r y t h i n g ) 
 
 ` ` ` 
 
 
 
 # # #   P r o f i l i n g   P e r f o r m a n c e 
 
 
 
 ` ` ` b a s h 
 
 p y t h o n   s c r i p t s / d e b u g _ p i p e l i n e . p y   - - p r o f i l e 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 # #   F r e q u e n t l y   A s k e d   Q u e s t i o n s 
 
 
 
 # # #   Q :   C a n   I   u s e   a   U S B   c a m e r a   i n s t e a d   o f   l a p t o p   w e b c a m ? 
 
 
 
 * * A : * *   Y e s !   S e t   i n   ` . e n v ` : 
 
 ` ` ` e n v 
 
 C A M E R A _ I N D I C E S = 0     #   0   =   d e f a u l t ,   1   =   f i r s t   U S B ,   2   =   s e c o n d   U S B ,   e t c 
 
 ` ` ` 
 
 
 
 # # #   Q :   H o w   m a n y   s t u d e n t s   c a n   t h e   s y s t e m   h a n d l e ? 
 
 
 
 * * A : * *   
 
 -   * * P e r   c a m e r a : * *   2 0 - 5 0   s i m u l t a n e o u s   t r a c k e d   f a c e s 
 
 -   * * T o t a l   d a t a b a s e : * *   1 0 , 0 0 0 +   s t u d e n t s   ( M o n g o D B   h a n d l e s   i t ) 
 
 -   * * T h r o u g h p u t : * *   1 - 2   s t u d e n t s   p e r   s e c o n d 
 
 
 
 # # #   Q :   W h a t   h a p p e n s   i f   t h e   i n t e r n e t   d r o p s ? 
 
 
 
 * * A : * *   
 
 -   A d m i n   a p p   g o e s   d o w n   ( n e e d s   M o n g o D B ) 
 
 -   C a m e r a   s t i l l   c a p t u r e s   f r a m e s   a n d   s t o r e s   l o c a l l y 
 
 -   A t t e n d a n c e   s y n c s   w h e n   i n t e r n e t   r e t u r n s 
 
 
 
 # # #   Q :   C a n   I   e x p o r t   a t t e n d a n c e   t o   E x c e l ? 
 
 
 
 * * A : * *   Y e s !   
 
 -   A d m i n   d a s h b o a r d   h a s   " E x p o r t "   b u t t o n 
 
 -   O r   u s e   A P I :   ` G E T   / a p i / a n a l y t i c s / t r e n d s ? f o r m a t = c s v ` 
 
 
 
 # # #   Q :   H o w   d o   I   c h a n g e   t h e   l o g o / c o l o r s ? 
 
 
 
 * * A : * * 
 
 1 .   R e p l a c e   l o g o :   ` s t a t i c / i m a g e s / l o g o . p n g ` 
 
 2 .   E d i t   c o l o r s :   ` s t a t i c / c s s / s t y l e . c s s ` 
 
 3 .   O r   e d i t   H T M L :   ` t e m p l a t e s / b a s e . h t m l ` 
 
 
 
 # # #   Q :   C a n   s t u d e n t s   c h e a t   w i t h   a   p h o t o ? 
 
 
 
 * * A : * *   V e r y   h a r d   b e c a u s e : 
 
 -   A n t i - s p o o f i n g   d e t e c t s   2 D   p h o t o s 
 
 -   S y s t e m   r e q u i r e s   m o v e m e n t 
 
 -   M a j o r i t y   v o t i n g   m a k e s   f o o l i n g   i t   v e r y   d i f f i c u l t 
 
 
 
 B u t   f o r   s e c u r i t y - c r i t i c a l   s i t u a t i o n s ,   u s e   s t r i c t e s t   s e t t i n g s : 
 
 ` ` ` e n v 
 
 L I V E N E S S _ C O N F I D E N C E _ T H R E S H O L D = 0 . 7 5 
 
 L I V E N E S S _ R E A L _ V O T E _ R A T I O = 0 . 9 
 
 ` ` ` 
 
 
 
 # # #   Q :   H o w   o f t e n   s h o u l d   I   b a c k u p   d a t a ? 
 
 
 
 * * A : * *   W e e k l y   m i n i m u m .   M o n g o D B   A t l a s   a u t o - b a c k u p s   d a i l y ,   b u t   a l s o : 
 
 ` ` ` b a s h 
 
 #   M a n u a l   e x p o r t 
 
 p y t h o n   s c r i p t s / m i g r a t e _ e n c o d i n g s . p y     #   E x p o r t s   a l l   d a t a 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 # #   P e r f o r m a n c e   B e n c h m a r k s 
 
 
 
 # # #   H a r d w a r e   R e q u i r e m e n t s 
 
 
 
 |   U s e   C a s e   |   C P U   |   R A M   |   S t o r a g e   | 
 
 | - - - - - - - - - - | - - - - - | - - - - - | - - - - - - - - - | 
 
 |   S m a l l   ( 1   c a m e r a ,   5 0   s t u d e n t s )   |   2 - c o r e   |   2 G B   |   2 0 G B   | 
 
 |   M e d i u m   ( 2 - 3   c a m e r a s ,   5 0 0   s t u d e n t s )   |   4 - c o r e   |   4 G B   |   5 0 G B   | 
 
 |   L a r g e   ( 4 +   c a m e r a s ,   2 0 0 0 +   s t u d e n t s )   |   8 - c o r e   |   8 G B   |   2 0 0 G B   | 
 
 
 
 # # #   P r o c e s s i n g   S p e e d 
 
 
 
 -   * * Y u N e t   d e t e c t i o n : * *   ~ 1 5 m s   p e r   f r a m e 
 
 -   * * C S R T   t r a c k i n g : * *   ~ 5 m s   p e r   f r a m e 
 
 -   * * A n t i - s p o o f i n g : * *   ~ 5 0 m s   ( o n l y   f o r   n e w   f a c e s ) 
 
 -   * * A r c F a c e   e m b e d d i n g : * *   ~ 2 0 m s   p e r   f a c e 
 
 -   * * D a t a b a s e   w r i t e : * *   ~ 1 0 m s   p e r   r e c o r d 
 
 
 
 # # #   A c c u r a c y   M e t r i c s 
 
 
 
 |   M e t r i c   |   V a l u e   | 
 
 | - - - - - - - - | - - - - - - - | 
 
 |   F a c e   D e t e c t i o n   |   9 8 % +   a c c u r a c y   | 
 
 |   F a c e   R e c o g n i t i o n   |   9 5 % +   a c c u r a c y   ( w i t h   p r o p e r   l i g h t i n g )   | 
 
 |   A n t i - S p o o f i n g   |   9 9 % +   d e t e c t i o n   r a t e   | 
 
 |   F a l s e   P o s i t i v e   R a t e   |   < 1 %   w i t h   d e f a u l t   s e t t i n g s   | 
 
 
 
 - - - 
 
 
 
 # #   R o a d m a p   &   F u t u r e   F e a t u r e s 
 
 
 
 * * P l a n n e d   i m p r o v e m e n t s : * * 
 
 -   [   ]   G P U   a c c e l e r a t i o n   ( C U D A   s u p p o r t ) 
 
 -   [   ]   M u l t i - s i t e   f e d e r a t i o n 
 
 -   [   ]   M o b i l e   a p p   f o r   s t u d e n t s 
 
 -   [   ]   B i o m e t r i c   f i n g e r p r i n t   i n t e g r a t i o n 
 
 -   [   ]   A d v a n c e d   a n a l y t i c s   ( p a t t e r n   d e t e c t i o n ) 
 
 -   [   ]   E m a i l   n o t i f i c a t i o n s 
 
 -   [   ]   I n t e g r a t i o n   w i t h   s c h o o l   E R P   s y s t e m s 
 
 
 
 - - - 
 
 
 
 # #   L i c e n s e   &   C r e d i t s 
 
 
 
 # # #   L i c e n s e 
 
 T h i s   p r o j e c t   i s   o p e n - s o u r c e   u n d e r   t h e   [ M I T   L i c e n s e ] ( L I C E N S E ) . 
 
 
 
 # # #   K e y   T e c h n o l o g i e s 
 
 -   * * Y u N e t * *   -   F a c e   d e t e c t i o n   b y   O p e n C V 
 
 -   * * A r c F a c e * *   -   F a c e   r e c o g n i t i o n   b y   I n s i g h t F a c e 
 
 -   * * S i l e n t - F a c e * *   -   A n t i - s p o o f i n g 
 
 -   * * F l a s k * *   -   W e b   f r a m e w o r k 
 
 -   * * M o n g o D B * *   -   D a t a b a s e 
 
 -   * * O p e n C V * *   -   C o m p u t e r   v i s i o n   l i b r a r y 
 
 
 
 # # #   A u t h o r s 
 
 -   L e a d :   S h u b h a m   P a t r a 
 
 -   C o n t r i b u t o r s :   [ S e e   G i t H u b ] ( h t t p s : / / g i t h u b . c o m / S h u b h a m P a t r a / a t t e n d a n c e _ s y s t e m / c o n t r i b u t o r s ) 
 
 
 
 - - - 
 
 
 
 # #   G e t t i n g   H e l p 
 
 
 
 # # #   D o c u m e n t a t i o n 
 
 -   [ F u l l   D e p l o y m e n t   G u i d e ] ( D E P L O Y M E N T . m d ) 
 
 -   [ K u b e r n e t e s   S e t u p ] ( d e p l o y / k 8 s / R E A D M E . m d ) 
 
 -   [ C o n f i g u r a t i o n   R e f e r e n c e ] ( . e n v . e x a m p l e ) 
 
 
 
 # # #   S u p p o r t 
 
 -   � x �   [ G i t H u b   I s s u e s ] ( h t t p s : / / g i t h u b . c o m / S h u b h a m P a t r a / a t t e n d a n c e _ s y s t e m / i s s u e s ) 
 
 -   � x �   [ D i s c u s s i o n s ] ( h t t p s : / / g i t h u b . c o m / S h u b h a m P a t r a / a t t e n d a n c e _ s y s t e m / d i s c u s s i o n s ) 
 
 -   � x �   R e p o r t   b u g s   o r   s e c u r i t y   i s s u e s 
 
 
 
 # # #   C o m m u n i t y 
 
 -   S t a r   � � �   i f   t h i s   h e l p e d   y o u ! 
 
 -   F o r k   a n d   c o n t r i b u t e   i m p r o v e m e n t s 
 
 -   S h a r e   y o u r   u s e   c a s e s 
 
 
 
 - - - 
 
 
 
 # #   Q u i c k   R e f e r e n c e   C a r d 
 
 
 
 ` ` ` b a s h 
 
 #   S e t u p 
 
 p y t h o n   - m   v e n v   v e n v   & &   s o u r c e   v e n v / b i n / a c t i v a t e 
 
 p i p   i n s t a l l   - r   r e q u i r e m e n t s . t x t 
 
 p y t h o n   s c r i p t s / b o o t s t r a p _ a d m i n . p y   - - u s e r n a m e   a d m i n 
 
 
 
 #   R u n 
 
 p y t h o n   r u n . p y                                         #   R u n   b o t h   a p p s 
 
 d o c k e r   c o m p o s e   u p   - d                           #   R u n   i n   D o c k e r 
 
 
 
 #   M a n a g e 
 
 p y t h o n   - m   s c r i p t s . c l e a r _ d b               #   D e l e t e   a l l   d a t a 
 
 p y t h o n   s c r i p t s / s e e d _ d e m o _ d a t a . p y   #   A d d   d e m o   d a t a 
 
 p y t h o n   s c r i p t s / s m o k e _ t e s t . p y           #   T e s t   s y s t e m 
 
 
 
 #   D e b u g 
 
 p y t h o n   s c r i p t s / d e b u g _ p i p e l i n e . p y   #   T e s t   c a m e r a / r e c o g n i t i o n 
 
 g r e p   " e r r o r \ | E R R O R "   l o g s / * . l o g       #   V i e w   e r r o r s 
 
 
 
 #   D e p l o y 
 
 g u n i c o r n   - - c o n f i g   g u n i c o r n . c o n f . p y   " a p p : c r e a t e _ a p p ( ) " 
 
 d o c k e r   c o m p o s e   u p   - d   - - b u i l d 
 
 ` ` ` 
 
 
 
 - - - 
 
 
 
 * * L a s t   U p d a t e d : * *   A p r i l   2 0 2 6     
 
 * * V e r s i o n : * *   2 . 0     
 
 * * S t a t u s : * *   P r o d u c t i o n   R e a d y   � S& 
 
 