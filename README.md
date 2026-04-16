# AutoAttendance

## Intelligent Face Recognition Attendance System with Anti-Spoofing Detection

AutoAttendance is a production-ready, computer-vision-based automated attendance system that uses facial recognition and liveness detection to securely and accurately mark student attendance in real-time. Designed for educational institutions, it combines YuNet face detection, ArcFace embeddings, and Silent-Face-Anti-Spoofing to deliver accurate, tamper-resistant attendance tracking with minimal manual intervention.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement & Solution](#problem-statement--solution)
3. [Key Features](#key-features)
4. [Real-World Applications](#real-world-applications)
5. [System Architecture](#system-architecture)
6. [Technology Stack](#technology-stack)
7. [Quick Start Guide](#quick-start-guide)
8. [Project Structure](#project-structure)
9. [Configuration](#configuration)
10. [API Overview](#api-overview)
11. [Documentation Index](#documentation-index)
12. [Contributing](#contributing)
13. [License](#license)

---

## Project Overview

**AutoAttendance** automates the attendance marking process by recognizing student faces in real-time using computer vision and deep learning. The system operates in two modes:

- **Admin Panel** (Port 5000): Manage students, view analytics, control camera feeds, and download attendance reports.
- **Student Portal** (Port 5001): Student onboarding with face verification, attendance history, and self-service verification status tracking.

The system is designed for accuracy, speed, and security:
- **Accuracy**: Multi-frame confirmation and two-stage recognition ensure robust matching across varying lighting and pose conditions.
- **Security**: Anti-spoofing detection prevents forged attendance via printed photos, videos, or masks.
- **Performance**: Lightweight tracker reuse and caching reduce computational overhead; runs on CPU or GPU.

---

## Problem Statement & Solution

### Traditional Attendance Systems

Manual or RFID-based attendance systems suffer from:
- **Time-consuming**: Manual roll calls waste classroom time.
- **Proxy attendance**: Classmates mark attendance on behalf of absent students.
- **Scalability issues**: Manual processes don't scale to large institutions.
- **No audit trail**: Limited accountability and difficult attendance verification.

### AutoAttendance Solution

AutoAttendance solves these problems with:
- **Automated marking**: Real-time face recognition eliminates manual roll calls.
- **Anti-spoofing**: Liveness detection prevents fraudulent proxy attendance.
- **Scalability**: Handles hundreds of students and multiple classrooms simultaneously.
- **Detailed audit trail**: Complete logs of all attendance events with confidence scores and timestamps.
- **Analytics**: Attendance trends, at-risk student identification, and absence alerts.

---

## Key Features

### Core Attendance & Recognition

- ✓ **Real-Time Face Detection & Tracking**: OpenCV CSRT trackers paired with YuNet ONNX detection.
- ✓ **ArcFace Face Recognition**: 512-D embeddings with cosine similarity matching.
- ✓ **Adaptive Quality Gating**: Blur, brightness, and face-size checks reject poor-quality captures.
- ✓ **Multi-Frame Confirmation**: Temporal voting (≥3 of 5 frames) prevents false matches.
- ✓ **Session-Based Attendance**: Attendance sessions can span multiple hours; idle auto-close after 15 minutes of inactivity.

### Anti-Spoofing & Liveness

- ✓ **Silent-Face-Anti-Spoofing**: CNN-based real/spoof classification.
- ✓ **Blink Detection**: Eye-Aspect Ratio (EAR) tracking supplements liveness scoring.
- ✓ **Head Movement Detection**: Subtle motion confirms genuine face presence.
- ✓ **Frame-Level Heuristics**: Contrast and brightness analysis flags screen replays and over-exposed footage.

### Student Onboarding & Verification

- ✓ **Self-Service Enrollment**: Students capture 3–5 face samples via webcam.
- ✓ **Automatic Scoring**: Liveness, consistency, and quality scores drive auto-approval or manual review.
- ✓ **Duplicate Prevention**: Cosine similarity check prevents the same person enrolling twice.
- ✓ **Admin Manual Approval**: Override auto-scores for edge cases.

### Admin & Analytics

- ✓ **Dashboard**: Real-time attendance count, heatmaps, and trends.
- ✓ **Attendance Reports**: Daily, weekly, and custom-date-range CSV exports.
- ✓ **At-Risk Detection**: Identifies students with low attendance percentages.
- ✓ **Student Management**: Batch import, profile editing, and re-verification.
- ✓ **Real-Time Camera Feed**: Live video with overlays showing identity, confidence, and liveness state.

### Deployment & Operations

- ✓ **Docker Support**: CPU and GPU builds with multi-stage Dockerfile.
- ✓ **MongoDB Atlas Integration**: Cloud or on-premise MongoDB support.
- ✓ **Environment Configuration**: 80+ tunable parameters for thresholds, paths, and runtime behavior.
- ✓ **Startup Diagnostics**: Automatic checks for models, camera, and database on startup.
- ✓ **Circuit Breaker**: Fault tolerance for database failures with graceful degradation.

---

## Real-World Applications

### Educational Institutions

- Universities and colleges can automate large lecture-hall attendance.
- Eliminates manual roll calls and proxy attendance fraud.
- Generates detailed attendance analytics for academic planning.

### Training Programs & Certification

- Professional training centers can verify attendance for certification eligibility.
- Prevents credential fraud by requiring genuine face presence.

### Events & Access Control

- Conferences and workshops can automate attendee check-in.
- Secure venue access with real-time attendee tracking.

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │   Admin Panel        │  │  Student Portal      │             │
│  │   (Port 5000)        │  │  (Port 5001)         │             │
│  └──────────┬───────────┘  └──────────┬───────────┘             │
└─────────────┼──────────────────────────┼────────────────────────┘
              │                          │
┌─────────────┼──────────────────────────┼────────────────────────┐
│             │   WEB & ROUTES LAYER     │                        │
│  ┌──────────▼──────────────────────────▼────────┐               │
│  │  Flask Blueprints & REST APIs                 │               │
│  │  ├─ Auth, Attendance, Registration            │               │
│  │  ├─ Camera (SocketIO real-time)               │               │
│  │  ├─ Reports & Analytics                       │               │
│  │  └─ Ops & Health Checks                       │               │
│  └──────────────────┬─────────────────────────────┘               │
└─────────────────────┼───────────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────────┐
│                     │   VISION & ML PIPELINE LAYER              │
│  ┌──────────────────▼──────────────────────────┐                │
│  │  Pipeline Components:                        │                │
│  │  ├─ YuNet Face Detection (ONNX)              │                │
│  │  ├─ OpenCV CSRT Tracking                     │                │
│  │  ├─ ArcFace Embeddings (InsightFace)         │                │
│  │  ├─ Silent-Face Anti-Spoofing                │                │
│  │  ├─ Face Alignment & Quality Gating          │                │
│  │  ├─ Blink Detection & Motion Heuristics      │                │
│  │  └─ Adaptive Thresholding                    │                │
│  └──────────────────┬──────────────────────────┘                │
│                     │                                            │
│  ┌──────────────────▼──────────────────────────┐                │
│  │  Camera Module & Real-Time Loop              │                │
│  │  ├─ Frame Capture & Resize                   │                │
│  │  ├─ Detect-Track-Recognize Pipeline          │                │
│  │  ├─ Multi-Frame Confirmation Logic           │                │
│  │  ├─ Unknown Face Snapshot Logging            │                │
│  │  └─ SocketIO Event Emission                  │                │
│  └──────────────────┬──────────────────────────┘                │
└─────────────────────┼───────────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────────┐
│                     │   CORE SERVICES LAYER                      │
│  ┌──────────────────▼──────────────────────────┐                │
│  │  Database & Config:                          │                │
│  │  ├─ MongoDB Connection & Circuit Breaker      │                │
│  │  ├─ Data Access Objects (DAOs)               │                │
│  │  ├─ Configuration & Secrets                  │                │
│  │  ├─ Logging & Profiling                      │                │
│  │  └─ Authentication & Hashing                 │                │
│  └──────────────────┬──────────────────────────┘                │
└─────────────────────┼───────────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────────┐
│                     │   DATA PERSISTENCE LAYER                   │
│  ┌──────────────────▼──────────────────────────┐                │
│  │  MongoDB Atlas                               │                │
│  │  ├─ students collection                      │                │
│  │  ├─ attendance & attendance_sessions         │                │
│  │  ├─ users (admin/teacher accounts)           │                │
│  │  └─ notification_events                      │                │
│  └──────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
FRAME CAPTURE
    │
    ▼
MOTION DETECTION (if NO_MOTION_DETECTION_INTERVAL elapsed)
    │
    ├─ No motion → skip detection, update tracking
    │
    └─ Motion detected → run YuNet detection every N frames
       │
       ▼
DETECT & ASSOCIATE (YuNet → new tracks + update existing)
    │
    ├─ Matched tracks → reset frames_missing counter
    │
    └─ New detections → create fresh FaceTrack objects
       │
       ▼
UPDATE TRACKERS (CSRT per track)
    │
    ├─ Success → continue tracking
    │
    └─ Failure → increment frames_missing, expire after threshold
       │
       ▼
FOR EACH TRACK (detection cycle):
  ├─ Quality Gate (blur, brightness, size) → reject poor crops
  ├─ Alignment (ArcFace 5-point or legacy eye-center)
  ├─ Encoding (recognition model on aligned face)
  ├─ Recognition (cosine similarity vs. encoding cache)
  ├─ Liveness (Silent-Face model on full frame or crop)
  ├─ Multi-Frame Voting (rolling ≥3/5 policy)
  │
  └─ If confirmed real + recognized:
     ├─ Emit SocketIO event (UI overlay)
     ├─ Write attendance mark (once per day per student)
     └─ Cache result to avoid re-recognition
```

### Data Flow Diagram

```
Student → Capture & Enroll
  │
  ├─ Upload 3-5 face samples (webcam or batch)
  ├─ YuNet detection → validate one face per sample
  ├─ Quality gate (blur, brightness, size)
  ├─ Liveness check (Silent-Face)
  ├─ Encoding generation (ArcFace)
  ├─ Duplicate detection (cosine similarity)
  │
  └─ VerificationResult:
     ├─ Auto-approve (score ≥ 85 + no duplicates + liveness pass)
     ├─ Pending (60–85; manual admin review)
     └─ Reject (< 60 or duplicate found)

Approved Student → Attendance Marking
  │
  ├─ Encoding cache load on startup
  ├─ Admin creates attendance session (camera + course)
  │
  └─ Per frame:
     ├─ Detect face → track → align → encode
     ├─ Cosine similarity match (threshold 0.38)
     ├─ Multi-frame confirmation (≥3/5 frames real + recognized)
     ├─ Write attendance record (one mark per day)
     └─ Emit attendance event (admin UI + logs)

Attendance Session → Admin Analytics
  │
  ├─ Daily counts (CSV export, heatmap)
  ├─ Per-student trends (present/absent %)
  ├─ At-risk detection (< 75% attendance)
  └─ Session closure (manual or auto-idle)
```

---

## Technology Stack

### Computer Vision & ML

| Technology | Version | Purpose |
|---|---|---|
| **OpenCV (contrib-python)** | 4.8+ | CSRT tracking, image processing, CLAHE preprocessing |
| **YuNet (ONNX)** | 2023 | Real-time face detection (320×320 input, < 50ms on CPU) |
| **InsightFace (ArcFace)** | 0.7+ | 512-D face embeddings, automatic 5-point landmarks |
| **ONNX Runtime** | 1.17+ | Cross-platform inference (CPU/GPU/TensorRT) |
| **PyTorch** | 2.0+ | Silent-Face-Anti-Spoofing model inference |
| **Silent-Face-Anti-Spoofing** | Latest | CNN-based liveness detection (real/spoof/attack) |

**Why These Choices?**
- **YuNet**: Ultra-lightweight ONNX model; runs at 320×320 in ~50ms on CPU, ~20ms on GPU.
- **ArcFace**: Industry-standard face embedding; 512-D L2-normalized vectors enable fast cosine similarity matching.
- **ONNX Runtime**: Single inference interface across CPU, CUDA, TensorRT; automatic provider selection.
- **OpenCV CSRT**: Robust tracking without retraining; combines CNN + correlation filter.
- **Silent-Face**: Minimal compute footprint; specifically trained on presentation attacks (photos, videos, masks).

### Backend & Web Framework

| Technology | Version | Purpose |
|---|---|---|
| **Flask** | 3.0+ | Lightweight web framework for admin + student apps |
| **Flask-SocketIO** | 5.3+ | Real-time bidirectional communication (camera feed, events) |
| **Flask-Login** | 1.0+ | Session management (student portal authentication) |
| **Flask-RESTX** | 1.2+ | OpenAPI/Swagger API documentation |
| **Werkzeug** | 3.0+ | WSGI utilities and secure password hashing |

**Why These Choices?**
- **Flask**: Minimal, modular, and easy to extend with blueprints.
- **SocketIO**: Enables real-time camera overlay without polling; reduces latency for live feeds.
- **Flask-Login**: Lightweight session manager ideal for small-to-medium institutions.
- **RESTX**: Automatic API documentation reduces integration friction.

### Database

| Technology | Version | Purpose |
|---|---|---|
| **MongoDB** | 4.4+ | Document-oriented storage; flexible schema for student encodings and attendance records |
| **PyMongo** | 4.5+ | Python MongoDB driver with connection pooling and circuit breaker support |

**Why MongoDB?**
- Flexible schema: Student encodings can be 128-D (legacy dlib) or 512-D (ArcFace) without schema migrations.
- Horizontal scaling: Sharding across multiple servers for large institutions.
- Aggregation pipeline: Complex analytics (at-risk detection, trends) via server-side queries.
- Atlas integration: Managed cloud option; automatic backups and security.

### Deployment & DevOps

| Technology | Version | Purpose |
|---|---|---|
| **Docker** | 20.10+ | Containerization (CPU and GPU images) |
| **Docker Compose** | 2.0+ | Multi-service orchestration (web + student + nginx) |
| **Gunicorn** | 21.0+ | WSGI application server (production-grade) |
| **Nginx** | 1.27+ | Reverse proxy, load balancing, static file serving |
| **Python** | 3.9–3.13 | Target: 3.12 by default (3.9+ compatible) |

**Why These Choices?**
- **Docker**: Reproducible builds across dev/test/prod; GPU support via NVIDIA Docker.
- **Gunicorn**: Battle-tested WSGI server; simple configuration; supports preload + worker restart.
- **Nginx**: High-performance reverse proxy; offloads static assets and SSL termination.

---

## Quick Start Guide

### Prerequisites

- **Python 3.9+** (tested on 3.12)
- **MongoDB Atlas** account or local MongoDB server
- **Webcam** or IP camera (for attendance marking)
- **GPU (optional)**: NVIDIA GPU with CUDA 12.1 for 3–5× speedup

### 1. Local Setup (CPU)

```bash
# Clone the repository
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set MONGO_URI, EMBEDDING_BACKEND=arcface, etc.
nano .env

# Download models
python scripts/verify_versions.py    # Verify installation
python scripts/download_models.py    # Download YuNet + Anti-Spoof models

# Run both apps (development)
python run.py
# Admin:   http://localhost:5000
# Student: http://localhost:5001
```

### 2. Docker Setup (CPU)

```bash
# Build images
docker compose build

# Run services
docker compose up -d

# View logs
docker compose logs -f web student-web nginx

# Stop services
docker compose down
```

### 3. Docker Setup (GPU)

```bash
# Build GPU images (requires NVIDIA Docker)
INSTALL_GPU=1 docker compose build

# Run with GPU
docker compose up -d

# Verify GPU is active
docker compose exec web python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Admin Workflow

1. **Bootstrap Admin User**:
   ```bash
   python scripts/bootstrap_admin.py --username admin --password <secure-password>
   ```

2. **Register Students** (single or batch):
   - Admin Panel → Manage Students → Batch Import (CSV + ZIP)
   - Or: Student Portal → Self-service enrollment

3. **Create Attendance Session**:
   - Admin Panel → Camera → Start Session (select course + camera)
   - Real-time face recognition begins

4. **View Attendance**:
   - Admin Panel → Reports → Download CSV or view Heatmap

### 5. Student Workflow

1. **Enroll**:
   - Student Portal → Register
   - Capture 3–5 face samples via webcam
   - System auto-evaluates; admin approves if pending

2. **Mark Attendance**:
   - System recognizes student face once during active session
   - Automatically marks present once per day

3. **Check Status**:
   - Student Portal → View Attendance History

---

## Project Structure

```
attendance_system/
├─ README.md                          # This file
├─ docs/                              # Technical documentation
│  ├─ ARCHITECTURE.md                 # System design & components
│  ├─ THEORY.md                       # ML concepts (YuNet, ArcFace, liveness)
│  ├─ PIPELINE.md                     # Frame processing pipeline details
│  ├─ DATABASE.md                     # MongoDB schema & queries
│  ├─ BACKEND.md                      # Flask routes & API design
│  ├─ DEPLOYMENT.md                   # Docker, environment, scaling
│  ├─ TESTING.md                      # Unit & integration tests
│  ├─ RESEARCH.md                     # Research paper support
│  └─ APPENDIX.md                     # Glossary, config reference, commands
├─ admin_app/                         # Admin Flask application
│  ├─ app.py                          # Admin app factory & startup
│  ├─ forms.py                        # Admin forms (WTForms)
│  └─ routes/                         # Admin-specific routes
├─ student_app/                       # Student Portal Flask application
│  ├─ app.py                          # Student app factory
│  ├─ auth.py                         # Flask-Login integration
│  ├─ database.py                     # Student-specific DB helpers
│  ├─ routes.py                       # Student portal routes
│  ├─ verification.py                 # Onboarding verification pipeline
│  ├─ templates/                      # Student HTML templates
│  └─ static/                         # Student portal assets
├─ camera/                            # Real-time camera loop
│  └─ camera.py                       # Threaded capture & processing
├─ vision/                            # ML pipeline components
│  ├─ pipeline.py                     # YuNet detection + CSRT tracking
│  ├─ recognition.py                  # Alignment & encoding
│  ├─ face_engine.py                  # ArcFace backend
│  ├─ anti_spoofing.py                # Liveness detection
│  ├─ ppe_detection.py                # PPE/mask detection (optional)
│  ├─ preprocessing.py                # CLAHE normalization
│  └─ overlay.py                      # Visualization helpers
├─ anti_spoofing/                     # Anti-spoofing model wrapper
│  ├─ model.py                        # Model initialization
│  ├─ spoof_detector.py               # Spoof detection logic
│  ├─ blink_detector.py               # Eye aspect ratio tracking
│  └─ movement_checker.py             # Motion heuristics
├─ recognition/                       # Recognition module (compat layer)
│  ├─ pipeline.py                     # Redirect to vision.pipeline
│  ├─ detector.py                     # YuNet redirect
│  ├─ embedder.py                     # Embedding cache + ArcFace
│  ├─ matcher.py                      # Cosine similarity matching
│  ├─ aligner.py                      # Face alignment
│  └─ tracker.py                      # CSRT tracker management
├─ core/                              # Shared core modules
│  ├─ config.py                       # Configuration (80+ parameters)
│  ├─ database.py                     # MongoDB connection & CRUD
│  ├─ models.py                       # Data Access Objects (DAOs)
│  ├─ auth.py                         # Password hashing & validation
│  ├─ extensions.py                   # Shared Flask extensions
│  ├─ utils.py                        # Logging, validation, helpers
│  ├─ notifications.py                # Absence alerts (optional)
│  ├─ performance.py                  # Metrics tracking
│  └─ profiling.py                    # Latency profiling
├─ web/                               # Shared web routes
│  ├─ routes.py                       # Blueprint coordinator
│  ├─ decorators.py                   # RBAC decorators (ENABLE_RBAC flag)
│  ├─ attendance_routes.py            # Attendance API
│  ├─ camera_routes.py                # Camera & SocketIO endpoints
│  ├─ registration_routes.py          # Student enrollment
│  ├─ student_routes.py               # Student admin routes
│  ├─ report_routes.py                # CSV export & analytics
│  ├─ auth_routes.py                  # Authentication
│  ├─ health_routes.py                # Health checks
│  └─ routes_helpers.py               # Validation utilities
├─ tasks/                             # Celery background tasks (optional)
│  ├─ celery_app.py                   # Celery app initialization
│  ├─ embedding_tasks.py              # Async encoding generation
│  ├─ cleanup_tasks.py                # Data cleanup & archival
│  └─ report_tasks.py                 # Report generation
├─ scripts/                           # Utility scripts
│  ├─ bootstrap_admin.py              # Create admin user
│  ├─ download_models.py              # Fetch YuNet + Anti-Spoof models
│  ├─ verify_versions.py              # Verify installation
│  ├─ seed_demo_data.py               # Populate demo students
│  ├─ calibrate_liveness_threshold.py # Tune liveness sensitivity
│  └─ migrate_encodings.py            # dlib → ArcFace migration
├─ templates/                         # Shared HTML templates
│  ├─ base.html                       # Base layout
│  ├─ attendance.html                 # Attendance view
│  ├─ dashboard.html                  # Admin dashboard
│  └─ ... (other shared templates)
├─ static/                            # Shared static assets
│  ├─ css/                            # Stylesheets
│  ├─ js/                             # JavaScript
│  └─ favicon.svg                     # Favicon
├─ tests/                             # Test suite
│  ├─ test_camera_pipeline.py         # Camera loop tests
│  ├─ test_anti_spoofing.py           # Liveness tests
│  ├─ test_recognition.py             # Encoding & matching tests
│  ├─ test_routes.py                  # Flask route tests
│  └─ test_security_rbac.py           # RBAC tests
├─ docker/                            # Docker configurations
│  ├─ Dockerfile                      # Admin app image
│  ├─ Dockerfile.student              # Student app image
│  └─ nginx/                          # Nginx reverse proxy config
├─ deploy/                            # Deployment configs
│  ├─ k8s/                            # Kubernetes manifests (optional)
│  └─ nginx/                          # Nginx configuration
├─ models/                            # Pre-trained models
│  ├─ face_detection_yunet_2023mar.onnx
│  ├─ shape_predictor_68_face_landmarks.dat
│  └─ ppe_mask_cap.onnx
├─ Silent-Face-Anti-Spoofing/         # Anti-spoofing submodule
├─ backups/                           # Database backups
├─ logs/                              # Application logs
├─ uploads/                           # Temporary student samples
├─ unknown_faces/                     # Unknown face snapshots
└─ config files
   ├─ .env.example                    # Environment template
   ├─ .gitignore                      # Git ignore rules
   ├─ docker-compose.yml              # Multi-service composition
   ├─ gunicorn.conf.py                # Gunicorn server config
   ├─ Makefile                        # Common commands
   ├─ pyproject.toml                  # Python package metadata
   ├─ requirements.txt                # Python dependencies (base)
   ├─ requirements-docker.txt         # Minimal Docker deps
   └─ app.py                          # Admin app wrapper (compat)
```

---

## Configuration

### Environment Variables

All configuration is driven by environment variables in `.env`. Key parameters:

#### Database

```bash
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/attendance_system
DATABASE_NAME=attendance_system
MONGO_MAX_POOL_SIZE=50
MONGO_CIRCUIT_BREAKER_THRESHOLD=5
```

#### ML Pipeline

```bash
EMBEDDING_BACKEND=arcface          # or 'dlib' for legacy
RECOGNITION_THRESHOLD=0.38         # Cosine similarity threshold
RECOGNITION_MIN_CONFIDENCE=0.46    # Minimum match score
RECOGNITION_CONFIRM_FRAMES=2       # Frames for multi-frame voting
LIVENESS_CONFIDENCE_THRESHOLD=0.55 # Liveness confidence gate
BLUR_THRESHOLD=6.0                 # Minimum sharpness (Laplacian var)
BRIGHTNESS_THRESHOLD=40            # Min brightness (0–255)
```

#### Camera & Performance

```bash
FRAME_PROCESS_WIDTH=512            # Resize frames for faster processing
DETECTION_INTERVAL=6               # Run detection every N frames
FRAME_RESIZE_FACTOR=0.25           # Additional scaling factor
PERF_MAX_FACES=5                   # Max simultaneous tracked faces
PERF_USE_KCF_TRACKER=0             # Use faster KCF (less accurate)
```

#### Attendance & Sessions

```bash
ATTENDANCE_SESSION_IDLE_TIMEOUT_SECONDS=900  # Auto-close after 15min idle
RECOGNITION_COOLDOWN=30            # Skip re-recognition for 30 frames
```

#### Student Portal

```bash
STUDENT_APP_HOST=0.0.0.0
STUDENT_APP_PORT=5001
STUDENT_MIN_CAPTURE_IMAGES=3
STUDENT_MAX_CAPTURE_IMAGES=5
STUDENT_AUTO_APPROVE_SCORE=85      # Auto-approve threshold
STUDENT_PENDING_SCORE=60           # Manual review threshold
```

#### GPU Support

```bash
ENABLE_GPU_PROVIDERS=1
ONNXRT_PROVIDER_PRIORITY=CUDAExecutionProvider,CPUExecutionProvider
INSTALL_GPU=0                      # Set to 1 in docker build
```

See [docs/APPENDIX.md](docs/APPENDIX.md#configuration-reference) for the complete configuration reference.

---

## API Overview

### Authentication

- **Admin**: Uses Flask session-based auth (login form, cookies).
- **Student**: Flask-Login session management.
- **API**: Optional role-based decorators (if `ENABLE_RBAC=1`).

### Key REST Endpoints

#### Admin APIs

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/attendance/sessions` | Start attendance session |
| `POST` | `/api/attendance/sessions/<id>/end` | End session |
| `GET` | `/api/attendance` | List attendance records |
| `POST` | `/api/attendance` | Mark single attendance |
| `POST` | `/api/register` | Single student enrollment |
| `POST` | `/api/register/batch` | Batch student import |
| `GET` | `/api/reports/csv` | CSV export attendance |

#### SocketIO Events (Real-Time)

| Event | Direction | Data |
|---|---|---|
| `connect` | Client ← Server | Session established |
| `frame` | Client ← Server | MJPEG stream frame |
| `attendance_event` | Client ← Server | `{student_id, name, confidence, timestamp}` |
| `camera_status` | Client ← Server | `{fps, tracked_faces, session_active}` |

See [docs/BACKEND.md](docs/BACKEND.md) for full API specification.

---

## Documentation Index

For detailed technical information, see:

1. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
   - Component breakdown
   - Module interactions
   - Design patterns

2. **[docs/THEORY.md](docs/THEORY.md)**
   - Face detection (YuNet)
   - Face recognition (ArcFace embeddings, cosine similarity)
   - Anti-spoofing (Silent-Face, blink detection, motion heuristics)
   - Alignment and quality gating
   - Mathematical formulas and trade-offs

3. **[docs/PIPELINE.md](docs/PIPELINE.md)**
   - Step-by-step processing pipeline
   - Frame → detection → tracking → alignment → embedding → matching → anti-spoofing → confirmation
   - Optimization techniques

4. **[docs/DATABASE.md](docs/DATABASE.md)**
   - MongoDB collections and schema
   - Indexes and query patterns
   - Session lifecycle
   - Attendance uniqueness enforcement

5. **[docs/BACKEND.md](docs/BACKEND.md)**
   - Flask app structure
   - Blueprint registration
   - Authentication and RBAC
   - Complete API reference

6. **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**
   - Local, Docker, and production setup
   - Environment configuration
   - Scaling and monitoring

7. **[docs/TESTING.md](docs/TESTING.md)**
   - Unit and integration tests
   - ML pipeline validation
   - Edge cases and regression checks

8. **[docs/RESEARCH.md](docs/RESEARCH.md)**
   - System novelty and contributions
   - Evaluation metrics (accuracy, precision, recall, FAR, FRR)
   - Comparison with traditional systems

9. **[docs/APPENDIX.md](docs/APPENDIX.md)**
   - Complete configuration reference
   - Useful commands and troubleshooting
   - Glossary of technical terms

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make changes and add tests
4. Commit with clear messages
5. Push and open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **YuNet**: Ultra-lightweight face detection model (OpenCV)
- **InsightFace**: ArcFace embeddings and face analysis
- **Silent-Face-Anti-Spoofing**: CNN-based liveness detection
- **OpenCV**: Computer vision library
- **Flask**: Web framework
- **MongoDB**: Document database

---

## Contact & Support

For issues, feature requests, or questions:

- **GitHub Issues**: [ShubhamPatra/attendance_system/issues](https://github.com/ShubhamPatra/attendance_system/issues)
- **Email**: [contact@shubhampatra.com](mailto:contact@shubhampatra.com)

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0 | **Status**: Production-Ready
