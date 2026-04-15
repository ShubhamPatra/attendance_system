# AutoAttendance — Complete Implementation Plan

> **Project:** AutoAttendance — Production-Grade Face Recognition Attendance System
> **Version:** 1.0.0
> **Date:** April 15, 2026
> **Status:** Planning

---

## 1. PROJECT OVERVIEW

### 1.1 System Description

AutoAttendance is a production-grade, real-time face recognition attendance system designed for educational institutions. It uses state-of-the-art deep learning models (ArcFace for embedding, YuNet for detection, MiniFASNet for anti-spoofing) to automatically verify student identities and record attendance — via backend-controlled camera feeds (webcam + IP cameras), with no student interaction required.

The system consists of two web applications (Admin dashboard and Student view-only portal), a camera management subsystem, a background attendance processing service, a high-performance ML inference pipeline, a background job queue, and a real-time dashboard — all deployed via Docker with Nginx reverse proxy.

> [!IMPORTANT]
> **Architecture Principle:** Attendance is captured ONLY via camera feeds controlled by the admin. Students cannot manually trigger attendance. The student portal is strictly read-only for attendance data.

### 1.2 Key Components

| Component | Technology | Purpose |
|---|---|---|
| Admin App | Flask + Jinja2 | Manage students, courses, cameras, sessions, reports, dashboard |
| Student Portal | Flask + Jinja2 | View-only: attendance history, status, percentage |
| **Camera Service** | OpenCV + Threading | Multi-camera management (webcam, IP cam, RTSP) |
| **Attendance Service** | Background threads | Automated attendance processing from camera feeds |
| Face Detection | OpenCV YuNet (DNN) | Real-time face detection at ~30 FPS |
| Face Tracking | OpenCV CSRT Tracker | Multi-face tracking, reduce per-frame detection overhead |
| Face Embedding | InsightFace ArcFace (ONNX) | 512-D face embeddings for matching |
| Anti-Spoofing | MiniFASNet (PyTorch/ONNX) + Behavioral | Liveness verification (triggered on match) |
| Database | MongoDB Atlas | Persistent storage (students, courses, cameras, attendance, embeddings) |
| Background Jobs | Celery + Redis | Async embedding generation, report export, cleanup |
| Real-Time Dashboard | Flask-SocketIO | Live camera feed, recognized students, statistics |
| Deployment | Docker Compose + Nginx | Containerized production deployment |

### 1.3 Design Principles

- **High Accuracy:** Multi-frame smoothing + ArcFace (99.83% LFW) + multi-layer anti-spoofing
- **Low Latency:** Detection every N frames, tracking in between; frame scaling; embedding cache
- **Modular:** Each pipeline stage is independently testable and replaceable
- **Secure:** Backend-controlled attendance only; students cannot trigger attendance; RBAC, anti-spoofing
- **Scalable:** Celery workers scale horizontally; stateless web tier
- **Automated:** Camera-based system runs independently — no student interaction required
- **Multi-Camera:** Support webcam + IP cameras + RTSP for flexible deployment

---

## 2. HIGH-LEVEL ARCHITECTURE

### 2.1 System Architecture Diagram

```mermaid
graph TB
    subgraph Camera Sources
        WC[Webcam - VideoCapture 0] --> CM[Camera Manager]
        IP[IP Camera - DroidCam/EpocCam] --> CM
        RT[RTSP Stream] --> CM
    end

    subgraph Client
        A[Admin Browser] -->|HTTPS| NG[Nginx Reverse Proxy]
        B[Student Browser - View Only] -->|HTTPS| NG
    end

    subgraph Backend
        NG -->|/admin/*| AA[Admin App - Flask :5000]
        NG -->|/student/*| SA[Student App - Flask :5001]
        AA -->|Start/Stop Session| APS[Attendance Processing Service]
        AA -->|SocketIO| DASH[Real-Time Dashboard]
        CM -->|Frames| APS
    end

    subgraph ML Pipeline
        APS --> DET[YuNet Face Detector]
        APS --> TRK[Multi-Face CSRT Tracker]
        APS --> ALN[5-Point Landmark Aligner]
        APS --> EMB[ArcFace Embedding - ONNX]
        APS --> ASP[Anti-Spoofing Module]
        ASP --> MFA[MiniFASNet Model]
        ASP --> BLK[Blink Detection]
        ASP --> MVT[Head Movement Check]
    end

    subgraph Data Layer
        AA --> MDB[(MongoDB Atlas)]
        SA --> MDB
        APS --> MDB
        AA --> RDS[(Redis)]
        RDS --> CEL[Celery Workers]
        CEL --> MDB
    end
```

### 2.2 Module Interaction Map

```mermaid
flowchart LR
    subgraph Web Tier
        AUTH[auth module] --> ADMIN[admin routes]
        AUTH --> STUDENT[student routes - view only]
        ADMIN --> DB[database module]
        STUDENT --> DB
    end

    subgraph Camera Tier
        CAM_MGR[camera manager] --> CAM_SRC[camera sources]
        CAM_SRC --> WEB[webcam]
        CAM_SRC --> IPC[IP camera]
        CAM_SRC --> RTSP[RTSP stream]
    end

    subgraph Attendance Processing Tier
        ADMIN -->|start/stop session| ATT_SVC[attendance service]
        ATT_SVC --> CAM_MGR
        ATT_SVC --> RECOG[recognition pipeline]
        ATT_SVC --> ANTISPOOF[anti-spoofing module]
        ATT_SVC --> ID_TRACK[identity tracker]
        RECOG --> DETECT[detector]
        RECOG --> TRACK[face tracker]
        RECOG --> ALIGN[aligner]
        RECOG --> EMBED[embedder]
        RECOG --> MATCH[matcher]
    end

    subgraph Infra Tier
        ATT_SVC --> DB
        ADMIN --> CELERY_T[celery tasks]
        CELERY_T --> DB
        ATT_SVC --> SOCKETIO[socketio events]
    end
```

---

## 3. DEPENDENCY LIST

> [!IMPORTANT]
> All versions below have been verified for mutual compatibility with **Python 3.11.x**.
> Install `onnxruntime` OR `onnxruntime-gpu` — **never both** in the same environment.
> NumPy is pinned to `<2.0` to ensure compatibility with `insightface 0.7.3`.

### 3.1 Python Version

```
Python 3.11.9
```

> **Rationale:** Python 3.11 provides the best compatibility across all ML/CV libraries used in this project. Python 3.12+ introduces breaking changes for some ONNX/InsightFace dependencies. Python 3.11 also offers 10-60% speed improvements over 3.10.

---

### 3.2 Full `requirements.txt`

```txt
# ============================================================
# AUTOATTENDANCE — DEPENDENCY MANIFEST
# Python 3.11.9 | Verified April 2026
# ============================================================

# ----------------------------------------------------------
# 1. BACKEND FRAMEWORK & EXTENSIONS
# ----------------------------------------------------------
Flask==3.1.3
Flask-Login==0.6.3
Flask-SocketIO==5.6.1
Flask-Limiter==3.12.0
Flask-RESTX==1.3.0
Flask-WTF==1.2.2
Werkzeug==3.1.3
Jinja2==3.1.6
eventlet==0.39.1                # Async server for SocketIO
python-socketio==5.12.1         # SocketIO protocol
python-engineio==4.12.1         # Engine.IO transport

# ----------------------------------------------------------
# 2. ML / COMPUTER VISION
# ----------------------------------------------------------
opencv-contrib-python==4.13.0.92   # Includes DNN, CSRT tracker, extra modules
insightface==0.7.3                  # ArcFace embedding models
numpy>=1.24.0,<2.0                  # Pinned below 2.0 for insightface compat
scipy==1.15.3                       # Cosine similarity, spatial transforms
Pillow==11.2.1                      # Image I/O and manipulation

# ----------------------------------------------------------
# 3. ONNX RUNTIME (choose ONE)
# ----------------------------------------------------------
# CPU-only:
onnxruntime==1.24.4
# GPU (CUDA 12.x) — replace above with:
# onnxruntime-gpu==1.24.4

# ----------------------------------------------------------
# 4. ANTI-SPOOFING (PyTorch)
# ----------------------------------------------------------
# Install via: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Or for CUDA 12.6: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
torch==2.11.0
torchvision==0.26.0

# ----------------------------------------------------------
# 5. DATABASE
# ----------------------------------------------------------
pymongo==4.16.0
pymongo[srv]                       # For MongoDB Atlas SRV connection strings
dnspython==2.7.0                   # Required by pymongo SRV

# ----------------------------------------------------------
# 6. ASYNC / QUEUE
# ----------------------------------------------------------
celery==5.6.3
redis==7.4.0

# ----------------------------------------------------------
# 7. DEVOPS & UTILITIES
# ----------------------------------------------------------
gunicorn==23.0.0
python-dotenv==1.1.0
click==8.2.1

# ----------------------------------------------------------
# 8. TESTING
# ----------------------------------------------------------
pytest==8.4.0
pytest-cov==6.1.0
pytest-flask==1.3.0
pytest-mock==3.14.0
mongomock==4.3.0                   # Mock MongoDB for unit tests

# ----------------------------------------------------------
# 9. MISC
# ----------------------------------------------------------
bcrypt==4.3.0                      # Password hashing
email-validator==2.2.0             # Email format validation
Markdown==3.7.0                    # Report rendering
apscheduler==3.11.0                # Scheduled tasks (backup to Celery)
```

### 3.3 Special Installation Notes

> [!WARNING]
> **PyTorch:** Do NOT install via the default `pip install torch`. Use the official PyTorch index URL to get the correct CPU or CUDA build:
> ```bash
> # CPU only
> pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
>
> # CUDA 12.6
> pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
> ```

> [!WARNING]
> **ONNX Runtime:** Install ONLY ONE of `onnxruntime` or `onnxruntime-gpu`. Installing both causes import conflicts:
> ```bash
> # CPU
> pip install onnxruntime==1.24.4
>
> # GPU (match your CUDA version)
> pip install onnxruntime-gpu==1.24.4
> ```

> [!NOTE]
> **InsightFace Models:** On first run, `insightface` will download the `buffalo_l` model pack (~300 MB). Ensure the deployment environment has internet access during initial setup, or pre-download models to `~/.insightface/models/`.

> [!NOTE]
> **Anti-Spoofing Models:** The MiniFASNet ONNX models must be placed in `models/anti_spoofing/`. These are converted from the Silent-Face-Anti-Spoofing PyTorch checkpoints. Pre-converted ONNX files are included in the repository.

---

## 4. IMPLEMENTATION PHASES

---

### PHASE 1: Project Setup

**Objective:** Initialize project structure, virtual environment, configuration system, and version control.

**Tasks:**

- ☐ Create root project directory `AutoAttendance/`
- ☐ Initialize Git repository with `.gitignore` (Python, venv, `.env`, `__pycache__`, model binaries)
- ☐ Create Python 3.11 virtual environment
  ```bash
  python -m venv venv
  ```
- ☐ Create `requirements.txt` with all dependencies from Section 3.2
- ☐ Install all dependencies (CPU mode initially)
  ```bash
  pip install -r requirements.txt
  pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
  ```
- ☐ Create project directory structure:
  ```
  AutoAttendance/
  ├── admin_app/
  │   ├── __init__.py
  │   ├── routes/
  │   │   ├── __init__.py
  │   │   ├── auth.py
  │   │   ├── dashboard.py
  │   │   ├── students.py
  │   │   ├── courses.py
  │   │   ├── attendance.py
  │   │   └── reports.py
  │   ├── templates/
  │   │   ├── base.html
  │   │   ├── auth/
  │   │   ├── dashboard/
  │   │   ├── students/
  │   │   ├── courses/
  │   │   └── reports/
  │   ├── static/
  │   │   ├── css/
  │   │   ├── js/
  │   │   └── img/
  │   └── forms.py
  ├── student_app/
  │   ├── __init__.py
  │   ├── routes/
  │   │   ├── __init__.py
  │   │   ├── auth.py
  │   │   ├── attendance.py          # VIEW-ONLY: history, status, percentage
  │   │   └── profile.py
  │   ├── templates/
  │   │   ├── base.html
  │   │   ├── auth/
  │   │   ├── attendance/              # View-only templates (history, status)
  │   │   └── profile/
  │   └── static/
  │       ├── css/
  │       └── img/
  ├── core/
  │   ├── __init__.py
  │   ├── config.py
  │   ├── database.py
  │   ├── models.py
  │   ├── extensions.py
  │   └── utils.py
  ├── recognition/
  │   ├── __init__.py
  │   ├── detector.py
  │   ├── tracker.py
  │   ├── aligner.py
  │   ├── embedder.py
  │   ├── matcher.py
  │   ├── pipeline.py
  │   └── config.py
  ├── anti_spoofing/
  │   ├── __init__.py
  │   ├── model.py
  │   ├── blink_detector.py
  │   ├── movement_checker.py
  │   └── spoof_detector.py
  ├── camera/
  │   ├── __init__.py
  │   ├── manager.py                # CameraManager: init, manage, switch sources
  │   ├── sources.py                # WebcamSource, IPCameraSource, RTSPSource
  │   ├── stream.py                 # Threaded frame capture, reconnection
  │   └── config.py                 # Camera configuration dataclass
  ├── attendance_service/
  │   ├── __init__.py
  │   ├── processor.py              # AttendanceProcessor: frame processing pipeline
  │   ├── session_manager.py        # Admin-controlled session lifecycle
  │   ├── tracker.py                # Multi-face identity tracker
  │   └── service.py                # Background service runner (thread/process)
  ├── tasks/
  │   ├── __init__.py
  │   ├── celery_app.py
  │   ├── embedding_tasks.py
  │   ├── report_tasks.py
  │   └── cleanup_tasks.py
  ├── models/
  │   ├── face_detection/
  │   │   └── face_detection_yunet_2023mar.onnx
  │   └── anti_spoofing/
  │       ├── 2.7_80x80_MiniFASNetV2.onnx
  │       └── 4_0_0_80x80_MiniFASNetV1SE.onnx
  ├── tests/
  │   ├── __init__.py
  │   ├── conftest.py
  │   ├── test_auth.py
  │   ├── test_database.py
  │   ├── test_recognition.py
  │   ├── test_anti_spoofing.py
  │   ├── test_verification.py
  │   ├── test_admin_routes.py
  │   └── test_student_routes.py
  ├── docker/
  │   ├── Dockerfile.admin
  │   ├── Dockerfile.student
  │   ├── Dockerfile.celery
  │   └── nginx/
  │       ├── Dockerfile
  │       └── nginx.conf
  ├── scripts/
  │   ├── seed_db.py
  │   ├── download_models.py
  │   └── convert_antispoof_to_onnx.py
  ├── docker-compose.yml
  ├── .env.example
  ├── .gitignore
  ├── requirements.txt
  ├── run_admin.py
  ├── run_student.py
  └── README.md
  ```
- ☐ Create `.env.example` with all required environment variables:
  ```env
  # Flask
  FLASK_SECRET_KEY=your-secret-key-here
  FLASK_ENV=development
  FLASK_DEBUG=1

  # MongoDB Atlas
  MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/autoattendance
  MONGODB_DB_NAME=autoattendance

  # Redis
  REDIS_URL=redis://localhost:6379/0

  # Celery
  CELERY_BROKER_URL=redis://localhost:6379/0
  CELERY_RESULT_BACKEND=redis://localhost:6379/1

  # Recognition
  RECOGNITION_MODE=BALANCED
  FACE_DETECTION_CONFIDENCE=0.7
  RECOGNITION_THRESHOLD=0.45
  ANTI_SPOOFING_THRESHOLD=0.5

  # Camera
  CAMERA_DEFAULT_RESOLUTION=640x480
  CAMERA_DEFAULT_FPS=15
  CAMERA_RECONNECT_ATTEMPTS=10
  CAMERA_RECONNECT_INTERVAL=2.0
  # IP Camera (DroidCam/EpocCam) — set at runtime or via admin panel
  # IP_CAMERA_URL=http://192.168.1.100:4747/video

  # Attendance Processing
  ATTENDANCE_MIN_CONFIRMATION_FRAMES=5
  ATTENDANCE_MIN_DURATION_SECONDS=3
  ATTENDANCE_PROCESS_INTERVAL_FRAMES=5
  ATTENDANCE_FRAME_SKIP=2

  # Admin
  ADMIN_EMAIL=admin@autoattendance.com
  ADMIN_DEFAULT_PASSWORD=changeme123

  # SocketIO
  SOCKETIO_ASYNC_MODE=eventlet
  ```
- ☐ Create `core/config.py` — centralized configuration class reading from `.env`
- ☐ Verify installation by importing all major libraries:
  ```python
  python -c "import flask, cv2, insightface, onnxruntime, torch, pymongo, celery, redis; print('All imports OK')"
  ```
- ☐ Create initial `README.md` with project description, setup instructions, and architecture overview

**Dependencies:** None (first phase)

**Expected Output:**
- Complete project scaffold
- Working virtual environment with all dependencies installed
- Configuration system operational
- Git repo initialized with first commit

---

### PHASE 2: Database Design

**Objective:** Design and implement MongoDB schema, connection management, and data access layer.

**Tasks:**

- ☐ Design MongoDB collections schema:

  **`admins` Collection:**
  ```json
  {
    "_id": "ObjectId",
    "email": "string (unique, indexed)",
    "password_hash": "string (bcrypt)",
    "name": "string",
    "role": "string (super_admin | admin | viewer)",
    "is_active": "boolean",
    "created_at": "datetime",
    "updated_at": "datetime",
    "last_login": "datetime"
  }
  ```

  **`students` Collection:**
  ```json
  {
    "_id": "ObjectId",
    "student_id": "string (unique, indexed)",
    "email": "string (unique, indexed)",
    "password_hash": "string (bcrypt)",
    "name": "string",
    "department": "string",
    "year": "integer",
    "enrolled_courses": ["ObjectId (ref: courses)"],
    "face_embeddings": [
      {
        "embedding": "Binary (512-D float32 array, BinData)",
        "quality_score": "float",
        "created_at": "datetime",
        "source": "string (admin_upload | self_register)"
      }
    ],
    "face_photo_path": "string (path to reference photo)",
    "is_active": "boolean",
    "created_at": "datetime",
    "updated_at": "datetime"
  }
  ```

  **`courses` Collection:**
  ```json
  {
    "_id": "ObjectId",
    "course_code": "string (unique, indexed)",
    "course_name": "string",
    "department": "string",
    "instructor": "string",
    "schedule": [
      {
        "day": "string (Monday-Sunday)",
        "start_time": "string (HH:MM)",
        "end_time": "string (HH:MM)",
        "room": "string"
      }
    ],
    "enrolled_students": ["ObjectId (ref: students)"],
    "attendance_window_minutes": "integer (default: 15)",
    "is_active": "boolean",
    "created_at": "datetime"
  }
  ```

  **`attendance_records` Collection:**
  ```json
  {
    "_id": "ObjectId",
    "student_id": "ObjectId (ref: students, indexed)",
    "course_id": "ObjectId (ref: courses, indexed)",
    "session_id": "ObjectId (ref: attendance_sessions, indexed)",
    "date": "date (indexed)",
    "entry_time": "datetime",
    "exit_time": "datetime | null",
    "status": "string (present | late | absent)",
    "verification_method": "string (camera_auto | manual)",
    "confidence_score": "float",
    "anti_spoofing_score": "float",
    "camera_id": "string (ref: camera_configs.camera_id) | null",
    "frames_matched": "integer",
    "tracking_duration_seconds": "float",
    "created_at": "datetime"
  }
  ```

  **`attendance_sessions` Collection:**
  ```json
  {
    "_id": "ObjectId",
    "course_id": "ObjectId (ref: courses, indexed)",
    "date": "date",
    "opened_by": "ObjectId (ref: admins)",
    "opened_at": "datetime",
    "closed_at": "datetime | null",
    "status": "string (open | closed)",
    "camera_ids": ["string (ref: camera_configs.camera_id)"],
    "processing_mode": "string (FAST | BALANCED | ACCURATE)",
    "auto_exit_tracking": "boolean (default: false)",
    "settings": {
      "recognition_mode": "string (FAST | BALANCED | ACCURATE)",
      "anti_spoofing_enabled": "boolean",
      "late_threshold_minutes": "integer",
      "min_confirmation_frames": "integer (default: 5)",
      "min_duration_seconds": "integer (default: 3)"
    }
  }
  ```

  **`camera_configs` Collection:**
  ```json
  {
    "_id": "ObjectId",
    "camera_id": "string (unique, indexed)",
    "name": "string (e.g., 'Room 101 Camera')",
    "source_type": "string (webcam | ip_camera | rtsp)",
    "device_id": "integer | null",
    "stream_url": "string | null",
    "resolution": {"width": "integer", "height": "integer"},
    "fps": "integer",
    "location": "string",
    "is_active": "boolean",
    "created_at": "datetime",
    "updated_at": "datetime"
  }
  ```

  **`system_logs` Collection:**
  ```json
  {
    "_id": "ObjectId",
    "event_type": "string (indexed)",
    "actor_id": "ObjectId | null",
    "actor_type": "string (admin | student | system)",
    "details": "object",
    "ip_address": "string",
    "timestamp": "datetime (indexed, TTL: 90 days)"
  }
  ```

- ☐ Implement `core/database.py` — MongoDB connection manager:
  - Singleton connection pool using `pymongo.MongoClient`
  - Connection health check method
  - Automatic index creation on startup
  - Connection retry with exponential backoff
- ☐ Create indexes:
  ```python
  # students
  db.students.create_index("student_id", unique=True)
  db.students.create_index("email", unique=True)
  
  # courses
  db.courses.create_index("course_code", unique=True)
  
  # attendance_records (compound index for queries)
  db.attendance_records.create_index([("course_id", 1), ("date", 1)])
  db.attendance_records.create_index([("student_id", 1), ("date", 1)])
  
  # attendance_sessions
  db.attendance_sessions.create_index([("course_id", 1), ("date", 1)])
  
  # system_logs (TTL index for auto-cleanup)
  db.system_logs.create_index("timestamp", expireAfterSeconds=7776000)
  ```
- ☐ Implement `core/models.py` — Data access objects (DAO):
  - `AdminDAO` — CRUD for admin users
  - `StudentDAO` — CRUD for students + embedding management
  - `CourseDAO` — CRUD for courses + enrollment
  - `AttendanceDAO` — Record creation, queries, aggregations
  - `SessionDAO` — Attendance session management
  - `LogDAO` — System event logging
- ☐ Create `scripts/seed_db.py` — Database seeder with sample data:
  - 3 admin users (super_admin, admin, viewer)
  - 50 sample students across 3 departments
  - 10 courses with schedules
  - Sample attendance records for the past 30 days
- ☐ Write unit tests for all DAO methods using `mongomock`
- ☐ Verify MongoDB Atlas connectivity from development environment

**Dependencies:** Phase 1 (project structure, config)

**Expected Output:**
- Complete MongoDB schema documentation
- Working database connection with connection pooling
- All DAO classes with full CRUD operations
- Proper indexing for query performance
- Database seeder script
- Passing unit tests

---

### PHASE 3: Admin App Core

**Objective:** Build the Flask admin application with authentication, authorization, and core management features.

**Tasks:**

- ☐ Implement `admin_app/__init__.py` — Flask app factory:
  - Register blueprints (auth, dashboard, students, courses, attendance, reports)
  - Initialize extensions (Flask-Login, Flask-SocketIO, Flask-Limiter, Flask-RESTX)
  - Configure error handlers (400, 403, 404, 500)
  - Setup request logging middleware
- ☐ Implement `admin_app/routes/auth.py`:
  - ☐ `GET/POST /admin/login` — Admin login with email/password
  - ☐ `GET /admin/logout` — Session invalidation
  - ☐ `GET/POST /admin/change-password` — Password change form
  - ☐ Flask-Login `user_loader` callback (loads admin from MongoDB)
  - ☐ Login rate limiting (5 attempts per 15 minutes per IP)
- ☐ Implement RBAC decorator:
  ```python
  def role_required(*roles):
      """Restrict access to specified roles: super_admin, admin, viewer"""
  ```
- ☐ Implement `admin_app/routes/dashboard.py`:
  - ☐ `GET /admin/dashboard` — Main dashboard page
  - ☐ Real-time attendance statistics via SocketIO:
    - Total students present today
    - Attendance rate per course
    - Recent check-ins feed (last 20)
    - Department-wise breakdown
  - ☐ SocketIO event handlers:
    - `connect` — Authenticate admin connection
    - `attendance_update` — Push new attendance to dashboard
    - `stats_refresh` — Periodic statistics update
- ☐ Implement `admin_app/routes/students.py`:
  - ☐ `GET /admin/students` — Paginated student list with search/filter
  - ☐ `GET /admin/students/<id>` — Student detail view (profile, attendance history, face data)
  - ☐ `GET/POST /admin/students/add` — Add new student form
  - ☐ `POST /admin/students/<id>/edit` — Edit student info
  - ☐ `POST /admin/students/<id>/delete` — Soft-delete student
  - ☐ `POST /admin/students/<id>/upload-face` — Upload student face photo + trigger embedding generation (Celery task)
  - ☐ `POST /admin/students/bulk-upload` — CSV bulk import of students
- ☐ Implement `admin_app/routes/courses.py`:
  - ☐ `GET /admin/courses` — Course list with filtering
  - ☐ `GET/POST /admin/courses/add` — Create new course
  - ☐ `POST /admin/courses/<id>/edit` — Edit course details
  - ☐ `POST /admin/courses/<id>/enroll` — Enroll students
  - ☐ `POST /admin/courses/<id>/session/open` — Open attendance session
  - ☐ `POST /admin/courses/<id>/session/close` — Close attendance session
- ☐ Implement `admin_app/routes/attendance.py`:
  - ☐ `GET /admin/attendance` — Attendance overview (today, by course)
  - ☐ `GET /admin/attendance/session/<id>` — Live session view with real-time camera feed and recognized students
  - ☐ `POST /admin/attendance/manual` — Manually mark attendance
  - ☐ `POST /admin/attendance/<id>/override` — Override attendance status
  - ☐ **Camera Session Controls:**
    - ☐ `GET /admin/attendance/cameras` — View and manage camera configurations
    - ☐ `POST /admin/attendance/cameras/add` — Register a new camera source (webcam/IP/RTSP)
    - ☐ `POST /admin/attendance/cameras/<id>/edit` — Edit camera settings
    - ☐ `POST /admin/attendance/cameras/<id>/test` — Test camera connection
    - ☐ `POST /admin/attendance/session/start` — Start attendance session with camera assignment
    - ☐ `POST /admin/attendance/session/<id>/stop` — Stop attendance session
    - ☐ `GET /admin/attendance/session/<id>/live` — Live camera feed with face detection overlay
    - ☐ `GET /admin/attendance/session/<id>/monitor` — Real-time monitoring of recognized students
- ☐ Implement `admin_app/routes/reports.py`:
  - ☐ `GET /admin/reports` — Report generation page
  - ☐ `POST /admin/reports/generate` — Generate report (triggers Celery task):
    - Per-student attendance report
    - Per-course attendance report
    - Department summary
    - Date range filter
  - ☐ `GET /admin/reports/download/<id>` — Download generated report (CSV/PDF)
- ☐ Implement `admin_app/forms.py` — Flask-WTF form classes:
  - `LoginForm`, `StudentForm`, `CourseForm`, `AttendanceFilterForm`, `ReportForm`
- ☐ Create `run_admin.py` — Admin app entry point:
  ```python
  from admin_app import create_app
  app = create_app()
  socketio.run(app, host='0.0.0.0', port=5000)
  ```

**Dependencies:** Phase 1 (structure), Phase 2 (database)

**Expected Output:**
- Fully functional admin web application
- Working authentication with RBAC
- CRUD for students and courses
- Attendance session management
- Real-time dashboard via SocketIO
- Report generation pipeline
- All routes protected with proper authorization

---

### PHASE 4: Student Portal (View-Only)

**Objective:** Build the student-facing web application with authentication and **read-only** attendance views. Students can ONLY view their attendance data — no attendance marking functionality.

> [!IMPORTANT]
> All attendance marking functionality has been removed from the student portal. Attendance is captured exclusively via backend-controlled camera feeds (see Phase 4A and Phase 7).

**Tasks:**

- ☐ Implement `student_app/__init__.py` — Flask app factory:
  - Register blueprints (auth, attendance, profile)
  - Initialize extensions
- ☐ Implement `student_app/routes/auth.py`:
  - ☐ `GET/POST /student/login` — Student login (student ID + password)
  - ☐ `GET /student/logout` — Logout
  - ☐ Login rate limiting (5 attempts per 15 minutes)
- ☐ Implement `student_app/routes/attendance.py` (**VIEW-ONLY**):
  - ☐ `GET /student/attendance/history` — View personal attendance history with course filter and date range
  - ☐ `GET /student/attendance/status` — View today's attendance status per enrolled course
  - ☐ `GET /student/attendance/percentage` — View attendance percentage per course with visual indicator
  - ☐ **NO** `/student/attendance/mark` route
  - ☐ **NO** `/student/attendance/verify` route
  - ☐ **NO** WebSocket frame streaming endpoints
- ☐ Implement `student_app/routes/profile.py`:
  - ☐ `GET /student/profile` — View profile
  - ☐ `GET /student/profile/courses` — View enrolled courses + schedules
- ☐ Create `run_student.py` — Student app entry point:
  ```python
  from student_app import create_app
  app = create_app()
  app.run(host='0.0.0.0', port=5001)
  ```

**Dependencies:** Phase 1, Phase 2

**Expected Output:**
- Working student portal with authentication
- Read-only attendance history with filtering and date range
- Attendance percentage display per course
- Course schedule viewer
- **NO** camera access or attendance marking from student side

---

### PHASE 4A: Camera Integration

**Objective:** Build a modular camera management system supporting webcam, IP camera (DroidCam/EpocCam), and RTSP streams for development and production.

**Tasks:**

#### 4A.1 Camera Source Abstraction (`camera/sources.py`)

- ☐ Implement `CameraSource` abstract base class:
  - `open()` → initialize connection
  - `read()` → returns `(success, frame)`
  - `release()` → cleanup
  - `is_opened()` → connection status
  - `get_properties()` → resolution, FPS, source info
- ☐ Implement `WebcamSource(device_id=0)`:
  - Uses `cv2.VideoCapture(device_id)`
  - Supports device enumeration (`cv2.VideoCapture` probe 0–9)
  - Configurable resolution and FPS
- ☐ Implement `IPCameraSource(url)`:
  - Supports HTTP stream URLs:
    - DroidCam: `http://<ip>:4747/video`
    - EpocCam: `http://<ip>:8080/video`
  - Configurable timeout and retry interval
  - Connection health monitoring with heartbeat
- ☐ Implement `RTSPSource(url)`:
  - Supports RTSP stream URLs (e.g., `rtsp://<ip>:554/stream`)
  - Buffer management for RTSP lag
  - Configurable transport (TCP/UDP)

#### 4A.2 Camera Manager (`camera/manager.py`)

- ☐ Implement `CameraManager` class:
  - ☐ `add_camera(camera_id, source_config)` — register a camera source
  - ☐ `remove_camera(camera_id)` — disconnect and unregister
  - ☐ `start_camera(camera_id)` — begin frame capture
  - ☐ `stop_camera(camera_id)` — stop frame capture
  - ☐ `get_frame(camera_id)` → returns latest frame (numpy array)
  - ☐ `list_cameras()` → returns all registered cameras with status
  - ☐ `switch_source(camera_id, new_config)` — hot-swap camera source
- ☐ Auto-detect available webcams on startup
- ☐ Support simultaneous multi-camera operation
- ☐ Thread-safe camera access with `threading.Lock`

#### 4A.3 Stream Processing (`camera/stream.py`)

- ☐ Implement `CameraStream` class (threaded frame capture):
  - ☐ Dedicated capture thread per camera (decouples capture from processing)
  - ☐ Circular buffer for latest N frames (default: 5)
  - ☐ Automatic reconnection on connection drop:
    - Exponential backoff: 1s, 2s, 4s, 8s, max 30s
    - Maximum reconnection attempts: 10 (configurable)
    - Event callback on reconnection failure
  - ☐ Frame quality check (brightness, blur detection via Laplacian variance)
  - ☐ Frame timestamp metadata
  - ☐ FPS monitoring and reporting

#### 4A.4 Camera Configuration (`camera/config.py`)

- ☐ Define `CameraConfig` dataclass:
  ```python
  @dataclass
  class CameraConfig:
      camera_id: str
      source_type: str          # "webcam" | "ip_camera" | "rtsp"
      device_id: int = 0        # For webcam
      stream_url: str = ""      # For IP/RTSP
      resolution: tuple = (640, 480)
      fps: int = 15
      reconnect_attempts: int = 10
      reconnect_interval: float = 2.0
  ```
- ☐ Support loading camera configs from `.env` or MongoDB `camera_configs` collection
- ☐ Validate camera configurations at startup

**Dependencies:** Phase 1 (structure), Phase 5 (recognition pipeline uses camera frames)

**Expected Output:**
- Modular camera abstraction supporting 3 source types (webcam, IP, RTSP)
- CameraManager handling multiple simultaneous cameras
- Threaded frame capture with automatic reconnection
- Quality-filtered frame output ready for recognition pipeline
- Hot-swappable camera sources at runtime

---

### PHASE 5: Face Recognition Pipeline

**Objective:** Implement the complete face recognition pipeline from detection to matching.

**Tasks:**

#### 5.1 Face Detector (`recognition/detector.py`)

- ☐ Implement `YuNetDetector` class:
  - ☐ Load YuNet ONNX model via `cv2.FaceDetectorYN.create()`
  - ☐ Configure detection parameters:
    - Confidence threshold: 0.7 (configurable)
    - NMS threshold: 0.3
    - Top-K: 5000
  - ☐ Implement `detect(frame)` → returns list of face bounding boxes + landmarks
  - ☐ Implement input frame resizing for performance (scale to 320×240 for detection)
  - ☐ Map detected coordinates back to original frame dimensions
  - ☐ Filter detections by minimum face size (60×60 pixels)
  - ☐ Handle multi-face detection (return all, let pipeline decide)

#### 5.2 Face Tracker (`recognition/tracker.py`)

- ☐ Implement `CSRTTracker` class:
  - ☐ Initialize CSRT tracker via `cv2.TrackerCSRT.create()`
  - ☐ Implement `init_track(frame, bbox)` — start tracking detected face
  - ☐ Implement `update(frame)` → returns (success, bbox)
  - ☐ Implement tracker reset logic:
    - Reset after N consecutive tracking failures (default: 5)
    - Reset after detection disagrees with tracker position (IoU < 0.3)
    - Reset every M frames to prevent drift (default: 30)
  - ☐ Implement `is_same_face(det_bbox, track_bbox)` — IoU comparison
  - ☐ Support tracking multiple faces simultaneously (dict of trackers)

#### 5.3 Face Aligner (`recognition/aligner.py`)

- ☐ Implement `FaceAligner` class:
  - ☐ Define standard 5-point landmark template (ArcFace alignment standard):
    ```python
    REFERENCE_LANDMARKS = np.array([
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # left mouth corner
        [70.7299, 92.2041],   # right mouth corner
    ], dtype=np.float32)
    ```
  - ☐ Implement `align(frame, landmarks)` → returns 112×112 aligned face crop
  - ☐ Compute similarity transform from detected landmarks to reference
  - ☐ Apply `cv2.warpAffine` with the transformation matrix
  - ☐ Handle edge cases: landmarks outside frame, extreme angles

#### 5.4 Face Embedder (`recognition/embedder.py`)

- ☐ Implement `ArcFaceEmbedder` class:
  - ☐ Load ArcFace model via `insightface.app.FaceAnalysis`:
    ```python
    self.app = insightface.app.FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    self.app.prepare(ctx_id=0, det_size=(640, 640))
    ```
  - ☐ **Alternative:** Direct ONNX inference for finer control:
    - Load `w600k_r50.onnx` via `onnxruntime.InferenceSession`
    - Preprocess: resize 112×112, normalize to [-1, 1], transpose to NCHW
    - Run inference → 512-D embedding vector
    - L2-normalize output
  - ☐ Implement `get_embedding(aligned_face)` → returns 512-D normalized float32 array
  - ☐ Implement batch embedding: `get_embeddings(faces_list)` for efficiency
  - ☐ Implement embedding quality score based on face confidence + alignment error

#### 5.5 Face Matcher (`recognition/matcher.py`)

- ☐ Implement `FaceMatcher` class:
  - ☐ Implement `cosine_similarity(emb1, emb2)` → float [-1, 1]
  - ☐ Implement `match_against_gallery(probe_embedding, gallery_embeddings)`:
    - Compute cosine similarity against all gallery embeddings
    - Return top-K matches with scores
    - Vectorized computation using `numpy` dot product
  - ☐ Implement threshold-based matching:
    - `THRESHOLD_STRICT = 0.55` (high security)
    - `THRESHOLD_BALANCED = 0.45` (default)
    - `THRESHOLD_LENIENT = 0.35` (convenience)
  - ☐ Implement multi-embedding matching:
    - Each student may have multiple enrolled embeddings
    - Match against all, use maximum similarity
  - ☐ Implement embedding cache:
    - Load course roster embeddings into memory on session open
    - Invalidate on student face update
    - Memory-efficient storage using `numpy` arrays

#### 5.6 Pipeline Orchestrator (`recognition/pipeline.py`)

- ☐ Implement `RecognitionPipeline` class:
  - ☐ Initialize all sub-modules (detector, tracker, aligner, embedder, matcher)
  - ☐ Implement main `process_frame(frame)` method:
    ```python
    def process_frame(self, frame):
        # 1. Detection (every N frames) or Tracking (in between)
        # 2. If face found: Align
        # 3. If aligned: Generate embedding
        # 4. If embedding: Match against gallery
        # 5. Return result with confidence
    ```
  - ☐ Implement detection/tracking interleave:
    - Run detection every `detection_interval` frames (default: 5)
    - Run CSRT tracking on intermediate frames
    - If tracker fails, fall back to detection
  - ☐ Implement multi-frame smoothing:
    - Maintain rolling buffer of last N recognition results (default: 5)
    - Final decision = majority vote with weighted confidence
    - Reject if < 3/5 frames agree on identity
  - ☐ Implement performance mode switching:
    ```python
    class RecognitionMode(Enum):
        FAST = "fast"           # Detect every 10 frames, 320px scale
        BALANCED = "balanced"   # Detect every 5 frames, 480px scale
        ACCURATE = "accurate"   # Detect every 2 frames, 640px scale
    ```
  - ☐ Implement frame counter and FPS tracking
  - ☐ Implement result caching (same face position within 5px → reuse result)

#### 5.7 Pipeline Configuration (`recognition/config.py`)

- ☐ Define all recognition parameters as a dataclass:
  ```python
  @dataclass
  class RecognitionConfig:
      detection_confidence: float = 0.7
      recognition_threshold: float = 0.45
      detection_interval: int = 5
      frame_scale: float = 0.75
      min_face_size: int = 60
      smoothing_window: int = 5
      smoothing_min_agreement: int = 3
      tracker_reset_interval: int = 30
      tracker_iou_threshold: float = 0.3
      mode: RecognitionMode = RecognitionMode.BALANCED
  ```

**Dependencies:** Phase 1 (structure), Phase 2 (database for gallery embeddings)

**Expected Output:**
- Complete face recognition pipeline from frame input to identity output
- YuNet detection at ~30 FPS on CPU
- CSRT tracking with automatic fallback
- ArcFace embeddings with cosine similarity matching
- Multi-frame smoothing for stable recognition
- Three performance modes (FAST / BALANCED / ACCURATE)
- Comprehensive unit tests for each sub-module

---

### PHASE 6: Anti-Spoofing System

**Objective:** Implement multi-layer liveness detection to prevent photo/video/mask attacks.

**Tasks:**

#### 6.1 Model-Based Anti-Spoofing (`anti_spoofing/model.py`)

- ☐ Implement `MiniFASNetAntiSpoof` class:
  - ☐ Load pre-converted ONNX models:
    - `2.7_80x80_MiniFASNetV2.onnx` (scale 2.7×)
    - `4_0_0_80x80_MiniFASNetV1SE.onnx` (scale 4.0×)
  - ☐ Implement multi-scale crop:
    - Given face bbox, crop at scale 2.7× and 4.0× the face size
    - Resize each crop to 80×80
  - ☐ Implement preprocessing:
    - BGR → RGB
    - Normalize: `(img - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]`
    - Transpose to NCHW format
  - ☐ Implement `predict(frame, face_bbox)` → `(is_real: bool, score: float)`
  - ☐ Ensemble scoring: average scores from both models
  - ☐ Configurable threshold (default: 0.5, range: 0.3–0.8)

#### 6.2 Blink Detection (`anti_spoofing/blink_detector.py`)

- ☐ Implement `BlinkDetector` class:
  - ☐ Compute Eye Aspect Ratio (EAR) from eye landmarks:
    ```python
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    ```
  - ☐ Track EAR over time (rolling buffer of 30 values)
  - ☐ Detect blink event: EAR drops below 0.21 for 2-4 consecutive frames
  - ☐ Count blinks in verification window
  - ☐ Anti-spoofing rule: require ≥ 1 natural blink within 5 seconds
  - ☐ Detect unnatural blink patterns (too regular = suspicious)

#### 6.3 Head Movement Detection (`anti_spoofing/movement_checker.py`)

- ☐ Implement `MovementChecker` class:
  - ☐ Track face position (bbox center) over frames
  - ☐ Compute frame-to-frame displacement vector
  - ☐ Detect micro-movements (natural head sway: 2-8px variation at 640px width)
  - ☐ Challenge-response protocol:
    - Request: "Turn head slightly left"
    - Verify: face bbox center moves > 20px horizontally
    - Timeout: 5 seconds per challenge
  - ☐ Detect static face (zero movement = photo/screen):
    - If std(position) < 1px over 30 frames → flag as suspicious

#### 6.4 Spoof Detection Orchestrator (`anti_spoofing/spoof_detector.py`)

- ☐ Implement `SpoofDetector` class:
  - ☐ Combine all anti-spoofing signals:
    ```python
    def check_liveness(self, frame, face_bbox, landmarks, frame_history):
        model_result = self.model.predict(frame, face_bbox)
        blink_result = self.blink_detector.check(landmarks)
        movement_result = self.movement_checker.check(face_bbox)
        
        # Weighted ensemble
        score = (
            0.5 * model_result.score +
            0.25 * blink_result.score +
            0.25 * movement_result.score
        )
        
        return LivenessResult(
            is_real=score > self.threshold,
            score=score,
            model_score=model_result.score,
            blink_detected=blink_result.blink_detected,
            movement_natural=movement_result.is_natural
        )
    ```
  - ☐ Implement confidence levels:
    - `HIGH`: model ≥ 0.7 AND blink detected AND natural movement
    - `MEDIUM`: model ≥ 0.5 AND (blink OR movement)
    - `LOW`: model ≥ 0.3 only
    - `REJECTED`: any signal below threshold
  - ☐ Implement adaptive thresholds based on environment:
    - Well-lit: strict thresholds
    - Low-light: slightly relaxed model threshold

**Dependencies:** Phase 5 (detector for face bbox, landmarks)

**Expected Output:**
- Multi-layer anti-spoofing system
- MiniFASNet model inference (CPU: <20ms per frame)
- Blink detection with natural pattern analysis
- Head movement verification
- Combined liveness score with configurable thresholds
- Unit tests with real/spoof image pairs

---

### PHASE 7: Verification System

**Objective:** Integrate recognition pipeline + anti-spoofing into a complete verification workflow.

**Tasks:**

- ☐ Implement `verification/session.py` — Verification session manager:
  - ☐ `VerificationSession` dataclass:
    ```python
    @dataclass
    class VerificationSession:
        session_id: str
        student_id: str
        course_id: str
        started_at: datetime
        status: str  # pending, verifying, success, failed, timeout
        frames_processed: int
        recognition_results: List[RecognitionResult]
        liveness_results: List[LivenessResult]
        final_identity: str | None
        final_confidence: float
        final_liveness_score: float
    ```
  - ☐ Session lifecycle management:
    - Create → Process Frames → Aggregate → Decide → Record
    - Timeout after 30 seconds of no successful match
    - Maximum 150 frames per session (30s at 5 FPS)
  - ☐ Session state persistence in Redis (TTL: 5 minutes)

- ☐ Implement `verification/verifier.py` — Main verification engine:
  - ☐ `Verifier.start_session(student_id, course_id)` → creates verification session
  - ☐ `Verifier.process_frame(session_id, frame)` → processes single frame:
    1. Run recognition pipeline
    2. Run anti-spoofing check
    3. Accumulate results in session
    4. Check if decision threshold met
    5. Return intermediate result (for real-time UI feedback)
  - ☐ `Verifier.finalize(session_id)` → make final decision:
    - Aggregate multi-frame recognition (majority vote)
    - Aggregate liveness scores (must pass in ≥ 60% of frames)
    - Cross-check recognized identity matches claimed student_id
    - If all checks pass → mark attendance in database
  - ☐ Implement gallery loading:
    - On session start: load target student's embeddings from DB
    - Also load course roster embeddings (to detect wrong student)
  - ☐ Implement verification result:
    ```python
    @dataclass
    class VerificationResult:
        status: str          # success | failed | spoofing_detected | no_face | wrong_person | timeout
        confidence: float
        liveness_score: float
        message: str
        attendance_recorded: bool
    ```
  - ☐ Emit SocketIO events on verification completion (for admin dashboard)

**Dependencies:** Phase 5 (recognition), Phase 6 (anti-spoofing), Phase 2 (database), Phase 3 (SocketIO)

**Expected Output:**
- End-to-end verification workflow
- Session management with state tracking
- Multi-frame aggregation for robust decisions
- Integrated anti-spoofing checks
- Automatic attendance recording on success
- Real-time status feedback to client

---

### PHASE 8: Performance Optimization

**Objective:** Optimize the system for low-latency, high-throughput operation.

**Tasks:**

- ☐ Implement frame scaling strategy:
  - ☐ Scale input frames to processing resolution before detection:
    - FAST mode: 320×240
    - BALANCED mode: 480×360
    - ACCURATE mode: 640×480
  - ☐ Use `cv2.resize` with `INTER_LINEAR` for downscaling
  - ☐ Map detection results back to original coordinates

- ☐ Implement detection interval optimization:
  - ☐ Adaptive detection interval based on tracking confidence:
    - Tracker confident (high CSRT score) → detect every 10 frames
    - Tracker uncertain → detect every 3 frames
    - Tracker lost → detect every frame until face found
  - ☐ Skip detection entirely if face bbox unchanged (< 3px movement)

- ☐ Implement embedding cache:
  - ☐ In-memory LRU cache for course roster embeddings
  - ☐ Cache key: `course_id:student_id:embedding_version`
  - ☐ Maximum cache size: 500 students (configurable)
  - ☐ Pre-warm cache when attendance session opens
  - ☐ Invalidate on student face data update

- ☐ Implement ONNX Runtime optimization:
  - ☐ Enable graph optimization: `sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL`
  - ☐ Set intra-op thread count based on CPU cores: `sess_options.intra_op_num_threads = 4`
  - ☐ Enable memory pattern optimization: `sess_options.enable_mem_pattern = True`
  - ☐ For GPU: configure CUDA execution provider with `device_id`, `arena_extend_strategy`

- ☐ Implement batch processing:
  - ☐ When multiple students verify simultaneously, batch embedding inference
  - ☐ ONNX batch inference: pad/collate multiple face crops into single tensor
  - ☐ Process anti-spoofing in parallel (thread pool)

- ☐ Implement model lazy loading:
  - ☐ Load detection model on first request (not app startup)
  - ☐ Load ArcFace model only when verification needed
  - ☐ Anti-spoofing models loaded on demand
  - ☐ Models remain in memory after first load (singleton pattern)

- ☐ Implement profiling and benchmarks:
  - ☐ Per-stage timing (detect, track, align, embed, match, antispoof)
  - ☐ Total pipeline latency per frame
  - ☐ Memory usage monitoring
  - ☐ FPS counter with rolling average

- ☐ Performance targets:
  | Stage | Target Latency (CPU) | Target Latency (GPU) |
  |---|---|---|
  | YuNet Detection | < 15ms | < 5ms |
  | CSRT Tracking | < 5ms | < 5ms |
  | Alignment | < 1ms | < 1ms |
  | ArcFace Embedding | < 30ms | < 8ms |
  | Cosine Matching (100 gallery) | < 1ms | < 1ms |
  | Anti-Spoofing (2 models) | < 25ms | < 10ms |
  | **Total Pipeline** | **< 80ms** | **< 30ms** |

**Dependencies:** Phase 5, Phase 6, Phase 7

**Expected Output:**
- Pipeline operating at < 80ms/frame on CPU
- Embedding cache reducing redundant computation
- Adaptive detection interval based on tracking state
- Performance benchmarking suite
- GPU acceleration path verified

---

### PHASE 9: UI/UX Implementation

**Objective:** Create polished, responsive user interfaces for both admin and student portals.

**Tasks:**

#### 9.1 Design System

- ☐ Define CSS custom properties (design tokens):
  - Color palette: primary (indigo-600), secondary (emerald-500), danger, warning, info
  - Dark mode palette variants
  - Typography scale: Inter font, 6 weight levels
  - Spacing scale: 4px base unit
  - Border radius: sm (4px), md (8px), lg (16px), xl (24px)
  - Shadow levels: sm, md, lg, xl
  - Transition curves: ease-out 200ms
- ☐ Create `base.css` — Reset + global styles + typography
- ☐ Create component CSS: buttons, inputs, cards, tables, modals, badges, alerts, dropdowns

#### 9.2 Admin Dashboard UI

- ☐ Dashboard layout: sidebar + main content area
- ☐ Real-time stats cards with animated counters (SocketIO-driven)
- ☐ Attendance chart (daily/weekly/monthly) using Chart.js
- ☐ Recent activity feed with live updates
- ☐ Quick action buttons (open session, export report, add student)
- ☐ Dark/light mode toggle with `prefers-color-scheme` detection

#### 9.3 Student Verification UI

- ☐ Full-screen camera view with canvas overlay
- ☐ Face bounding box drawn in real-time (green = detected, red = no face)
- ☐ Step-by-step progress indicator with animations:
  1. Camera initialization → spinner
  2. Face detection → bounding box appears
  3. Liveness check → "Blink naturally" prompt
  4. Identity verification → progress percentage
  5. Success → green checkmark animation
- ☐ Error states with clear recovery instructions
- ☐ Mobile-responsive camera layout (portrait mode support)

#### 9.4 Responsive Design

- ☐ Breakpoints: mobile (< 640px), tablet (640-1024px), desktop (> 1024px)
- ☐ Admin sidebar collapses to bottom nav on mobile
- ☐ Data tables convert to card layout on mobile
- ☐ Camera view adapts to device orientation

#### 9.5 Accessibility

- ☐ WCAG 2.1 AA compliance
- ☐ Keyboard navigation for all interactive elements
- ☐ ARIA labels on dynamic content
- ☐ Skip-to-content link
- ☐ Focus indicator styles

**Dependencies:** Phase 3 (admin routes), Phase 4 (student routes)

**Expected Output:**
- Modern, premium-feel UI for both portals
- Real-time dashboard with live data
- Intuitive webcam verification flow
- Fully responsive across devices
- Dark/light mode support
- Accessible design

---

### PHASE 10: DevOps & Deployment

**Objective:** Containerize all services and set up production deployment with Nginx.

**Tasks:**

#### 10.1 Docker Setup

- ☐ Create `docker/Dockerfile.admin`:
  ```dockerfile
  FROM python:3.11.9-slim
  
  RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
      && rm -rf /var/lib/apt/lists/*
  
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
      --index-url https://download.pytorch.org/whl/cpu
  
  COPY . .
  
  EXPOSE 5000
  CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", \
       "--bind", "0.0.0.0:5000", "run_admin:app"]
  ```
- ☐ Create `docker/Dockerfile.student` (similar, exposes port 5001)
- ☐ Create `docker/Dockerfile.celery`:
  ```dockerfile
  FROM python:3.11.9-slim
  # ... same base setup ...
  CMD ["celery", "-A", "tasks.celery_app", "worker", \
       "--loglevel=info", "--concurrency=2"]
  ```
- ☐ Create `docker/nginx/nginx.conf`:
  ```nginx
  upstream admin_app {
      server admin:5000;
  }
  upstream student_app {
      server student:5001;
  }

  server {
      listen 80;
      server_name autoattendance.local;

      client_max_body_size 10M;

      # Admin routes
      location /admin/ {
          proxy_pass http://admin_app;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }

      # Admin SocketIO
      location /socket.io/ {
          proxy_pass http://admin_app;
          proxy_http_version 1.1;
          proxy_set_header Upgrade $http_upgrade;
          proxy_set_header Connection "upgrade";
          proxy_set_header Host $host;
      }

      # Student routes
      location /student/ {
          proxy_pass http://student_app;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }

      # Student WebSocket (verification stream)
      location /student/ws/ {
          proxy_pass http://student_app;
          proxy_http_version 1.1;
          proxy_set_header Upgrade $http_upgrade;
          proxy_set_header Connection "upgrade";
          proxy_read_timeout 300s;
      }

      # Static files (admin)
      location /admin/static/ {
          alias /app/admin_app/static/;
          expires 30d;
          add_header Cache-Control "public, immutable";
      }

      # Static files (student)
      location /student/static/ {
          alias /app/student_app/static/;
          expires 30d;
          add_header Cache-Control "public, immutable";
      }
  }
  ```
- ☐ Create `docker/nginx/Dockerfile`:
  ```dockerfile
  FROM nginx:1.27-alpine
  COPY nginx.conf /etc/nginx/conf.d/default.conf
  ```

#### 10.2 Docker Compose

- ☐ Create `docker-compose.yml`:
  ```yaml
  version: '3.9'

  services:
    admin:
      build:
        context: .
        dockerfile: docker/Dockerfile.admin
      env_file: .env
      volumes:
        - model_data:/app/models
      depends_on:
        - redis
      networks:
        - backend

    student:
      build:
        context: .
        dockerfile: docker/Dockerfile.student
      env_file: .env
      volumes:
        - model_data:/app/models
      depends_on:
        - redis
      networks:
        - backend

    celery_worker:
      build:
        context: .
        dockerfile: docker/Dockerfile.celery
      env_file: .env
      volumes:
        - model_data:/app/models
      depends_on:
        - redis
      networks:
        - backend

    redis:
      image: redis:7.4-alpine
      command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
      volumes:
        - redis_data:/data
      networks:
        - backend

    nginx:
      build:
        context: docker/nginx
      ports:
        - "80:80"
        - "443:443"
      depends_on:
        - admin
        - student
      networks:
        - backend

  volumes:
    model_data:
    redis_data:

  networks:
    backend:
      driver: bridge
  ```

#### 10.3 Environment & Operations

- ☐ Create `.env.production` with production-safe defaults
- ☐ Implement health check endpoints:
  - `GET /admin/health` → `{"status": "ok", "db": "connected", "redis": "connected"}`
  - `GET /student/health` → same
- ☐ Configure container resource limits in docker-compose (memory, CPU)
- ☐ Create `scripts/download_models.py` — downloads all ML models to `models/` directory
- ☐ Create `Makefile` with common commands:
  ```makefile
  build:      docker-compose build
  up:         docker-compose up -d
  down:       docker-compose down
  logs:       docker-compose logs -f
  seed:       docker-compose exec admin python scripts/seed_db.py
  test:       docker-compose exec admin pytest
  shell:      docker-compose exec admin bash
  ```

**Dependencies:** Phase 1–9 complete

**Expected Output:**
- All services containerized and orchestrated via Docker Compose
- Nginx reverse proxy routing to correct service
- WebSocket proxying configured
- Static file serving with caching
- Health check endpoints
- Model volume shared across containers
- Production-ready `.env` template

---

### PHASE 11: Testing & Validation

**Objective:** Comprehensive test suite covering unit, integration, ML pipeline, and edge cases.

**Tasks:**

#### 11.1 Test Infrastructure

- ☐ Implement `tests/conftest.py`:
  - ☐ Flask test client fixtures for admin and student apps
  - ☐ MongoDB mock fixture using `mongomock`
  - ☐ Redis mock fixture
  - ☐ Sample face image fixtures (real + spoof)
  - ☐ Pre-computed embedding fixtures
  - ☐ Authentication helper fixtures (logged-in admin, logged-in student)

#### 11.2 Unit Tests

- ☐ `tests/test_auth.py`:
  - ☐ Test login with valid credentials
  - ☐ Test login with invalid password
  - ☐ Test rate limiting (6th attempt blocked)
  - ☐ Test RBAC decorator (viewer cannot access admin routes)
  - ☐ Test session expiration
- ☐ `tests/test_database.py`:
  - ☐ Test StudentDAO CRUD operations
  - ☐ Test CourseDAO enrollment/unenrollment
  - ☐ Test AttendanceDAO duplicate prevention
  - ☐ Test index creation
  - ☐ Test connection retry logic
- ☐ `tests/test_recognition.py`:
  - ☐ Test YuNet detection on known face images
  - ☐ Test detection returns correct landmark count (5 points)
  - ☐ Test alignment produces 112×112 output
  - ☐ Test embedding dimensionality (512-D)
  - ☐ Test embedding normalization (L2 norm ≈ 1.0)
  - ☐ Test cosine similarity of same person > 0.5
  - ☐ Test cosine similarity of different persons < 0.3
  - ☐ Test multi-frame smoothing majority vote logic
- ☐ `tests/test_anti_spoofing.py`:
  - ☐ Test MiniFASNet on real face image → is_real = True
  - ☐ Test MiniFASNet on screen photo → is_real = False
  - ☐ Test blink detection with EAR sequence
  - ☐ Test movement detection with static position sequence
  - ☐ Test ensemble scoring weights
- ☐ `tests/test_verification.py`:
  - ☐ Test session creation and state transitions
  - ☐ Test frame processing returns intermediate results
  - ☐ Test finalization with passing results → attendance recorded
  - ☐ Test finalization with failing liveness → attendance NOT recorded
  - ☐ Test session timeout behavior
- ☐ `tests/test_admin_routes.py`:
  - ☐ Test all CRUD endpoints return correct status codes
  - ☐ Test unauthorized access returns 403
  - ☐ Test student search and pagination
  - ☐ Test attendance session open/close lifecycle
- ☐ `tests/test_student_routes.py`:
  - ☐ Test login flow
  - ☐ Test attendance history API
  - ☐ Test profile update

#### 11.3 Integration Tests

- ☐ Test end-to-end verification flow:
  - Login → Open camera → Submit frames → Get result → Check attendance DB
- ☐ Test SocketIO dashboard updates:
  - Mark attendance → Assert dashboard received event
- ☐ Test Celery task execution:
  - Upload face photo → Assert embedding generated in DB
- ☐ Test concurrent verification sessions (2+ students simultaneously)

#### 11.4 ML Pipeline Validation

- ☐ Create validation dataset: 20 real faces, 20 spoof images (printed photos, screen displays)
- ☐ Benchmark pipeline accuracy:
  - True Accept Rate (TAR) at threshold 0.45
  - False Accept Rate (FAR) at threshold 0.45
  - Target: TAR > 95%, FAR < 1%
- ☐ Benchmark anti-spoofing accuracy:
  - Attack Presentation Classification Error Rate (APCER) < 5%
  - Bona Fide Presentation Classification Error Rate (BPCER) < 5%
- ☐ Benchmark latency: measure p50, p95, p99 for each pipeline stage

#### 11.5 Edge Case Tests

- ☐ No face in frame → graceful "no_face" result
- ☐ Multiple faces in frame → select largest face
- ☐ Extreme lighting (very dark, very bright) → detection still works or graceful failure
- ☐ Partial face (occluded by mask or hand) → reject with message
- ☐ Face at extreme angle (> 45° yaw) → reject but don't crash
- ☐ Very small face (< 60px) → skip detection
- ☐ Network interruption during verification → session timeout, no partial attendance
- ☐ Concurrent attendance marking for same student → only first succeeds
- ☐ Student not enrolled in course → "wrong_person" result

**Dependencies:** All previous phases

**Expected Output:**
- > 90% code coverage on core modules
- All unit tests passing
- Integration tests validating end-to-end flows
- ML pipeline accuracy benchmarks documented
- Edge cases handled gracefully
- CI-runnable test suite (`pytest --cov`)

---

## 5. DETAILED MODULE BREAKDOWN

### 5.1 `core/` — Shared Foundation

| File | Description | Responsibilities |
|---|---|---|
| `config.py` | Configuration manager | Load `.env`, provide config objects, validate settings |
| `database.py` | MongoDB connection | Connection pool, health check, index setup, retry logic |
| `models.py` | Data Access Objects | CRUD operations for all collections |
| `extensions.py` | Flask extension instances | Shared `login_manager`, `socketio`, `limiter` instances |
| `utils.py` | Utility functions | Password hashing, ID generation, date helpers, validators |

### 5.2 `admin_app/` — Admin Portal

| File | Description | Responsibilities |
|---|---|---|
| `__init__.py` | App factory | Create Flask app, register blueprints, configure middleware |
| `routes/auth.py` | Auth blueprint | Login, logout, change password, session management |
| `routes/dashboard.py` | Dashboard blueprint | Main dashboard, SocketIO handlers, real-time stats |
| `routes/students.py` | Student management | CRUD, face upload, bulk import |
| `routes/courses.py` | Course management | CRUD, enrollment, session management |
| `routes/attendance.py` | Attendance views | Overview, live session, manual marking |
| `routes/reports.py` | Report generation | Generate, download, scheduled reports |
| `forms.py` | WTForms definitions | Form validation and CSRF protection |

### 5.3 `student_app/` — Student Portal

| File | Description | Responsibilities |
|---|---|---|
| `__init__.py` | App factory | Create Flask app, configure webcam routes |
| `routes/auth.py` | Auth blueprint | Login, logout, self-registration |
| `routes/attendance.py` | Attendance blueprint | Webcam verification page, frame processing endpoint, history |
| `routes/profile.py` | Profile blueprint | View/update profile, re-register face |
| `static/js/webcam.js` | Camera handler | getUserMedia, frame capture, WebSocket client, overlay rendering |
| `static/js/verification.js` | Verification UI | Progress indicators, status messages, animations |

### 5.4 `recognition/` — Face Recognition Pipeline

| File | Description | Responsibilities |
|---|---|---|
| `detector.py` | Face detection | YuNet ONNX detection, bbox + landmark extraction |
| `tracker.py` | Face tracking | CSRT tracking between detections, reset logic |
| `aligner.py` | Face alignment | 5-point similarity transform, 112×112 crop |
| `embedder.py` | Face embedding | ArcFace ONNX inference, 512-D vector output |
| `matcher.py` | Embedding matching | Cosine similarity, gallery management, threshold |
| `pipeline.py` | Pipeline orchestrator | Frame routing, mode switching, multi-frame smoothing |
| `config.py` | Pipeline configuration | All tunable parameters as dataclass |

### 5.5 `anti_spoofing/` — Liveness Detection

| File | Description | Responsibilities |
|---|---|---|
| `model.py` | MiniFASNet inference | Multi-scale ONNX inference, ensemble scoring |
| `blink_detector.py` | Blink detection | EAR computation, blink event detection, pattern analysis |
| `movement_checker.py` | Movement analysis | Position tracking, micro-movement detection, challenge-response |
| `spoof_detector.py` | Orchestrator | Combine all signals, weighted ensemble, confidence levels |

### 5.6 `verification/` — Verification Workflow

| File | Description | Responsibilities |
|---|---|---|
| `verifier.py` | Verification engine | Session-based multi-frame verification, decision aggregation |
| `session.py` | Session manager | Session state, lifecycle, Redis persistence |

### 5.7 `tasks/` — Background Jobs

| File | Description | Responsibilities |
|---|---|---|
| `celery_app.py` | Celery configuration | Broker/backend setup, task discovery |
| `embedding_tasks.py` | Embedding generation | Process uploaded face photos, generate + store embeddings |
| `report_tasks.py` | Report generation | Async CSV/PDF report creation |
| `cleanup_tasks.py` | Maintenance | Purge old sessions, archive logs, recalculate stats |

---

## 6. ML PIPELINE PLAN (DETAILED)

### 6.1 Pipeline Data Flow

```mermaid
flowchart TD
    INPUT[Raw Frame - 640×480 BGR] --> SCALE[Resize to Processing Resolution]
    SCALE --> CHECK{Detection Frame?}
    
    CHECK -->|Every N frames| DETECT[YuNet Face Detection]
    CHECK -->|In between| TRACK[CSRT Tracker Update]
    
    DETECT --> FOUND{Face Detected?}
    TRACK --> TFOUND{Tracking OK?}
    
    TFOUND -->|Yes| ALIGN
    TFOUND -->|No| DETECT
    
    FOUND -->|Yes| INIT_TRACK[Initialize Tracker]
    FOUND -->|No| RESULT_NONE[Result: No Face]
    
    INIT_TRACK --> ALIGN[5-Point Landmark Alignment]
    
    ALIGN --> CROP[112×112 Aligned Face Crop]
    
    CROP --> EMBED_CHECK{Embedding Frame?}
    EMBED_CHECK -->|Every M frames| EMBED[ArcFace Embedding - 512D]
    EMBED_CHECK -->|In between| CACHE[Use Cached Embedding]
    
    EMBED --> MATCH[Cosine Similarity vs Gallery]
    CACHE --> MATCH
    
    MATCH --> SMOOTH[Multi-Frame Smoothing Buffer]
    
    SMOOTH --> DECISION{Consensus Reached?}
    DECISION -->|≥ 3/5 agree| RESULT_ID[Result: Identity + Confidence]
    DECISION -->|No consensus| RESULT_UNC[Result: Uncertain]
    
    CROP --> ANTISPOOF[Anti-Spoofing Check]
    ANTISPOOF --> LIVENESS[Liveness Result]
```

### 6.2 Execution Frequency Table

| Stage | FAST Mode | BALANCED Mode | ACCURATE Mode |
|---|---|---|---|
| Frame input | 5 FPS | 5 FPS | 5 FPS |
| Frame scale | 320×240 | 480×360 | 640×480 |
| YuNet Detection | Every 10th frame | Every 5th frame | Every 2nd frame |
| CSRT Tracking | Intermediate frames | Intermediate frames | Intermediate frames |
| Alignment | Every detected/tracked frame | Same | Same |
| ArcFace Embedding | Every 5th frame | Every 3rd frame | Every frame |
| Cosine Matching | When embedding computed | Same | Same |
| Anti-Spoofing | Every 10th frame | Every 5th frame | Every 3rd frame |

### 6.3 Performance Trade-offs

| Aspect | FAST | BALANCED | ACCURATE |
|---|---|---|---|
| CPU per frame | ~15ms | ~40ms | ~80ms |
| Recognition latency (time-to-match) | ~3s | ~2s | ~1.5s |
| False Rejection Rate | Higher (~5%) | Low (~2%) | Very Low (~1%) |
| False Accept Rate | ~1% | ~0.5% | ~0.3% |
| Anti-spoofing robustness | Basic (model only) | Good (model + blink) | Best (model + blink + movement) |
| Best use case | High-traffic lobby | Classroom (default) | Secure exam |

### 6.4 Model Specifications

| Model | File | Size | Input | Output | Framework |
|---|---|---|---|---|---|
| YuNet | `face_detection_yunet_2023mar.onnx` | 338 KB | Variable × Variable × 3 (BGR) | Nx15 (bbox + landmarks + score) | OpenCV DNN |
| ArcFace (R50) | `w600k_r50.onnx` (via `buffalo_l`) | 166 MB | 1×3×112×112 (RGB, [-1,1]) | 1×512 (float32) | ONNX Runtime |
| MiniFASNetV2 | `2.7_80x80_MiniFASNetV2.onnx` | ~1 MB | 1×3×80×80 (RGB, [-1,1]) | 1×3 (class scores) | ONNX Runtime |
| MiniFASNetV1SE | `4_0_0_80x80_MiniFASNetV1SE.onnx` | ~2 MB | 1×3×80×80 (RGB, [-1,1]) | 1×3 (class scores) | ONNX Runtime |

---

## 7. PERFORMANCE PLAN

### 7.1 Frame Scaling Strategy

```python
SCALE_FACTORS = {
    RecognitionMode.FAST: 0.5,       # 640→320
    RecognitionMode.BALANCED: 0.75,  # 640→480
    RecognitionMode.ACCURATE: 1.0,   # 640→640
}
```

- Apply scaling BEFORE detection to reduce DNN input size
- Map detected bounding boxes and landmarks back to original resolution for alignment
- Alignment always crops at original resolution for embedding quality

### 7.2 Caching Strategy

| Cache | Location | Capacity | TTL | Invalidation |
|---|---|---|---|---|
| Course roster embeddings | In-memory (numpy) | 500 students | Session duration | Face data update |
| Recognition results | Session dict | 5 frames | Per verification session | New frame overwrites oldest |
| ONNX sessions | Module-level singleton | 1 per model | App lifetime | Never (lazy loaded) |
| Student profile data | Redis | 1000 entries | 15 minutes | Profile update |
| Attendance status | Redis | Per session | 5 minutes | Attendance recorded |

### 7.3 Load Mode Profiles

```python
PERFORMANCE_PROFILES = {
    "FAST": {
        "detection_interval": 10,
        "frame_scale": 0.5,
        "embedding_interval": 5,
        "antispoof_interval": 10,
        "smoothing_window": 3,
        "tracker_reset": 50,
    },
    "BALANCED": {
        "detection_interval": 5,
        "frame_scale": 0.75,
        "embedding_interval": 3,
        "antispoof_interval": 5,
        "smoothing_window": 5,
        "tracker_reset": 30,
    },
    "ACCURATE": {
        "detection_interval": 2,
        "frame_scale": 1.0,
        "embedding_interval": 1,
        "antispoof_interval": 3,
        "smoothing_window": 7,
        "tracker_reset": 15,
    },
}
```

---

## 8. SECURITY PLAN

### 8.1 Authentication

| Component | Mechanism | Details |
|---|---|---|
| Admin Login | Email + Password | bcrypt hashing, session-based auth (Flask-Login) |
| Student Login | Student ID + Password | bcrypt hashing, session-based auth |
| Session Management | Server-side (Flask) | Secure cookie, HttpOnly, SameSite=Lax, 30-min inactivity timeout |
| Password Policy | Enforced | Min 8 chars, 1 uppercase, 1 digit, 1 special char |

### 8.2 Authorization (RBAC)

| Role | Permissions |
|---|---|
| `super_admin` | All operations, user management, system config |
| `admin` | Manage students, courses, attendance, reports |
| `viewer` | View-only dashboard and reports |
| `student` | Mark own attendance, view own records, update own profile |

### 8.3 Data Validation

- ☐ All form inputs validated server-side (Flask-WTF validators)
- ☐ File uploads validated: MIME type, file size (max 5 MB), image dimensions
- ☐ MongoDB injection prevention: use parameterized queries only (no string interpolation in queries)
- ☐ XSS prevention: Jinja2 auto-escaping enabled, CSP headers
- ☐ CSRF protection: Flask-WTF CSRF tokens on all forms
- ☐ Rate limiting: Flask-Limiter on login (5/15min), API endpoints (30/min)

### 8.4 Anti-Spoofing Safeguards

| Attack Vector | Defense |
|---|---|
| Printed photo | MiniFASNet model detects flat surface texture |
| Screen display | MiniFASNet detects moiré patterns and color artifacts |
| Video replay | Blink detection (videos often lack natural blinks at correct timing) |
| Static image held up | Movement checker detects zero micro-movement |
| 3D mask | Challenge-response head turn (masks have limited range) |
| Multiple attempts | Rate limit verification sessions (max 3 per 10 min per student) |
| Session replay | Unique session ID per attempt, Redis-backed state |

### 8.5 Network Security

- ☐ Nginx: Hide server version, set security headers
- ☐ HTTPS with TLS 1.3 (configure certs in Nginx)
- ☐ CORS: Restrict to known origins
- ☐ WebSocket: Validate auth token on connection
- ☐ Internal services: No external port exposure (Docker network isolation)

---

## 9. DEPLOYMENT PLAN

### 9.1 Service Architecture

```mermaid
graph LR
    subgraph Docker Network
        NG[Nginx :80/:443] --> AA[Admin App :5000]
        NG --> SA[Student App :5001]
        AA --> RD[Redis :6379]
        SA --> RD
        CW[Celery Worker] --> RD
        AA --> MDB[MongoDB Atlas - External]
        SA --> MDB
        CW --> MDB
    end

    INTERNET[Internet] --> NG
```

### 9.2 Services Summary

| Service | Image | Port | Replicas | Resources |
|---|---|---|---|---|
| nginx | Custom (nginx:1.27-alpine) | 80, 443 | 1 | 128 MB RAM |
| admin | Custom (python:3.11-slim) | 5000 (internal) | 1 | 1 GB RAM, 2 CPU |
| student | Custom (python:3.11-slim) | 5001 (internal) | 1-2 | 2 GB RAM, 2 CPU |
| celery_worker | Custom (python:3.11-slim) | N/A | 1-3 | 1 GB RAM, 2 CPU |
| redis | redis:7.4-alpine | 6379 (internal) | 1 | 256 MB RAM |
| MongoDB Atlas | Managed (M10+) | N/A | 3 (replica set) | Managed |

### 9.3 Environment Variables (Production)

```env
# Flask
FLASK_SECRET_KEY=<generated-256-bit-key>
FLASK_ENV=production
FLASK_DEBUG=0

# MongoDB Atlas
MONGODB_URI=mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/autoattendance?retryWrites=true&w=majority
MONGODB_DB_NAME=autoattendance

# Redis
REDIS_URL=redis://redis:6379/0

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# Recognition
RECOGNITION_MODE=BALANCED
FACE_DETECTION_CONFIDENCE=0.7
RECOGNITION_THRESHOLD=0.45
ANTI_SPOOFING_THRESHOLD=0.5
ANTI_SPOOFING_ENABLED=1

# Security
SESSION_COOKIE_SECURE=1
SESSION_COOKIE_HTTPONLY=1
SESSION_COOKIE_SAMESITE=Lax
PERMANENT_SESSION_LIFETIME=1800

# Gunicorn
GUNICORN_WORKERS=1
GUNICORN_WORKER_CLASS=eventlet
GUNICORN_TIMEOUT=120
```

### 9.4 Deployment Checklist

- ☐ Generate production secrets (`FLASK_SECRET_KEY`)
- ☐ Configure MongoDB Atlas IP whitelist for deployment server
- ☐ Pre-download ML models into Docker volume
- ☐ Build Docker images: `docker-compose build`
- ☐ Seed database: `docker-compose exec admin python scripts/seed_db.py`
- ☐ Run health checks: `curl http://localhost/admin/health`
- ☐ Configure TLS certificates in Nginx (Let's Encrypt or custom)
- ☐ Set up log aggregation (optional: ELK stack or CloudWatch)
- ☐ Configure backups for Redis (RDB snapshots)
- ☐ Create admin super_admin account
- ☐ Run smoke tests against deployed instance
- ☐ Configure monitoring (uptime, response time, error rate)

---

## 10. TESTING PLAN

### 10.1 Test Categories

| Category | Tool | Coverage Target | Run Frequency |
|---|---|---|---|
| Unit Tests | pytest + mongomock | > 90% on core, recognition, anti-spoofing | Every commit |
| Integration Tests | pytest + Flask test client | > 80% on routes | Every PR |
| ML Validation | Custom benchmark script | Accuracy metrics | Weekly / after model changes |
| Load Tests | locust (optional) | Response time under load | Pre-deployment |
| E2E Browser Tests | Manual / Playwright (optional) | Critical flows | Pre-release |

### 10.2 Test Execution Commands

```bash
# Run all unit tests
pytest tests/ -v --cov=core --cov=recognition --cov=anti_spoofing --cov=verification

# Run specific module tests
pytest tests/test_recognition.py -v

# Run with coverage report
pytest tests/ --cov-report=html --cov-report=term-missing

# Run only fast tests (exclude ML model tests)
pytest tests/ -v -m "not slow"

# Run ML benchmark
python scripts/benchmark_pipeline.py --dataset tests/fixtures/validation_faces/
```

### 10.3 CI/CD Pipeline (Recommended)

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      - run: pytest tests/ -v --cov --cov-report=xml -m "not slow"
      - uses: codecov/codecov-action@v4
```

---

## 11. PROGRESS TRACKING

### Phase Completion Checklist

| Phase | Status | Description |
|---|---|---|
| PHASE 1 | ☐ | Project Setup |
| PHASE 2 | ☐ | Database Design |
| PHASE 3 | ☐ | Admin App Core |
| PHASE 4 | ☐ | Student Portal |
| PHASE 5 | ☐ | Face Recognition Pipeline |
| PHASE 6 | ☐ | Anti-Spoofing System |
| PHASE 7 | ☐ | Verification System |
| PHASE 8 | ☐ | Performance Optimization |
| PHASE 9 | ☐ | UI/UX Implementation |
| PHASE 10 | ☐ | DevOps & Deployment |
| PHASE 11 | ☐ | Testing & Validation |

### Milestone Targets

| Milestone | Phases | Target | Description |
|---|---|---|---|
| **M1: Foundation** | 1–2 | Week 1–2 | Project scaffold + database running |
| **M2: Core Apps** | 3–4 | Week 3–5 | Admin + Student portals functional (no ML) |
| **M3: ML Pipeline** | 5–6 | Week 6–8 | Face recognition + anti-spoofing working |
| **M4: Integration** | 7–8 | Week 9–10 | End-to-end verification, performance tuned |
| **M5: Polish** | 9 | Week 11 | UI polished, responsive, accessible |
| **M6: Ship** | 10–11 | Week 12–13 | Deployed, tested, production-ready |

### Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| InsightFace model download fails in Docker | High | Medium | Pre-download models, mount as volume |
| ONNX Runtime + PyTorch conflict | High | Low | Separate venvs or pin compatible versions |
| MongoDB Atlas latency | Medium | Low | Connection pooling, local cache, read preference: nearest |
| Anti-spoofing false positives | Medium | Medium | Tunable thresholds, admin override |
| WebSocket disconnects during verification | High | Medium | Auto-reconnect, session persistence in Redis |
| Large embedding gallery slows matching | Medium | Low | Numpy vectorized ops, pre-filter by course roster |

---

> [!TIP]
> **Recommended Development Order:** Follow phases sequentially (1→11). However, Phase 4 (Student Portal UI) and Phase 5 (Recognition Pipeline) can be developed in parallel by different team members, using mock data at the integration boundary.

> [!IMPORTANT]
> **Critical Path:** Phases 5→6→7 form the ML critical path. These should receive the most development and testing attention, as they directly determine system accuracy and security.
