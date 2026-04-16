# Architecture & System Design

## Table of Contents

1. [System Overview](#system-overview)
2. [Layered Architecture](#layered-architecture)
3. [Module Organization](#module-organization)
4. [Design Patterns](#design-patterns)
5. [Data Models & Relationships](#data-models--relationships)
6. [Processing Pipelines](#processing-pipelines)
7. [Error Handling & Resilience](#error-handling--resilience)
8. [Performance Considerations](#performance-considerations)

---

## System Overview

**AutoAttendance** is a dual-application system designed for real-time face recognition-based attendance marking in educational settings. The architecture emphasizes:

- **Modularity**: Separation of concerns into vision, database, and web layers.
- **Reusability**: Shared core modules (config, database, authentication) across admin and student applications.
- **Resilience**: Circuit breaker pattern for database failures; graceful degradation for optional components (anti-spoofing models).
- **Performance**: Caching, motion-gated detection, track reuse, and optional GPU acceleration.

### High-Level Layers

```
┌─────────────────────────────┐
│   CLIENT LAYER              │ (Web browsers, JavaScript)
├─────────────────────────────┤
│   WEB & ROUTES LAYER        │ (Flask blueprints, REST APIs, SocketIO)
├─────────────────────────────┤
│   VISION & ML LAYER         │ (YuNet, ArcFace, tracking, anti-spoofing)
├─────────────────────────────┤
│   CORE SERVICES LAYER       │ (Database, config, auth, logging)
├─────────────────────────────┤
│   PERSISTENCE LAYER         │ (MongoDB collections)
└─────────────────────────────┘
```

---

## Layered Architecture

### Layer 1: Client Layer (Web Browsers)

**Responsibility**: Render UI and capture user input.

**Components**:
- **Admin Dashboard** (port 5000):
  - Real-time camera feed (SocketIO MJPEG stream).
  - Attendance session controls.
  - Student management and reporting.
  - Analytics and heatmaps.
  
- **Student Portal** (port 5001):
  - Registration and self-enrollment.
  - Face capture interface (webcam).
  - Attendance history viewer.

**Technology**: HTML5, CSS3, JavaScript (Fetch API, SocketIO client library).

---

### Layer 2: Web & Routes Layer

**Responsibility**: HTTP request handling, authentication, and business logic orchestration.

**Architecture**: Flask blueprints enable modular route organization.

**Key Modules** (in [web/](../web/)):

```
web/
├─ routes.py                 # Blueprint coordinator
├─ decorators.py             # RBAC & login enforcement
├─ attendance_routes.py      # Attendance session APIs
├─ camera_routes.py          # Camera & SocketIO endpoints
├─ registration_routes.py    # Student enrollment APIs
├─ student_routes.py         # Admin student management APIs
├─ report_routes.py          # CSV export & analytics
├─ auth_routes.py            # Login/logout
├─ health_routes.py          # Health checks
└─ routes_helpers.py         # Validation utilities
```

**Design Pattern**: Blueprint Registration

```python
# In admin_app/app.py (application factory)
app = Flask(__name__)
bp = Blueprint('main', __name__)

from web.routes import register_all_routes
register_all_routes(bp)         # Registers all sub-blueprints

app.register_blueprint(bp)
```

**Authentication**: 
- Admin users: Flask session-based (cookies, HttpOnly flag).
- Student users: Flask-Login with user_loader callback.
- Optional RBAC: Decorators check roles if `ENABLE_RBAC=1` (currently no-op).

---

### Layer 3: Vision & ML Layer

**Responsibility**: Real-time computer vision processing (detection, tracking, recognition, liveness).

**Key Modules** (in [vision/](../vision/)):

```
vision/
├─ pipeline.py              # YuNet detection + CSRT tracking + association
├─ recognition.py           # Alignment, quality gating, encoding
├─ face_engine.py           # ArcFace embedding backend (with GPU support)
├─ anti_spoofing.py         # Silent-Face liveness detection
├─ preprocessing.py         # CLAHE brightness normalization
└─ overlay.py               # Visualization helpers
```

**anti_spoofing/** (Anti-spoofing wrapper):

```
anti_spoofing/
├─ model.py                 # Model initialization
├─ spoof_detector.py        # Spoof detection logic
├─ blink_detector.py        # Eye Aspect Ratio (EAR) tracking
└─ movement_checker.py      # Motion heuristics
```

**Camera Module** (in [camera/](../camera/)):

```
camera/
└─ camera.py                # Threaded loop: capture → detect → track → recognize → mark
```

---

### Layer 4: Core Services Layer

**Responsibility**: Shared utilities, configuration, database access, and cross-cutting concerns.

**Key Modules** (in [core/](../core/)):

```
core/
├─ config.py                # Configuration & environment variables (80+ params)
├─ database.py              # MongoDB connection & CRUD operations
├─ models.py                # Data Access Objects (DAOs)
├─ auth.py                  # Password hashing & validation
├─ extensions.py            # Shared Flask extensions (SQLAlchemy, CORS, etc.)
├─ utils.py                 # Logging, validation, file helpers
├─ notifications.py         # Absence alerts (optional)
├─ performance.py           # Metrics tracking
└─ profiling.py             # Latency profiling
```

**Key Pattern: Data Access Objects (DAOs)**

DAOs provide a thin abstraction over the database layer:

```python
# In core/models.py
class StudentDAO:
    def __init__(self, db_module):
        self._db = db_module
    
    def get_by_reg_no(self, reg_no):
        """Fetch student by registration number."""
        return self._db.get_student_by_reg_no(reg_no)
    
    def save_encodings(self, reg_no, encodings):
        """Persist face encodings."""
        return self._db.save_student_encodings(reg_no, encodings)
```

**Circuit Breaker Pattern**

MongoDB failures are handled gracefully:

```python
# In core/database.py
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.opened_at = None
    
    def call(self, func, *args, **kwargs):
        if self.is_open():
            raise CircuitBreakerException("Database unavailable")
        
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise
```

---

### Layer 5: Persistence Layer

**Responsibility**: Data storage and retrieval.

**Database**: MongoDB Atlas (or local MongoDB instance).

**Collections**:

1. **students**: Enrollment records with ArcFace 512-D embeddings.
2. **attendance**: Daily attendance marks (unique per student per date).
3. **attendance_sessions**: Active sessions per camera (enforces single active session).
4. **users**: Admin and teacher accounts.
5. **notification_events**: Absence alerts (optional).

See [DATABASE.md](DATABASE.md) for full schema and queries.

---

## Module Organization

### Shared Core Modules

**core/config.py** (Configuration Source of Truth)

- Loads all environment variables and defines defaults.
- ~80 tunable parameters for ML thresholds, paths, timing, and performance.
- Single import point: `from core.config import RECOGNITION_THRESHOLD`.

**core/database.py** (MongoDB Interface)

- Connection management with pooling and circuit breaker.
- CRUD operations for all collections.
- Index creation and verification.

**core/models.py** (Data Access Objects)

- `StudentDAO`: Student registration and encoding management.
- `AttendanceDAO`: Attendance record creation and queries.
- `AttendanceSessionDAO`: Session lifecycle (create, end, auto-close).

**core/auth.py** (Authentication Utilities)

- `hash_password()`: Bcrypt hashing for secure storage.
- `check_password()`: Verification for login.

### Application-Specific Modules

**admin_app/app.py** (Admin Application Factory)

- Initializes Flask app with SocketIO.
- Loads YuNet, ArcFace, and Silent-Face models at startup.
- Registers blueprints from [web/routes.py](../web/routes.py).
- Runs on port 5000.

**student_app/app.py** (Student Application Factory)

- Lightweight Flask app (no SocketIO).
- Flask-Login initialization.
- Registers student-specific routes.
- Runs on port 5001.

**student_app/auth.py** (Student Portal Authentication)

- `StudentUser` class (inherits Flask-Login UserMixin).
- `authenticate_student()`: Credentials validation (reg_no or email + password).
- `init_auth()`: Flask-Login configuration.

**student_app/verification.py** (Onboarding Pipeline)

- `evaluate_student_samples()`: Orchestrates enrollment verification.
- Scoring: liveness (40%) + consistency (25%) + quality (20%) + duplicate check (15%).
- Auto-approval if score ≥ 85 and no duplicates detected.

---

## Design Patterns

### 1. Application Factory Pattern

Both admin and student apps use Flask app factories:

```python
# admin_app/app.py
def create_app(config_name='development'):
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Initialize extensions
    db.init_app(app)
    socketio.init_app(app)
    login_manager.init_app(app)
    
    # Register blueprints
    from web.routes import register_all_routes
    bp = Blueprint('main', __name__)
    register_all_routes(bp)
    app.register_blueprint(bp)
    
    return app
```

**Benefit**: Testability and configuration flexibility.

### 2. Blueprint Registration Pattern

All routes are organized as blueprints and registered centrally:

```python
# web/routes.py
def register_all_routes(bp):
    from web.attendance_routes import register_attendance_routes
    from web.camera_routes import register_camera_routes
    # ... other route registrations
    
    register_attendance_routes(bp)
    register_camera_routes(bp)
    # ...
```

**Benefit**: Modular route organization without circular imports.

### 3. Data Access Object (DAO) Pattern

DAOs provide a clean interface to database operations:

```python
# In web/attendance_routes.py
def _build_daos():
    return {
        'attendance': AttendanceDAO(database),
        'session': AttendanceSessionDAO(database),
        'student': StudentDAO(database)
    }

@bp.route('/api/attendance/sessions', methods=['POST'])
def start_session():
    daos = _build_daos()
    session_id = daos['session'].create_session(camera_id, course_id)
    return {'session_id': session_id}
```

**Benefit**: Loose coupling between routes and database; easy unit testing.

### 4. Circuit Breaker Pattern

Database failures don't crash the application:

```python
# In core/database.py
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def get_client():
    try:
        return breaker.call(_create_connection)
    except CircuitBreakerException:
        logger.error("Database circuit open; using cached data or returning error")
        raise
```

**Benefit**: Graceful degradation; prevents cascading failures.

### 5. Caching Patterns

Multiple caching strategies optimize performance:

```
┌─ Recognition Cache (track-level, TTL 2s)
├─ Encoding Cache (per-student, RAM-resident)
├─ Track Identity Cache (per-track, expires after 30 frames)
└─ Frame Buffer (MJPEG deque)
```

### 6. Configuration Externalization

All environment-specific settings are in `.env`:

```bash
# .env
MONGO_URI=mongodb+srv://user:pass@...
RECOGNITION_THRESHOLD=0.38
ENABLE_GPU_PROVIDERS=1
```

**Benefit**: Same Docker image works across dev/test/prod with different `.env` files.

---

## Data Models & Relationships

### Students Collection

```json
{
  "_id": ObjectId,
  "reg_no": "CS21001",           // Unique registration number
  "name": "Alice Johnson",
  "email": "alice@university.edu",
  "phone": "+1234567890",
  "semester": "6",
  "section": "A",
  "department": "Computer Science",
  "status": "approved",           // or "pending", "rejected"
  "encodings": [                  // ArcFace 512-D embeddings
    [0.123, 0.456, ..., 0.789]
  ],
  "legacy_dlib_encoding": null,   // For backward compatibility
  "enrollment_date": ISODate("2024-01-15"),
  "verified_at": ISODate("2024-01-15"),
  "verification_samples": 4
}
```

**Indexes**:
- `unique: reg_no` (primary key)
- `email` (for student login)
- `status` (for approval queries)

### Attendance Collection

```json
{
  "_id": ObjectId,
  "student_id": ObjectId,        // References students._id
  "reg_no": "CS21001",           // Denormalized for reporting
  "date": ISODate("2024-09-15"),
  "status": "Present",            // or "Absent", "Late"
  "marked_at": ISODate("2024-09-15T09:15:30Z"),
  "confidence": 0.92,             // Cosine similarity score
  "session_id": ObjectId,         // References attendance_sessions._id
  "camera_id": "lab-1",
  "course_id": "CS101",
  "verified": true,               // Admin-approved if disputed
  "notes": ""
}
```

**Indexes**:
- `unique: {student_id, date}` (one mark per day per student)
- `{date, status}` (for daily reports)
- `{student_id, date}` (for per-student history)

### AttendanceSessions Collection

```json
{
  "_id": ObjectId,
  "camera_id": "lab-1",           // Unique per camera (enforced at app level)
  "course_id": "CS101",
  "session_start": ISODate("2024-09-15T09:00:00Z"),
  "session_end": ISODate("2024-09-15T12:00:00Z"),
  "status": "active",             // or "ended", "auto_closed"
  "created_by": ObjectId,         // References users._id
  "attendance_count": 45,         // Students marked present in this session
  "auto_closed_at": null,         // Populated if idle-closed
  "last_activity": ISODate("2024-09-15T11:50:00Z")
}
```

**Indexes**:
- `unique: camera_id (active=true)` (only one active session per camera)
- `{status, session_start}` (for session history)

---

## Processing Pipelines

### Camera Real-Time Loop

The camera module runs a continuous loop that processes frames:

```python
# Simplified pseudocode from camera/camera.py
while True:
    ret, frame = cap.read()
    
    # Step 1: Motion detection (every 6 frames)
    if frame_count % DETECTION_INTERVAL == 0:
        motion_detected, gray = detect_motion(frame)
        if not motion_detected:
            continue
    
    # Step 2: Run YuNet detection
    detections = detect_faces_yunet(frame)
    
    # Step 3: Associate detections with existing tracks
    new_tracks, matched_indices = associate_detections(detections, self._tracks)
    
    # Step 4: For each matched or new track:
    for track in self._tracks:
        # Step 4a: Update tracker
        track.update(frame)
        
        # Step 4b: Extract face chip
        x, y, w, h = track.bbox
        face_chip = frame[y:y+h, x:x+w]
        
        # Step 4c: Quality gate
        if not check_quality_gate(face_chip):
            continue
        
        # Step 4d: Align and encode
        encoding = encode_face(face_chip)
        
        # Step 4e: Recognize (cosine similarity)
        matches = match_against_database(encoding, threshold=0.38)
        
        # Step 4f: Liveness check
        liveness_label, confidence = check_liveness(frame, track.landmarks)
        
        # Step 4g: Multi-frame voting
        if should_confirm(track, matches, liveness_label):
            mark_attendance(matches[0], track.id)
```

See [PIPELINE.md](PIPELINE.md) for detailed step-by-step breakdown.

---

## Error Handling & Resilience

### Graceful Degradation Levels

1. **Level 1 (Anti-Spoofing Optional)**: If Silent-Face models fail to load, all faces are marked "real" with confidence 1.0. Processing continues without liveness verification.

2. **Level 2 (Camera Failure)**: If camera disconnects, admin receives notification; attendance session remains open for manual resumption.

3. **Level 3 (Database Failure)**: Circuit breaker opens after 5 consecutive failures. New requests receive error responses. Cached encodings remain available in RAM.

### Timeout & Retry Strategies

```python
# In core/database.py
max_retries = 3
initial_delay = 0.5  # seconds
backoff_factor = 2.0

for attempt in range(max_retries):
    try:
        return perform_operation()
    except ConnectionError:
        if attempt == max_retries - 1:
            raise
        delay = initial_delay * (backoff_factor ** attempt)
        time.sleep(delay)
```

---

## Performance Considerations

### Caching Strategy

```
Recognition Cache (track level):
├─ Store: {encoding → (match_result, confidence, timestamp)}
├─ TTL: 2 seconds (RECOGNITION_TRACK_CACHE_TTL_SECONDS)
└─ Purpose: Avoid re-encoding same face in successive frames

Encoding Cache (per-student):
├─ Load: All student encodings into RAM at startup
├─ Store: {reg_no → [embedding1, embedding2, ...]}
└─ Purpose: O(1) lookup for cosine similarity matching

Track Cache:
├─ Store: {track_id → {result, expires_at}}
├─ TTL: Expires after 30 frames without match (TRACK_EXPIRATION_FRAMES)
└─ Purpose: Prevent re-recognition of same track
```

### Computational Bottlenecks & Mitigations

| Bottleneck | Mitigation | Impact |
|---|---|---|
| YuNet detection | Motion gating; run every N frames | ~80% latency reduction |
| ArcFace encoding | GPU acceleration; encoding cache | ~3–5× speedup on GPU |
| Cosine similarity | In-RAM encoding cache | O(n) → O(log n) effective |
| Silent-Face liveness | Optional; graceful degradation | Can disable for speed |
| CSRT tracking | KCF fallback (if `PERF_USE_KCF_TRACKER=1`) | Faster but less accurate |
| Frame resize | Configurable `FRAME_PROCESS_WIDTH` | Adjustable speed-accuracy tradeoff |

### Memory Management

- **Encoding cache**: ~0.5 MB per 1000 students (512-D float32).
- **Frame buffer**: Configurable; typically 5 MJPEG frames in deque.
- **Track objects**: One per active face; max ~5 by default (`PERF_MAX_FACES`).

---

## Security Considerations

### Authentication & Authorization

- Admin users: Session-based (HttpOnly cookies).
- Student users: Flask-Login with secure session tokens.
- RBAC: Optional decorators (`ENABLE_RBAC` flag); currently no-op for compatibility.

### Data Protection

- **Passwords**: Bcrypt hashing (cost factor 12).
- **Encodings**: Stored as binary in MongoDB; not reversible to original image.
- **Logs**: Sensitive data (passwords, PII) excluded from logs.

### Fraud Prevention

- **Anti-spoofing**: CNN-based liveness detection.
- **Duplicate enrollment**: Cosine similarity check against all enrolled students.
- **Audit trail**: Complete attendance logs with confidence scores.

---

## Summary

AutoAttendance's architecture balances:

- **Modularity**: Clear separation of concerns across layers.
- **Performance**: Caching, GPU acceleration, and motion gating.
- **Resilience**: Circuit breaker, graceful degradation, error handling.
- **Security**: Anti-spoofing, duplicate prevention, secure authentication.

Next Steps:
- See [THEORY.md](THEORY.md) for mathematical foundations of CV algorithms.
- See [PIPELINE.md](PIPELINE.md) for detailed frame-by-frame processing.
- See [DATABASE.md](DATABASE.md) for MongoDB schema and queries.

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
