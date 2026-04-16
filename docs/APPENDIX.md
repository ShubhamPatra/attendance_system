# Appendix: Configuration Reference & Utilities

## Table of Contents

1. [Configuration Parameters](#configuration-parameters)
2. [Glossary](#glossary)
3. [Useful Commands](#useful-commands)
4. [Troubleshooting](#troubleshooting)
5. [API Reference](#api-reference)
6. [Database Queries](#database-queries)

---

## Configuration Parameters

### Complete Reference ([core/config.py](../core/config.py))

All parameters configurable via environment variables. Format: `ATTENDANCE_<PARAM_NAME>`.

#### Detection & Tracking

| Parameter | Default | Type | Description |
|---|---|---|---|
| `DETECTION_INTERVAL` | 6 | int | Run face detection every N frames (0–60) |
| `MIN_FACE_WIDTH_PIXELS` | 36 | int | Minimum face width to process |
| `MIN_FACE_HEIGHT_PIXELS` | 36 | int | Minimum face height to process |
| `FRAME_PROCESS_WIDTH` | 512 | int | Frame resize width (lower = faster) |
| `TRACKER_TYPE` | "CSRT" | str | Tracker algorithm: CSRT, MIL, KCF |
| `MAX_TRACK_AGE` | 30 | int | Max frames to keep unmatched track |
| `DETECTION_CONFIDENCE` | 0.5 | float | YuNet confidence threshold (0–1) |

**Environment Variable Example**:
```bash
export ATTENDANCE_DETECTION_INTERVAL=4
export ATTENDANCE_FRAME_PROCESS_WIDTH=640
```

#### Recognition & Matching

| Parameter | Default | Type | Description |
|---|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.38 | float | Cosine similarity threshold for match (0–1) |
| `RECOGNITION_CONFIRM_FRAMES` | 2 | int | Frames needed to confirm identity (1–10) |
| `RECOGNITION_HISTORY_SIZE` | 5 | int | Rolling buffer size for confirmation voting |
| `DUPLICATE_DETECTION_THRESHOLD` | 0.38 | float | Threshold for duplicate enrollment detection |
| `USE_ARCFACE` | True | bool | Use ArcFace (True) or dlib legacy (False) |
| `EMBEDDING_BACKEND` | "arcface" | str | Backend: arcface, dlib |
| `ONNXRT_PROVIDER_PRIORITY` | "CUDAExecutionProvider, CPUExecutionProvider" | str | ONNX Runtime providers in priority order |

**Example Configuration**:
```python
# core/config.py (or environment)
RECOGNITION_THRESHOLD = 0.38  # Tighter matching
RECOGNITION_CONFIRM_FRAMES = 3  # Higher confidence requirement
```

#### Liveness & Anti-Spoofing

| Parameter | Default | Type | Description |
|---|---|---|---|
| `LIVENESS_THRESHOLD` | 0.55 | float | Confidence needed for "real" classification (0–1) |
| `LIVENESS_HISTORY_SIZE` | 5 | int | Rolling buffer for liveness voting |
| `LIVENESS_CONFIRM_THRESHOLD` | 3 | int | Votes needed to confirm real face |
| `CHECK_EAR_BLINK` | True | bool | Enable eye aspect ratio (blink) detection |
| `EAR_THRESHOLD` | 0.2 | float | Eye Aspect Ratio for blink detection |
| `MOTION_MAGNITUDE_THRESHOLD` | 2.0 | float | Optical flow magnitude for motion detection |

**Tuning Strategy**:
- **Stricter**: `LIVENESS_THRESHOLD=0.65`, `LIVENESS_CONFIRM_THRESHOLD=4`
- **Lenient**: `LIVENESS_THRESHOLD=0.45`, `LIVENESS_CONFIRM_THRESHOLD=2`

#### Quality Gating

| Parameter | Default | Type | Description |
|---|---|---|---|
| `BLUR_THRESHOLD` | 6.0 | float | Laplacian variance (higher = sharper) |
| `BRIGHTNESS_THRESHOLD` | 40 | int | Minimum brightness (0–255) |
| `MIN_FACE_SIZE_PIXELS` | 36 | int | Minimum detectable face size |

#### Attendance & Session

| Parameter | Default | Type | Description |
|---|---|---|---|
| `SESSION_IDLE_TIMEOUT_MINUTES` | 120 | int | Auto-close session after N minutes idle |
| `SAME_DAY_DUPLICATE_PREVENTION` | True | bool | Prevent multiple marks per student per day |
| `STUDENT_RECOGNITION_COOLDOWN_SECONDS` | 3 | int | Prevent re-recognition of same student within N sec |
| `MAX_STUDENTS_PER_SESSION` | 500 | int | Hard limit on simultaneous tracked students |

#### Performance & Optimization

| Parameter | Default | Type | Description |
|---|---|---|---|
| `ENABLE_GPU` | False | bool | Use GPU (CUDA/TensorRT) if available |
| `CACHE_EMBEDDINGS` | True | bool | Cache embeddings in memory (TTL 2s) |
| `EMBEDDING_CACHE_TTL_SECONDS` | 2 | int | Embedding cache time-to-live |
| `BATCH_RECOGNITION` | False | bool | Batch embeddings for throughput (experimental) |
| `NUM_THREADS` | 4 | int | OpenCV threading threads |

**GPU Enabling**:
```bash
export ATTENDANCE_ENABLE_GPU=1
# Requires: onnxruntime[gpu], torch with CUDA
```

#### Database

| Parameter | Default | Type | Description |
|---|---|---|---|
| `MONGODB_URI` | "mongodb://localhost" | str | MongoDB connection string |
| `MONGODB_DATABASE` | "attendance_system" | str | Database name |
| `MONGODB_POOL_SIZE` | 50 | int | Connection pool size (MongoDB) |
| `MONGODB_TIMEOUT_SECONDS` | 30 | int | Operation timeout |
| `CIRCUIT_BREAKER_ENABLED` | True | bool | Enable circuit breaker pattern |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 5 | int | Failures before circuit opens |

#### Logging & Monitoring

| Parameter | Default | Type | Description |
|---|---|---|---|
| `LOG_LEVEL` | "INFO" | str | Logging level: DEBUG, INFO, WARNING, ERROR |
| `LOG_FILE_PATH` | "logs/attendance.log" | str | Log file location |
| `METRICS_ENABLED` | True | bool | Enable performance metrics collection |
| `PROFILING_ENABLED` | False | bool | Enable cProfile profiling (overhead) |

#### Feature Flags

| Parameter | Default | Type | Description |
|---|---|---|---|
| `ENABLE_RBAC` | False | bool | Enable role-based access control (future) |
| `ENABLE_NOTIFICATIONS` | True | bool | Enable email/SMS notifications |
| `ENABLE_ANALYTICS` | True | bool | Enable session analytics & heatmaps |
| `ENABLE_HEALTH_CHECKS` | True | bool | Enable `/health` diagnostic endpoint |

---

## Glossary

### Biometric & Recognition Terms

| Term | Acronym | Definition |
|---|---|---|
| **True Accept Rate** | TAR | % of genuine faces correctly accepted |
| **False Accept Rate** | FAR | % of imposter faces incorrectly accepted |
| **False Reject Rate** | FRR | % of genuine faces incorrectly rejected |
| **Equal Error Rate** | EER | Operating point where TAR = 1 - FRR |
| **Receiver Operating Characteristic** | ROC | TAR vs. FAR curve (higher = better) |
| **Area Under Curve** | AUC | Integral under ROC (max 1.0, min 0.0) |
| **Attack Presentation Classification Error Rate** | APCER | % of spoofing attacks accepted (ISO/IEC 30107-3) |
| **Bona Fide Presentation Classification Error Rate** | BPCER | % of genuine faces rejected (ISO/IEC 30107-3) |
| **Genuine Acceptance Rate** | GAR | % of legitimate presentations accepted |

### Computer Vision Terms

| Term | Definition |
|---|---|
| **Embedding** | Fixed-size numerical representation (e.g., 512-D ArcFace) |
| **Cosine Similarity** | Distance metric: $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ |
| **L2 Normalization** | Ensure embedding magnitude = 1.0 |
| **Bounding Box (bbox)** | Rectangle coordinates (x, y, width, height) for detected face |
| **Landmark** | Facial keypoints (68 for dlib, 5 for ArcFace alignment) |
| **IoU (Intersection over Union)** | Overlap metric: $\frac{\text{Area}_{\text{intersection}}}{\text{Area}_{\text{union}}}$ |
| **Optical Flow** | Motion estimation between frames (direction & magnitude) |
| **Affine Warp** | Geometric transformation to align image (rotation, scale, skew) |
| **Laplacian Variance** | Measure of image sharpness (blur detection) |

### System Architecture Terms

| Term | Definition |
|---|---|
| **Track** | Unique face instance being tracked across frames |
| **Track ID** | Unique identifier for a track (UUID) |
| **Motion-Gated** | Only run expensive operation when motion detected |
| **TTL (Time-To-Live)** | Expiration time for cached data |
| **Circuit Breaker** | Fail-safe pattern to prevent cascading failures |
| **Graceful Degradation** | System continues operation with reduced functionality on failure |
| **Multi-Frame Voting** | Aggregate decisions over N frames for confidence |

### Anti-Spoofing Terms

| Term | Definition |
|---|---|
| **Spoof** | Fake face presentation (photo, video, mask) |
| **Liveness** | Confirmation that presentation is genuine (alive) person |
| **Presentation Attack** | Intentional attempt to spoof biometric system |
| **Texture Analysis** | Surface property analysis to detect printed images |
| **Temporal Consistency** | Coherence of motion across time (detect video artifacts) |
| **Deep Fake** | AI-generated synthetic face video |

### Database Terms

| Term | Definition |
|---|---|
| **Collection** | MongoDB equivalent of SQL table |
| **Document** | MongoDB equivalent of SQL row (JSON-like) |
| **Index** | Data structure for fast lookups (B-tree) |
| **Unique Index** | Index enforcing one-to-one constraint |
| **Compound Index** | Index on multiple fields |
| **TTL Index** | Auto-delete documents after expiration time |
| **Projection** | SQL SELECT specific fields |
| **Aggregation Pipeline** | Multi-stage transformation (map-reduce-like) |

---

## Useful Commands

### Setting Up Environment

```bash
# Clone repository
git clone https://github.com/ShubhamPatra/attendance_system.git
cd attendance_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download ML models (required once)
python scripts/verify_versions.py  # Checks environment
python scripts/download_models.py  # Downloads YuNet, ArcFace, Silent-Face models
```

### Local Development

```bash
# Run admin app (port 5000)
python run_admin.py

# Run student app (port 5001)
python run_student.py

# Run both simultaneously (requires two terminals)
# Terminal 1: python run_admin.py
# Terminal 2: python run_student.py

# Access applications
# Admin: http://localhost:5000
# Student: http://localhost:5001
```

### Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f web          # Admin app logs
docker-compose logs -f student-web  # Student app logs

# Rebuild with GPU support (if NVIDIA available)
INSTALL_GPU=1 docker-compose build
```

### Database Operations

```bash
# Connect to MongoDB (local)
mongosh  # MongoDB Shell

# Connect to MongoDB Atlas (cloud)
mongosh "mongodb+srv://user:password@cluster.mongodb.net/attendance_system"
```

#### MongoDB Shell Commands

```bash
# Switch database
use attendance_system

# List collections
show collections

# Count students
db.students.countDocuments()

# Find attendance for student ID
db.attendance.find({ "student_id": "507f1f77bcf86cd799439011" })

# Count attendance marks for date
db.attendance.countDocuments({ "date": "2024-09-15" })

# Find active sessions
db.attendance_sessions.find({ "status": "active" })

# Export collection to CSV (from mongosh)
# Method 1: Use mongoexport (command line)
mongoexport --uri="mongodb://localhost" --db=attendance_system --collection=attendance --out=attendance.json

# Method 2: Aggregation pipeline (in mongosh)
db.attendance.aggregate([
  { $group: { _id: "$date", count: { $sum: 1 } } },
  { $sort: { _id: -1 } }
])
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_recognition.py

# Run with coverage
pytest --cov=. --cov-report=html
# View report: open htmlcov/index.html

# Run specific test
pytest tests/test_recognition.py::test_cosine_similarity_same_embedding

# Run with verbose output
pytest -v --tb=short

# Run only fast tests (skip integration tests)
pytest -m "not integration"
```

### Performance Analysis

```bash
# Run latency benchmark
python scripts/benchmark_latency.py

# Profile face recognition accuracy
python scripts/test_face_recognition.py

# Profile anti-spoofing accuracy
python scripts/test_anti_spoofing.py

# Generate performance report
python core/profiling.py  # Generates profiling.html
```

### Administrative Tasks

```bash
# Bootstrap admin user
python scripts/bootstrap_admin.py

# Seed demo data (for testing)
python scripts/seed_demo_data.py

# Clear database (WARNING: destructive)
python scripts/clear_db.py

# Export enrollment encodings
python scripts/migrate_encodings.py --export --output=encodings_backup.pkl

# Import enrollment encodings
python scripts/migrate_encodings.py --import --input=encodings_backup.pkl

# Verify all versions
python scripts/verify_versions.py
```

### Calibration

```bash
# Calibrate liveness threshold (determine optimal threshold)
python scripts/calibrate_liveness_threshold.py --dataset=tests/fixtures/real_faces --output=liveness_calibration.json

# Calibrate PPE detection threshold
python scripts/calibrate_ppe_threshold.py --dataset=tests/fixtures/ppe_dataset
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python run_admin.py

# Run smoke tests to verify setup
python scripts/smoke_test.py

# Debug face detection pipeline
python scripts/debug_pipeline.py --image=test_image.jpg
# Output: visualized detection, tracking, matching results

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import onnxruntime as rt; print(rt.get_available_providers())"
```

### Deployment

```bash
# Production startup (with Gunicorn)
gunicorn -c gunicorn.conf.py admin_app:app

# Monitor system
docker stats  # Docker container resource usage
top -u celery  # Celery task worker processes

# Health check
curl http://localhost:5000/health

# Graceful restart (Nginx + Gunicorn)
sudo systemctl reload nginx
sudo systemctl restart gunicorn
```

---

## Troubleshooting

### Common Issues

#### 1. "YuNet model not found" Error

**Symptom**: `RuntimeError: Failed to load YuNet model`

**Cause**: Models not downloaded.

**Solution**:
```bash
python scripts/verify_versions.py
python scripts/download_models.py
```

#### 2. "CUDA out of memory" Error

**Symptom**: `RuntimeError: CUDA out of memory`

**Cause**: GPU memory exhausted.

**Solutions**:
```bash
# Option 1: Reduce frame processing width
export ATTENDANCE_FRAME_PROCESS_WIDTH=384

# Option 2: Disable GPU (fallback to CPU)
export ATTENDANCE_ENABLE_GPU=0

# Option 3: Clear GPU cache (in Python)
import torch; torch.cuda.empty_cache()

# Option 4: Reduce batch size (if using batch processing)
export ATTENDANCE_BATCH_SIZE=4
```

#### 3. "No faces detected" in Classroom

**Symptom**: Face detection working in single images but fails in camera feed.

**Cause**: 
- Lighting too dim/bright
- Faces too small or too far
- Camera resolution too low
- Detection interval too high (skipping faces)

**Solutions**:
```bash
# Reduce detection interval (check faces more frequently)
export ATTENDANCE_DETECTION_INTERVAL=3

# Lower detection confidence (more sensitive, may increase false positives)
export ATTENDANCE_DETECTION_CONFIDENCE=0.4

# Increase frame processing width (preserve more detail)
export ATTENDANCE_FRAME_PROCESS_WIDTH=640

# Check lighting with debug script
python scripts/debug_pipeline.py --camera=0 --visualize=true
```

#### 4. "High False Positive Rate" (Wrong Students Marked)

**Symptom**: Attendance marks for students not present.

**Cause**:
- `RECOGNITION_THRESHOLD` too low (lenient matching)
- `LIVENESS_THRESHOLD` too low (accepts spoofs)
- `RECOGNITION_CONFIRM_FRAMES` too low (insufficient voting)

**Solutions**:
```bash
# Increase recognition threshold
export ATTENDANCE_RECOGNITION_THRESHOLD=0.42

# Increase liveness threshold
export ATTENDANCE_LIVENESS_THRESHOLD=0.65

# Require more frame confirmations
export ATTENDANCE_RECOGNITION_CONFIRM_FRAMES=3

# Increase liveness confirmation requirement
export ATTENDANCE_LIVENESS_CONFIRM_THRESHOLD=4
```

#### 5. "MongoDB Connection Refused"

**Symptom**: `pymongo.errors.ServerSelectionTimeoutError`

**Cause**: MongoDB not running or URI incorrect.

**Solutions**:
```bash
# Check if MongoDB is running (local)
mongosh

# If not running, start it
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # Mac
# or use MongoDB Atlas (cloud)

# Verify connection string
echo $MONGODB_URI
# Should be: mongodb://localhost (local)
# or: mongodb+srv://user:password@cluster.mongodb.net/attendance_system (Atlas)
```

#### 6. "Slow Face Recognition Latency"

**Symptom**: > 100ms per frame.

**Cause**: 
- CPU only (GPU disabled)
- Too many faces being tracked simultaneously
- Embedding cache disabled

**Solutions**:
```bash
# Enable GPU
export ATTENDANCE_ENABLE_GPU=1

# Enable embedding cache (default=True)
export ATTENDANCE_CACHE_EMBEDDINGS=1

# Reduce tracking limit
export ATTENDANCE_MAX_STUDENTS_PER_SESSION=100

# Lower frame processing width
export ATTENDANCE_FRAME_PROCESS_WIDTH=384

# Increase detection interval (skip frames)
export ATTENDANCE_DETECTION_INTERVAL=8
```

#### 7. "Student Enrollment Rejected Too Frequently"

**Symptom**: Most self-enrollment attempts fail verification.

**Cause**: 
- Thresholds too strict
- Image quality issues (lighting, pose)
- Model mismatch (uploaded with ArcFace, verified with dlib)

**Solutions**:
```bash
# Relax enrollment verification thresholds
# In student_app/verification.py, lower score thresholds:
AUTO_APPROVE_THRESHOLD = 80  # Default: 85

# Check image quality
python scripts/debug_pipeline.py --image=enrollment_sample.jpg --check_quality=true

# Ensure consistent backend
export ATTENDANCE_USE_ARCFACE=1
```

#### 8. "Anti-Spoofing Model Load Fails"

**Symptom**: `Warning: Silent-Face models failed to load. System will run without liveness check.`

**Cause**: 
- Silent-Face model file missing
- PyTorch not installed
- Model corrupted

**Solutions**:
```bash
# Re-download models
python scripts/download_models.py

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check model file exists
ls models/Silent-Face*/  # Should show model files

# Fallback: Disable anti-spoofing (not recommended)
export ATTENDANCE_LIVENESS_THRESHOLD=0.0  # All faces accepted as real
```

---

## API Reference

### RESTful Endpoints

#### Attendance Sessions

```http
POST /api/attendance/sessions
Content-Type: application/json

{
  "camera_id": "lab-1",
  "course_id": "CS101",
  "start_time": "2024-09-15T09:00:00Z"
}

Response (201):
{
  "session_id": "507f1f77bcf86cd799439011",
  "status": "active",
  "created_at": "2024-09-15T09:00:00Z"
}
```

```http
POST /api/attendance/sessions/<session_id>/end
Content-Type: application/json

Response (200):
{
  "session_id": "507f1f77bcf86cd799439011",
  "status": "closed",
  "duration_seconds": 3600,
  "students_marked": 45
}
```

#### Attendance Records

```http
GET /api/attendance?date=2024-09-15&course_id=CS101
Response (200):
[
  {
    "student_id": "507f1f77bcf86cd799439011",
    "date": "2024-09-15",
    "status": "Present",
    "confidence": 0.94
  },
  ...
]
```

```http
POST /api/attendance
Content-Type: application/json

{
  "marks": [
    {
      "student_id": "507f1f77bcf86cd799439011",
      "date": "2024-09-15",
      "status": "Present",
      "confidence": 0.94
    }
  ]
}

Response (201):
{ "inserted": 45 }
```

#### Health & Diagnostics

```http
GET /health

Response (200):
{
  "status": "healthy",
  "database": "connected",
  "models": {
    "yunet": "loaded",
    "arcface": "loaded",
    "silent_face": "loaded"
  },
  "uptime_seconds": 3600
}
```

---

## Database Queries

### Common MongoDB Queries

#### Attendance Summary by Date

```javascript
db.attendance.aggregate([
  { $match: { date: "2024-09-15" } },
  { $group: { 
    _id: "$status",
    count: { $sum: 1 },
    avg_confidence: { $avg: "$confidence" }
  } },
  { $sort: { _id: 1 } }
])

// Output:
// { "_id": "Absent", "count": 5, "avg_confidence": null }
// { "_id": "Present", "count": 45, "avg_confidence": 0.924 }
```

#### Student Attendance Trend (Last 30 Days)

```javascript
db.attendance.aggregate([
  { $match: { 
    student_id: ObjectId("507f1f77bcf86cd799439011"),
    date: { $gte: "2024-08-15", $lte: "2024-09-15" }
  } },
  { $group: { 
    _id: "$date",
    status: { $first: "$status" }
  } },
  { $sort: { _id: 1 } }
])
```

#### Sessions with Longest Duration

```javascript
db.attendance_sessions.find().sort({ duration_seconds: -1 }).limit(10)
```

#### Most Frequently Absent Students

```javascript
db.attendance.aggregate([
  { $match: { status: "Absent" } },
  { $group: { 
    _id: "$student_id",
    absences: { $sum: 1 }
  } },
  { $sort: { absences: -1 } },
  { $limit: 10 },
  { $lookup: {
    from: "students",
    localField: "_id",
    foreignField: "_id",
    as: "student"
  } }
])
```

#### Create TTL Index (Auto-Delete Old Records)

```javascript
// Delete attendance records older than 2 years
db.attendance.createIndex(
  { "created_at": 1 },
  { expireAfterSeconds: 63072000 }  // 2 years
)

// Check existing indexes
db.attendance.getIndexes()
```

---

**Last Updated**: April 16, 2026 | **Version**: 2.0.0
