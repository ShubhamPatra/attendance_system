# AutoAttendance

Real-time face recognition attendance system with deep learning anti-spoofing, MongoDB Atlas cloud storage, and a Flask web dashboard.

## Features

- **Real-time face detection and recognition** using a Detect-Track-Recognize pipeline (YuNet detection, CSRT tracking, dlib 128-D encodings)
- **Deep learning anti-spoofing** via Silent-Face-Anti-Spoofing (MiniFASNet) to prevent photo and video replay attacks
- **Eye-based face alignment** for improved recognition accuracy across varying camera angles
- **Vectorised face matching** against a thread-safe encoding cache for low-latency recognition
- **Duplicate attendance prevention** with a unique compound index (one record per student per day)
- **Motion-gated detection** to reduce unnecessary computation on idle frames
- **Performance metrics and auto-tuning** of recognition thresholds based on accuracy feedback
- **MongoDB Atlas** cloud database with indexed collections
- **Flask web dashboard** with registration, live attendance, dashboard, reports, logs, and metrics pages
- **CSV export** for attendance reports with date range and per-student filtering
- **Unknown face snapshots** saved automatically for review

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Flask 3.0.2 |
| Face Detection | YuNet (ONNX) |
| Face Recognition | face-recognition 1.3.0 / dlib |
| Object Tracking | CSRT (OpenCV Contrib), MIL fallback |
| Anti-Spoofing | Silent-Face-Anti-Spoofing (MiniFASNetV1SE, MiniFASNetV2) |
| Deep Learning | PyTorch |
| Database | MongoDB Atlas (pymongo 4.6.3) |
| Computer Vision | OpenCV 4.9.0.80 |
| Data Processing | NumPy 1.26.4, pandas 2.2.1 |

## Prerequisites

1. **Python 3.11** (not 3.12 -- dlib compatibility)
2. **CMake** (required by dlib) -- install via `pip install cmake` or from https://cmake.org
3. **MongoDB Atlas** free-tier cluster (see setup instructions below)
4. A webcam

## MongoDB Atlas Setup

1. Go to [https://cloud.mongodb.com](https://cloud.mongodb.com) and create a free account
2. Create a free shared cluster (M0)
3. Under **Database Access**, create a database user with read/write permissions
4. Under **Network Access**, add your IP address (or `0.0.0.0/0` for development)
5. Click **Connect > Drivers > Python 3.11+** and copy the SRV connection string
6. Replace `<password>` in the URI with your database user's password
7. The database `attendance_system` and its collections are created automatically on first run

## Installation

```bash
cd attendance_system

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/attendance_system?retryWrites=true&w=majority
SECRET_KEY=your-random-secret-key
```

Alternatively, set environment variables directly (PowerShell example):

```powershell
$env:MONGO_URI = "mongodb+srv://user:pass@cluster.mongodb.net/attendance_system?retryWrites=true&w=majority"
$env:SECRET_KEY = "change-me"
```

### Configurable Parameters

Key thresholds can be adjusted in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.55 | Euclidean distance cutoff for face matching (lower = stricter) |
| `LIVENESS_CONFIDENCE_THRESHOLD` | 0.80 | Minimum anti-spoof confidence to accept as real |
| `DETECTION_INTERVAL` | 10 | Run face detection every N frames |
| `RECOGNITION_COOLDOWN` | 30 | Seconds before re-processing a recognized student |
| `BLUR_THRESHOLD` | 100.0 | Laplacian variance minimum for image quality |
| `MAX_REGISTRATION_IMAGES` | 5 | Maximum face images accepted per student registration |

## Running the Application

```bash
cd attendance_system
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Usage

1. **Register students** at `/register` -- provide student details and upload up to 5 clear, front-facing face images
2. **Start attendance** at `/attendance` -- students face the webcam; the system verifies liveness and marks attendance automatically
3. **View dashboard** at `/dashboard` -- see today's attendance count, percentage, and recent records
4. **Generate reports** at `/report` -- filter by date or student and download CSV exports
5. **Monitor logs** at `/logs` -- view real-time recognition events
6. **Review metrics** at `/metrics` -- inspect system accuracy, FAR, FRR, and FPS

## Web Routes

### Pages

| Route | Description |
|---|---|
| `/` | Landing page |
| `/register` | Student registration form |
| `/attendance` | Live video feed with real-time recognition |
| `/dashboard` | Overview of today's attendance statistics |
| `/report` | Attendance records with date filtering and CSV export |
| `/logs` | Real-time recognition log viewer |
| `/metrics` | System performance metrics dashboard |
| `/attendance_activity` | Attendance by hour of day visualization |

### API Endpoints

| Route | Description |
|---|---|
| `GET /video_feed` | MJPEG stream of annotated camera frames |
| `GET /api/metrics` | JSON performance metrics (accuracy, FAR, FRR, FPS, threshold) |
| `GET /api/events` | JSON recent attendance events |
| `GET /api/logs` | JSON recognition log buffer |
| `GET /api/attendance_activity` | JSON hourly attendance counts |
| `GET /api/registration_numbers` | JSON list of all registration numbers |
| `GET /report/csv` | CSV export with optional date, reg_no, and date range parameters |

## Pipeline Architecture

The core processing pipeline operates in four phases per frame:

1. **Track Update** -- All active CSRT trackers are advanced. Tracks that have not been updated for more than 15 consecutive frames are expired.

2. **Detection (gated)** -- Every 10 frames, motion detection (Gaussian blur + frame differencing) gates YuNet face detection. If no motion is detected, a fallback detection still runs every 50 frames. New detections are associated to existing tracks via IoU and centroid distance thresholds.

3. **Recognition** -- For each newly created track:
   - Anti-spoof inference runs on the full-resolution frame using two MiniFASNet models with aggregated predictions.
   - If classified as real (confidence >= 0.8), eye-based affine alignment is applied and a 128-D encoding is generated.
   - The encoding is matched against the in-memory cache using vectorised Euclidean distance.
   - On match, attendance is recorded in MongoDB (subject to a 30-second cooldown).
   - Spoofed faces are flagged and logged. Unknown faces are saved to `unknown_faces/`.

4. **Overlay** -- Bounding boxes and labels are drawn per track: green for recognized students, red for spoofed or unknown faces, with name, confidence, and liveness score annotations.

## Project Structure

```
attendance_system/
├── app.py                # Flask application entry point
├── config.py             # Environment variables and constants
├── camera.py             # Threaded webcam capture and pipeline orchestration
├── pipeline.py           # YuNet detection, CSRT tracking, IoU/centroid association
├── face_engine.py        # Encoding generation, thread-safe cache, vectorised matching
├── recognition.py        # Eye-based affine alignment for camera-path encoding
├── anti_spoofing.py      # Deep learning liveness via Silent-Face-Anti-Spoofing
├── liveness.py           # Compatibility re-exports from anti_spoofing
├── overlay.py            # Bounding box and label rendering on frames
├── database.py           # MongoDB Atlas connection, CRUD, aggregation, CSV export
├── routes.py             # Flask blueprint with all web and API routes
├── utils.py              # Input sanitisation, file validation, image quality, logging
├── performance.py        # Metrics tracking (accuracy, FAR, FRR, FPS) and auto-tuning
├── requirements.txt      # Pinned dependencies
├── models/               # YuNet ONNX model for face detection
├── logs/                 # Application logs (auto-created, daily rotation, 30-day retention)
├── unknown_faces/        # Snapshots of unrecognised faces (auto-created)
├── uploads/              # Temporary upload directory for registration images
├── Silent-Face-Anti-Spoofing/  # Anti-spoofing library and pre-trained models
│   └── resources/
│       └── anti_spoof_models/  # MiniFASNetV2 and MiniFASNetV1SE weights
├── templates/            # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── register.html
│   ├── attendance.html
│   ├── attendance_activity.html
│   ├── dashboard.html
│   ├── report.html
│   ├── logs.html
│   └── metrics.html
├── static/
│   └── css/style.css
└── tests/                # Unit tests (mocked, no live DB or webcam required)
    ├── test_anti_spoofing.py
    ├── test_database.py
    ├── test_liveness.py
    ├── test_performance.py
    ├── test_recognition.py
    ├── test_routes.py
    └── test_utils.py
```

## Performance Metrics

The system tracks and exposes the following metrics at `/api/metrics`:

| Metric | Description |
|---|---|
| Accuracy | (TP + TN) / Total |
| FAR | False Acceptance Rate -- proportion of impostors incorrectly accepted |
| FRR | False Rejection Rate -- proportion of genuine users incorrectly rejected |
| Avg Frame Time | Average processing time per frame (ms) |
| FPS | Frames per second (rolling window of last 500 frames) |

### Auto-Tuning

The recognition threshold is automatically adjusted every 200 recognitions:

- Accuracy below 95%: threshold decreases by 0.02 (stricter matching)
- Accuracy above 98%: threshold increases by 0.01 (more lenient matching)
- Threshold is clamped to the range [0.45, 0.60]

## Database Schema

### students

| Field | Type | Description |
|---|---|---|
| `name` | String | Student name |
| `semester` | String | Current semester |
| `registration_number` | String | Unique identifier (indexed) |
| `section` | String | Class section |
| `encodings` | Array of Binary | 128-D float64 face encoding vectors |
| `created_at` | DateTime | Registration timestamp (UTC) |

### attendance

| Field | Type | Description |
|---|---|---|
| `student_id` | ObjectId | Reference to students collection |
| `date` | String | Date in YYYY-MM-DD format |
| `time` | String | Time in HH:MM:SS format |
| `status` | String | Attendance status |
| `confidence_score` | Float | Recognition confidence |

A unique compound index on `(student_id, date)` enforces one attendance record per student per day.

## Security

- MongoDB URI loaded from environment variables, never hardcoded
- Deep learning anti-spoofing to prevent photo and video replay attacks
- Image upload validation with extension whitelist, MIME type verification (magic bytes), and 5 MB size limit
- Image quality gates: blur detection (Laplacian variance) and brightness check
- Input sanitisation with HTML tag stripping, whitespace normalisation, and length truncation
- Unique database constraints preventing duplicate registrations and attendance records
- Recognition cooldown preventing repeated marking within 30 seconds
- Bounded in-memory buffers (events: 50, logs: 200) to prevent memory growth
- MongoDB connection timeouts (5 seconds) for resilience
- Date parameter validation with strict format parsing
- Daily rotating log files with 30-day retention

## Running Tests

```bash
cd attendance_system
pytest tests/ -v
```

All tests use mocks and do not require a live MongoDB connection or webcam. The test suite covers anti-spoofing, database operations, recognition, performance metrics, route handling, and utility functions.

## Improvement Roadmap

1. **Authentication** -- Add Flask-Login for admin and teacher access control
2. **Multi-camera support** -- Queue-based architecture for multiple attendance stations
3. **Face thumbnails** -- Store cropped face images for audit trail
4. **Notification system** -- Email or SMS alerts for absences
5. **HTTPS deployment** -- Deploy behind Nginx with SSL certificates
6. **Docker containerisation** -- Dockerfile and docker-compose for portable deployment
7. **REST API** -- Full CRUD API for mobile application integration
8. **Batch registration** -- Upload CSV and image archive for bulk student enrolment
9. **Attendance analytics** -- Weekly and monthly trends, prediction models
