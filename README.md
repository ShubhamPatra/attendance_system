# AutoAttendance

AutoAttendance is a Flask + OpenCV + MongoDB Atlas attendance platform with anti-spoofing, an open-access staff/admin interface (no auth enforcement), a standalone student self-service portal, analytics, Celery background tasks, and deployment-ready Docker/Kubernetes artifacts.

## Overview

The system keeps the original Detect-Track-Recognize pipeline and adds production controls around it:

- Flask-Login auth with Admin and Teacher roles
- Authentication/RBAC enforcement disabled by default
- Optional RESTX docs for `/api/v2/*` (feature-flag controlled)
- Docker, compose, and Nginx deployment support
- Admin student management and batch import workflows
- Analytics dashboards and notification dry-run events
- Standalone student portal app for registration and attendance status
- Multi-camera diagnostics without changing single-camera defaults

## Architecture

```text
Browser UI
    -> Flask app
         -> open-access admin routes (no auth gate)
         -> REST APIs and RESTX docs
         -> Dashboard / reports / batch import
         -> Camera routes
                -> per-camera threaded capture
                -> YuNet detection
                -> CSRT tracking
                -> Anti-spoofing
                -> ArcFace embedding and identity matching
                -> MongoDB attendance writes
                -> metrics / logs / snapshots
Student Browser UI
    -> Flask student app
         -> /student/login, /student/register, /student/capture
         -> student attendance status and verification flow
MongoDB Atlas
    -> students, attendance, users, notification_events
```

## Tech Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.11 |
| Web | Flask 3.0.2 |
| Auth | Flask-Login |
| Docs | Flasgger + Flask-RESTX |
| Face Detection | YuNet ONNX |
| Face Recognition | ArcFace (InsightFace + ONNX Runtime) |
| Anti-Spoofing | Silent-Face-Anti-Spoofing |
| Database | MongoDB Atlas / pymongo |
| Queue / async | Celery |
| Realtime transport | Flask-SocketIO |
| Broker/cache | Redis |
| Container | Docker / Docker Compose |
| Proxy | Nginx |

## Features

- Real-time camera attendance with anti-spoofing and face recognition
- Open-access staff/admin routes and APIs (no login required)
- Dashboard charts and attendance analytics
- Batch registration from CSV + ZIP image archives
- Admin controls for edit, delete, and encoding recompute
- Dry-run notification events for absences and low attendance
- Student portal with verification workflow and attendance self-check
- Multi-camera diagnostics
- Centralized logging and health endpoints
- RESTX docs for `/api/v2/*` when enabled

## Quick Start

### Local dev

```bash
cd attendance_system
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

In a second terminal, run the student portal:

```bash
python student_app/app.py
```

Set at minimum:

```env
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/attendance_system?retryWrites=true&w=majority
SECRET_KEY=change-me
```

`ENABLE_RBAC` defaults to `0` (auth disabled). Set `ENABLE_RBAC=1` only if you intentionally re-enable auth checks.

Recommended local toggles:

```env
APP_DEBUG=0
ENABLE_RESTX_API=0
STRICT_STARTUP_CHECKS=1
STARTUP_CAMERA_PROBE=0
```

### First admin

```bash
python scripts/bootstrap_admin.py --username admin --role admin
```

### Demo data

```bash
python scripts/seed_demo_data.py
```

This seeds demo students and a default admin account (`admin / admin1234`).

## Docker

```bash
docker compose up -d --build
```

Services:

- `web` Flask/Gunicorn app
- `student-web` standalone student portal
- `celery-worker` background jobs
- `celery-beat` scheduled jobs
- `redis` queue backend
- `nginx` reverse proxy

Useful URLs:

- Main app (via Nginx): `http://localhost`
- Student portal (direct container): `http://localhost:5001/student/login`
- Legacy docs (when RESTX disabled): `http://localhost/api/docs`
- RESTX docs (when `ENABLE_RESTX_API=1`): `http://localhost/api/v2/docs`

## Deployment

Production notes and Kubernetes baseline are documented in [DEPLOYMENT.md](DEPLOYMENT.md).

### VM deployment

Use the provided Gunicorn config and Nginx sample:

- [gunicorn.conf.py](gunicorn.conf.py)
- [deploy/nginx/default.conf](deploy/nginx/default.conf)

### Kubernetes baseline

- [deploy/k8s/README.md](deploy/k8s/README.md)
- [deploy/k8s/autoattendance.yaml](deploy/k8s/autoattendance.yaml)

## Key Routes

### Pages

| Route | Purpose |
| --- | --- |
| `/` | Landing page |
| `/login` | Staff login |
| `/dashboard` | Attendance overview |
| `/attendance` | Live camera attendance |
| `/register` | Student registration |
| `/register/batch` | Batch CSV + ZIP import |
| `/admin/students` | Admin student management |
| `/report` | Attendance report viewer |
| `/logs` | Recognition logs |
| `/metrics` | Runtime metrics |
| `/heatmap` | Attendance heatmap |
| `/attendance_activity` | Hourly attendance chart |
| `/student` | Redirect to standalone student portal login |
| `/student/*` | Student portal routes (served by student app) |

### APIs

| Route | Purpose |
| --- | --- |
| `GET /api/metrics` | Performance metrics |
| `GET /api/events` | Recent attendance events |
| `GET /api/logs` | Persistent log buffer |
| `GET /api/attendance_activity` | Hourly attendance counts |
| `GET /api/heatmap` | Attendance heatmap data |
| `GET /api/registration_numbers` | Registration number list |
| `GET /api/analytics/trends` | Daily attendance trend data |
| `GET /api/analytics/at_risk` | Low-attendance students |
| `GET /api/cameras` | Multi-camera diagnostics |
| `GET /health` | Liveness |
| `GET /ready` | Readiness |
| `GET /api/v2/docs` | RESTX docs |

## Pipeline

Per frame:

1. Capture and track update
2. Motion-gated YuNet detection
3. Anti-spoofing on new tracks only
4. Face embedding extraction and identity matching
5. Attendance write with cooldown
6. Overlay + metrics + logs

Unknown face snapshots now use a bounded background executor, so repeated unknown detections do not create unbounded threads.

## Project Layout

- `app.py`: main staff/admin Flask app factory and startup checks
- `student_app/`: standalone student portal Flask app
- `app_web/`: route registration and web/API endpoint modules
- `app_vision/`: detection, anti-spoofing, recognition, overlays, and pipeline logic
- `app_core/`: shared config, auth, database, notifications, and utilities
- `app_camera/`: camera stream management and SocketIO integration
- `scripts/`: bootstrap, calibration, migration, seed, and smoke-test helpers
- `deploy/`: Nginx and Kubernetes deployment assets

## Database Schema

### students

| Field | Type |
| --- | --- |
| `name` | String |
| `semester` | Integer |
| `registration_number` | String |
| `section` | String |
| `email` | String |
| `encodings` | Array of Binary |
| `password_hash` | String (nullable) |
| `verification_status` | String (`pending`, `approved`, `rejected`) |
| `verification_score` | Float (nullable) |
| `verification_reason` | String (nullable) |
| `face_samples` | Array of String |
| `verification_updated_at` | DateTime |
| `created_at` | DateTime |
| `approved_at` | DateTime (nullable) |
| `rejected_at` | DateTime (nullable) |

### users

| Field | Type |
| --- | --- |
| `username` | String |
| `password_hash` | String |
| `role` | `admin` or `teacher` |
| `email` | String |
| `is_active` | Boolean |
| `created_at` | DateTime |
| `last_login` | DateTime (nullable) |

### attendance

| Field | Type |
| --- | --- |
| `student_id` | ObjectId |
| `date` | String YYYY-MM-DD |
| `time` | String HH:MM:SS |
| `status` | Present / Absent |
| `confidence_score` | Float |

### notification_events

| Field | Type |
| --- | --- |
| `event_type` | String |
| `recipient` | String |
| `subject` | String |
| `payload` | Object |
| `mode` | String |
| `created_at` | DateTime |

## Tests

```bash
pytest -q
```

Current targeted checks pass in this repo, including route, database, camera multi-camera, notifications, batch import, and admin workflow coverage.

Run a focused suite quickly:

```bash
pytest -q tests/test_routes.py tests/test_database.py tests/test_camera_pipeline.py
```

## Smoke Test

```bash
python scripts/smoke_test.py --base-url http://localhost --check-video
```

## Environment Flags

Important flags in [`.env.example`](.env.example):

- `EMBEDDING_BACKEND`
- `APP_DEBUG`
- `ENABLE_RBAC` (defaults to `0`; set `1` only if you want to re-enable auth checks)
- `ENABLE_RESTX_API`
- `APP_HOST`, `APP_PORT` (and optional `STUDENT_APP_HOST`, `STUDENT_APP_PORT`)
- `STRICT_STARTUP_CHECKS`, `STARTUP_CAMERA_PROBE`
- `SESSION_COOKIE_SECURE`
- `SESSION_COOKIE_SAMESITE`
- `SOCKETIO_CORS_ORIGINS`
- `DEBUG_MODE`
- `BYPASS_ANTISPOOF`

## Notes

- Keep Python at 3.11 for dependency compatibility and stable model runtimes.
- Auth enforcement is currently disabled by default. Re-enable with `ENABLE_RBAC=1` only if you also restore role/login checks.
- Use MongoDB Atlas backups for the main database.
- Keep `STRICT_STARTUP_CHECKS=1` in production.
- Batch import expects CSV columns: `registration_number,name,semester,section,email`.

## Troubleshooting

### `python app.py` fails with MongoDB connection refused (`localhost:27017`)

If startup logs show connection attempts to `localhost:27017` but your `.env` has an Atlas URI:

1. Clear conflicting shell/system `MONGO_URI` variables.
2. Confirm `.env` has the expected `MONGO_URI` value.
3. Re-run the app from the repository root.

This project now loads `.env` with override enabled by default, so local `.env` values take precedence. Set `DOTENV_OVERRIDE=0` if you intentionally want shell/system environment variables to win.
