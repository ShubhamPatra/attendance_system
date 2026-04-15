# AutoAttendance

Production-grade face recognition attendance system for educational institutions.

## Overview

AutoAttendance includes:

- Admin web app for student/course/attendance management
- Student web app for webcam-based attendance marking
- Face recognition and anti-spoofing pipeline
- MongoDB + Redis + Celery background processing
- Docker-based deployment with Nginx reverse proxy

## Tech Stack

- Python 3.11
- Flask, Flask-SocketIO, Flask-Login, Flask-Limiter
- OpenCV, InsightFace, ONNX Runtime, PyTorch
- MongoDB, Redis, Celery

## Quick Start (Local)

1. Create virtual environment:

```powershell
python -m venv venv
```

1. Activate virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

1. Install dependencies:

```powershell
pip install -r requirements.txt
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

1. Create local environment file:

```powershell
Copy-Item .env.example .env
```

1. Run apps:

```powershell
python run_admin.py
python run_student.py
```

Open the apps in your browser at:

- Admin: http://localhost:5000
- Student: http://localhost:5001

## Configuration

All runtime configuration is environment-driven. Use `.env.example` as the baseline.

## Testing

```powershell
pytest tests/ -v
```

## Deployment

Container manifests are located in `docker/` with reverse proxy config in `docker/nginx/nginx.conf`.

## Operations

- Download ML models: `python scripts/download_models.py`
- Validate existing models: `python scripts/download_models.py --validate-only`
- Use the production env template: copy `.env.production` and fill in real secrets/connection strings
- Common container commands: `make build`, `make up`, `make down`, `make logs`, `make test`, `make shell`
- Seed development data: `make seed`
