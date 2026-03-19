"""
Configuration module for AutoAttendance.
Loads settings from environment variables and defines constants.
"""

import os
import secrets
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise EnvironmentError(
        "MONGO_URI environment variable is not set. "
        "Set it to your MongoDB Atlas SRV connection string."
    )

DATABASE_NAME = "attendance_system"

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)

# ---------------------------------------------------------------------------
# Face Recognition
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD = 0.50  # Euclidean distance; lower = stricter
FRAME_RESIZE_FACTOR = 0.25    # Resize frames to 1/4 for speed
RECOGNITION_COOLDOWN = 30     # Seconds to skip re-processing a recognized student

# ---------------------------------------------------------------------------
# Detect-Track-Recognize Pipeline
# ---------------------------------------------------------------------------
DETECTION_INTERVAL = 10        # Run face detection every N-th frame
TRACK_EXPIRATION_FRAMES = 30   # Remove track after N consecutive failed updates
MOTION_THRESHOLD = 5000        # Min non-zero pixels to consider "motion detected"

# ---------------------------------------------------------------------------
# Liveness / Anti-Spoofing
# ---------------------------------------------------------------------------
LIVENESS_CONFIDENCE_THRESHOLD = float(
    os.environ.get("LIVENESS_CONFIDENCE_THRESHOLD", "0.8")
)

# ---------------------------------------------------------------------------
# Image Quality
# ---------------------------------------------------------------------------
BLUR_THRESHOLD = 10.0              # Laplacian variance; below = too blurry
BRIGHTNESS_THRESHOLD = 40.0        # Mean pixel brightness; below = too dark
MAX_REGISTRATION_IMAGES = 5        # Max images accepted per registration

# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------
UPLOAD_MAX_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------------------------------------------------------------------------
# Frame Processing
# ---------------------------------------------------------------------------
FRAME_PROCESS_WIDTH = 640          # Resize frames to this width before processing

# ---------------------------------------------------------------------------
# Motion & Detection Gating
# ---------------------------------------------------------------------------
NO_MOTION_DETECTION_INTERVAL = 30  # Run detection every N frames even without motion

# ---------------------------------------------------------------------------
# Track Association
# ---------------------------------------------------------------------------
IOU_THRESHOLD = 0.3                # Min IoU to match detection -> track
CENTROID_DISTANCE_THRESHOLD = 100  # Max centroid distance (px) for match

# ---------------------------------------------------------------------------
# MJPEG Streaming
# ---------------------------------------------------------------------------
MJPEG_TARGET_FPS = 12              # Cap MJPEG stream frame rate

# ---------------------------------------------------------------------------
# Bounded Buffers
# ---------------------------------------------------------------------------
EVENT_BUFFER_MAX = 50              # Max attendance events in deque
LOG_BUFFER_MAX = 200               # Max log entries in deque

# ---------------------------------------------------------------------------
# YuNet Face Detector
# ---------------------------------------------------------------------------
YUNET_SCORE_THRESHOLD = 0.7        # Detection confidence threshold
YUNET_NMS_THRESHOLD = 0.3          # Non-maximum suppression threshold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SHAPE_PREDICTOR_PATH = os.path.join(
    MODELS_DIR, "shape_predictor_68_face_landmarks.dat"
)
YUNET_MODEL_PATH = os.path.join(
    MODELS_DIR, "face_detection_yunet_2023mar.onnx"
)
ANTI_SPOOF_MODEL_DIR = os.environ.get(
    "ANTI_SPOOF_MODEL_DIR",
    os.path.join(BASE_DIR, "Silent-Face-Anti-Spoofing", "resources", "anti_spoof_models"),
)
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, "unknown_faces")
UNKNOWN_FACE_COOLDOWN = 10         # Seconds between unknown-face snapshot saves

# ---------------------------------------------------------------------------
# Incremental Learning
# ---------------------------------------------------------------------------
INCREMENTAL_LEARNING_CONFIDENCE = float(
    os.environ.get("INCREMENTAL_LEARNING_CONFIDENCE", "0.92")
)  # Threshold above which a new encoding is appended automatically
MAX_ENCODINGS_PER_STUDENT = int(
    os.environ.get("MAX_ENCODINGS_PER_STUDENT", "15")
)  # Cap per student; oldest encodings are dropped when exceeded

# ---------------------------------------------------------------------------
# Recognition Quality Gate
# ---------------------------------------------------------------------------
BRIGHTNESS_MAX = float(
    os.environ.get("BRIGHTNESS_MAX", "250.0")
)  # Max mean brightness for face ROI during pipeline recognition

# ---------------------------------------------------------------------------
# Celery + Redis
# ---------------------------------------------------------------------------
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# SendGrid / Notifications
# ---------------------------------------------------------------------------
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "")
ABSENCE_THRESHOLD = int(os.environ.get("ABSENCE_THRESHOLD", "75"))

# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------
SUBJECTS = [s.strip() for s in os.environ.get("SUBJECTS", "General").split(",")]

# ---------------------------------------------------------------------------
# Multi-Camera
# ---------------------------------------------------------------------------
CAMERA_INDICES = [int(x) for x in os.environ.get("CAMERA_INDICES", "0").split(",")]

# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------
BACKUP_DIR = os.path.join(BASE_DIR, "backups")
BACKUP_RETENTION_DAYS = int(os.environ.get("BACKUP_RETENTION_DAYS", "30"))

# ---------------------------------------------------------------------------
# Upload Directory
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
UPLOAD_RETENTION_SECONDS = int(
    os.environ.get("UPLOAD_RETENTION_SECONDS", "3600")
)

# ---------------------------------------------------------------------------
# API Rate Limiting
# ---------------------------------------------------------------------------
API_RATE_LIMIT_WINDOW_SEC = int(
    os.environ.get("API_RATE_LIMIT_WINDOW_SEC", "60")
)
API_RATE_LIMIT_MAX_REQUESTS = int(
    os.environ.get("API_RATE_LIMIT_MAX_REQUESTS", "30")
)

# ---------------------------------------------------------------------------
# SocketIO / CORS
# ---------------------------------------------------------------------------
SOCKETIO_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "SOCKETIO_CORS_ORIGINS",
        "http://localhost:5000,http://127.0.0.1:5000",
    ).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Debug / Diagnostic Bypass Flags
# ---------------------------------------------------------------------------
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
BYPASS_ANTISPOOF = os.environ.get("BYPASS_ANTISPOOF", "0") == "1"
BYPASS_MOTION_DETECTION = os.environ.get("BYPASS_MOTION_DETECTION", "0") == "1"
BYPASS_QUALITY_GATE = os.environ.get("BYPASS_QUALITY_GATE", "0") == "1"
