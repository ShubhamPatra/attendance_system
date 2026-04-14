"""
Configuration module for AutoAttendance.
Loads settings from environment variables and defines constants.
"""

import os
import secrets
from pathlib import Path
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
MONGO_SERVER_SELECTION_TIMEOUT_MS = int(
    os.environ.get("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000")
)
MONGO_CONNECT_TIMEOUT_MS = int(
    os.environ.get("MONGO_CONNECT_TIMEOUT_MS", "5000")
)
MONGO_CONNECT_RETRIES = int(
    os.environ.get("MONGO_CONNECT_RETRIES", "3")
)
MONGO_CONNECT_RETRY_DELAY_SECONDS = float(
    os.environ.get("MONGO_CONNECT_RETRY_DELAY_SECONDS", "1.0")
)

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)

APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "5000"))
APP_DEBUG = os.environ.get("APP_DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# Face Recognition
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD = float(
    os.environ.get("RECOGNITION_THRESHOLD", "0.50")
)  # Euclidean distance; lower = stricter
RECOGNITION_MIN_CONFIDENCE = float(
    os.environ.get("RECOGNITION_MIN_CONFIDENCE", "0.55")
)
RECOGNITION_MIN_DISTANCE_GAP = float(
    os.environ.get("RECOGNITION_MIN_DISTANCE_GAP", "0.04")
)
RECOGNITION_CONFIDENCE_ALPHA = float(
    os.environ.get("RECOGNITION_CONFIDENCE_ALPHA", "2.3")
)
RECOGNITION_CONFIRM_FRAMES = int(
    os.environ.get("RECOGNITION_CONFIRM_FRAMES", "2")
)
FRAME_RESIZE_FACTOR = 0.25    # Resize frames to 1/4 for speed
RECOGNITION_COOLDOWN = 30     # Seconds to skip re-processing a recognized student

# ---------------------------------------------------------------------------
# Detect-Track-Recognize Pipeline
# ---------------------------------------------------------------------------
DETECTION_INTERVAL = 10        # Run face detection every N-th frame
DETECTION_INTERVAL_MIN = int(os.environ.get("DETECTION_INTERVAL_MIN", "3"))
DETECTION_INTERVAL_MAX = int(os.environ.get("DETECTION_INTERVAL_MAX", "15"))
TRACK_EXPIRATION_FRAMES = 30   # Remove track after N consecutive failed updates
TRACK_DETECTOR_MISS_TOLERANCE = 2  # Expire track after N detector cycles without support
MOTION_THRESHOLD = 5000        # Min non-zero pixels to consider "motion detected"

# ---------------------------------------------------------------------------
# Liveness / Anti-Spoofing
# ---------------------------------------------------------------------------
LIVENESS_CONFIDENCE_THRESHOLD = float(
    os.environ.get("LIVENESS_CONFIDENCE_THRESHOLD", "0.55")
)
LIVENESS_REAL_FAST_CONFIDENCE = float(
    os.environ.get("LIVENESS_REAL_FAST_CONFIDENCE", "0.72")
)
LIVENESS_HISTORY_SIZE = int(
    os.environ.get("LIVENESS_HISTORY_SIZE", "8")
)
LIVENESS_MIN_HISTORY = int(
    os.environ.get("LIVENESS_MIN_HISTORY", "3")
)
LIVENESS_REAL_VOTE_RATIO = float(
    os.environ.get("LIVENESS_REAL_VOTE_RATIO", "0.7")
)
LIVENESS_SPOOF_VOTE_RATIO = float(
    os.environ.get("LIVENESS_SPOOF_VOTE_RATIO", "0.6")
)
LIVENESS_SPOOF_CONFIDENCE_MIN = float(
    os.environ.get("LIVENESS_SPOOF_CONFIDENCE_MIN", "0.6")
)
LIVENESS_SPOOF_WEAK_CONFIDENCE_MIN = float(
    os.environ.get("LIVENESS_SPOOF_WEAK_CONFIDENCE_MIN", "0.45")
)
LIVENESS_STRONG_SPOOF_CONFIDENCE = float(
    os.environ.get("LIVENESS_STRONG_SPOOF_CONFIDENCE", "0.85")
)
LIVENESS_NO_ENCODE_MARGIN = float(
    os.environ.get("LIVENESS_NO_ENCODE_MARGIN", "0.03")
)
SPOOF_HOLD_SECONDS = float(
    os.environ.get("SPOOF_HOLD_SECONDS", "1.5")
)
TRACK_STATE_PENDING_SECONDS = float(
    os.environ.get("TRACK_STATE_PENDING_SECONDS", "2.5")
)
ANTI_SPOOF_PAD_RATIO_BASE = float(
    os.environ.get("ANTI_SPOOF_PAD_RATIO_BASE", "0.12")
)
ANTI_SPOOF_PAD_RATIO_MAX = float(
    os.environ.get("ANTI_SPOOF_PAD_RATIO_MAX", "0.3")
)
ANTI_SPOOF_PAD_MIN_PIXELS = int(
    os.environ.get("ANTI_SPOOF_PAD_MIN_PIXELS", "8")
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
INCREMENTAL_LEARNING_MIN_LIVENESS = float(
    os.environ.get("INCREMENTAL_LEARNING_MIN_LIVENESS", "0.60")
)  # Minimum liveness confidence required before appending a new encoding
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
# PPE Detection / Occlusion-Aware Recognition
# ---------------------------------------------------------------------------
PPE_DETECTION_ENABLED = os.environ.get("PPE_DETECTION_ENABLED", "0") == "1"
PPE_MODEL_PATH = os.environ.get(
    "PPE_MODEL_PATH",
    os.path.join(MODELS_DIR, "ppe_mask_cap.onnx"),
)
PPE_MASK_THRESHOLD = float(os.environ.get("PPE_MASK_THRESHOLD", "0.6"))
PPE_CAP_THRESHOLD = float(os.environ.get("PPE_CAP_THRESHOLD", "0.6"))
PPE_MIN_CONFIDENCE = float(os.environ.get("PPE_MIN_CONFIDENCE", "0.55"))
PPE_HISTORY_SIZE = int(os.environ.get("PPE_HISTORY_SIZE", "6"))
PPE_MIN_HISTORY = int(os.environ.get("PPE_MIN_HISTORY", "3"))
PPE_VOTE_RATIO = float(os.environ.get("PPE_VOTE_RATIO", "0.6"))
PPE_MAX_LATENCY_MS = float(os.environ.get("PPE_MAX_LATENCY_MS", "60.0"))

OCCLUDED_RECOGNITION_THRESHOLD_DELTA = float(
    os.environ.get("OCCLUDED_RECOGNITION_THRESHOLD_DELTA", "0.05")
)
OCCLUDED_MIN_DISTANCE_GAP = float(
    os.environ.get("OCCLUDED_MIN_DISTANCE_GAP", "0.08")
)
OCCLUDED_MIN_CONFIDENCE = float(
    os.environ.get("OCCLUDED_MIN_CONFIDENCE", "0.5")
)
OCCLUDED_DISABLE_INCREMENTAL_LEARNING = (
    os.environ.get("OCCLUDED_DISABLE_INCREMENTAL_LEARNING", "1") == "1"
)

# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------
CELERY_DATA_DIR = os.environ.get(
    "CELERY_DATA_DIR",
    os.path.join(BASE_DIR, "celery_data"),
)
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "filesystem://")
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND",
    str(Path(os.path.join(CELERY_DATA_DIR, "results")).resolve().as_uri()),
)

# ---------------------------------------------------------------------------
# Attendance Analytics
# ---------------------------------------------------------------------------
ABSENCE_THRESHOLD = int(os.environ.get("ABSENCE_THRESHOLD", "75"))

# ---------------------------------------------------------------------------
# Multi-Camera
# ---------------------------------------------------------------------------
CAMERA_INDICES = [int(x) for x in os.environ.get("CAMERA_INDICES", "0").split(",")]
CAMERA_HEALTHCHECK_INDEX = int(
    os.environ.get("CAMERA_HEALTHCHECK_INDEX", str(CAMERA_INDICES[0] if CAMERA_INDICES else 0))
)

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

# ---------------------------------------------------------------------------
# Startup Diagnostics
# ---------------------------------------------------------------------------
STRICT_STARTUP_CHECKS = os.environ.get("STRICT_STARTUP_CHECKS", "1") == "1"
STARTUP_CAMERA_PROBE = os.environ.get("STARTUP_CAMERA_PROBE", "0") == "1"
