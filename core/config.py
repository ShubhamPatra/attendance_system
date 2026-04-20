"""
Configuration module for AutoAttendance.
Loads settings from environment variables and defines constants.
"""

import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Prefer repository .env values for local runs unless explicitly disabled.
load_dotenv(override=os.environ.get("DOTENV_OVERRIDE", "1") == "1")

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

# MongoDB connection pooling
MONGO_MAX_POOL_SIZE = int(
    os.environ.get("MONGO_MAX_POOL_SIZE", "50")
)
MONGO_MIN_POOL_SIZE = int(
    os.environ.get("MONGO_MIN_POOL_SIZE", "5")
)
MONGO_MAX_IDLE_TIME_MS = int(
    os.environ.get("MONGO_MAX_IDLE_TIME_MS", "45000")
)

# MongoDB circuit breaker (prevent cascading failures)
MONGO_CIRCUIT_BREAKER_THRESHOLD = int(
    os.environ.get("MONGO_CIRCUIT_BREAKER_THRESHOLD", "5")
)
MONGO_CIRCUIT_BREAKER_TIMEOUT_SECONDS = float(
    os.environ.get("MONGO_CIRCUIT_BREAKER_TIMEOUT_SECONDS", "60.0")
)

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    import warnings
    warnings.warn(
        "SECRET_KEY not set in environment. Generating random key for this session. "
        "Set SECRET_KEY environment variable for production use (persists across restarts).",
        RuntimeWarning
    )
    SECRET_KEY = secrets.token_hex(32)

APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "5000"))
APP_DEBUG = os.environ.get("APP_DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# JWT / API Authentication
# ---------------------------------------------------------------------------
JWT_ACCESS_TOKEN_LIFETIME_HOURS = int(
    os.environ.get("JWT_ACCESS_TOKEN_LIFETIME_HOURS", "1")
)
JWT_REFRESH_TOKEN_LIFETIME_HOURS = int(
    os.environ.get("JWT_REFRESH_TOKEN_LIFETIME_HOURS", "24")
)
JWT_ALGORITHM = "HS256"

# ---------------------------------------------------------------------------
# ML backend
# ---------------------------------------------------------------------------
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "arcface").strip().lower()
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "512"))
ENABLE_GPU_PROVIDERS = os.environ.get("ENABLE_GPU_PROVIDERS", "1") == "1"
ONNXRT_PROVIDER_PRIORITY = os.environ.get(
    "ONNXRT_PROVIDER_PRIORITY",
    "CUDAExecutionProvider,CPUExecutionProvider",
)

# ---------------------------------------------------------------------------
# ArcFace (InsightFace) configuration
# ---------------------------------------------------------------------------
ARCFACE_MODEL_NAME = os.environ.get("ARCFACE_MODEL_NAME", "buffalo_l")
ARCFACE_DET_SIZE = int(os.environ.get("ARCFACE_DET_SIZE", "320"))

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ENABLE_RBAC = os.environ.get("ENABLE_RBAC", "0") == "1"
ENABLE_RESTX_API = os.environ.get("ENABLE_RESTX_API", "0") == "1"

# ---------------------------------------------------------------------------
# Attendance Session Lifecycle
# ---------------------------------------------------------------------------
ATTENDANCE_SESSION_IDLE_TIMEOUT_SECONDS = int(
    os.environ.get("ATTENDANCE_SESSION_IDLE_TIMEOUT_SECONDS", "900")
)
ATTENDANCE_SESSION_CACHE_SECONDS = float(
    os.environ.get("ATTENDANCE_SESSION_CACHE_SECONDS", "2.0")
)

# ---------------------------------------------------------------------------
# Face Recognition (tuned for cosine similarity with ArcFace)
# ---------------------------------------------------------------------------
RECOGNITION_THRESHOLD = float(
    os.environ.get("RECOGNITION_THRESHOLD", "0.38")
)  # Cosine similarity; higher = stricter match requirement
RECOGNITION_MIN_CONFIDENCE = float(
    os.environ.get("RECOGNITION_MIN_CONFIDENCE", "0.42")
)  # Minimum cosine similarity score to accept (relaxed from 0.46)
RECOGNITION_MIN_DISTANCE_GAP = float(
    os.environ.get("RECOGNITION_MIN_DISTANCE_GAP", "0.08")
)  # Gap between best and 2nd-best must be substantial
RECOGNITION_CONFIDENCE_ALPHA = float(
    os.environ.get("RECOGNITION_CONFIDENCE_ALPHA", "3.0")
)
RECOGNITION_CONFIRM_FRAMES = int(
    os.environ.get("RECOGNITION_CONFIRM_FRAMES", "1")
)  # Require 1 consecutive confident frame for faster detection (still validates via liveness)
RECOGNITION_STABILITY_WINDOW = int(
    os.environ.get("RECOGNITION_STABILITY_WINDOW", "5")
)
RECOGNITION_STABILITY_MIN_HITS = int(
    os.environ.get("RECOGNITION_STABILITY_MIN_HITS", "3")
)
RECOGNITION_TWO_STAGE_ENABLED = os.environ.get(
    "RECOGNITION_TWO_STAGE_ENABLED", "1"
) == "1"
RECOGNITION_STAGE1_TOP_K = int(
    os.environ.get("RECOGNITION_STAGE1_TOP_K", "8")
)
RECOGNITION_STAGE1_MIN_SIMILARITY = float(
    os.environ.get("RECOGNITION_STAGE1_MIN_SIMILARITY", "0.30")
)
RECOGNITION_STAGE1_MARGIN = float(
    os.environ.get("RECOGNITION_STAGE1_MARGIN", "0.08")
)
RECOGNITION_STAGE2_MIN_CANDIDATES = int(
    os.environ.get("RECOGNITION_STAGE2_MIN_CANDIDATES", "2")
)
RECOGNITION_TRACK_CACHE_TTL_SECONDS = float(
    os.environ.get("RECOGNITION_TRACK_CACHE_TTL_SECONDS", "2.0")
)
RECOGNITION_TRACK_CACHE_MAX_ENTRIES = int(
    os.environ.get("RECOGNITION_TRACK_CACHE_MAX_ENTRIES", "1024")
)
FRAME_RESIZE_FACTOR = 0.25    # Resize frames to 1/4 for speed
RECOGNITION_COOLDOWN = 30     # Seconds to skip re-processing a recognized student

# ---------------------------------------------------------------------------
# Detect-Track-Recognize Pipeline
# ---------------------------------------------------------------------------
DETECTION_INTERVAL = 2         # Run face detection every 2nd frame (aggressive optimization)
DETECTION_INTERVAL_MIN = int(os.environ.get("DETECTION_INTERVAL_MIN", "2"))
DETECTION_INTERVAL_MAX = int(os.environ.get("DETECTION_INTERVAL_MAX", "5"))
TRACK_EXPIRATION_FRAMES = 30   # Remove track after N consecutive failed updates
TRACK_DETECTOR_MISS_TOLERANCE = 2  # Expire track after N detector cycles without support
MOTION_THRESHOLD = 5000        # Min non-zero pixels to consider "motion detected"

# ---------------------------------------------------------------------------
# Liveness / Anti-Spoofing
# ---------------------------------------------------------------------------
LIVENESS_CONFIDENCE_THRESHOLD = float(
    os.environ.get("LIVENESS_CONFIDENCE_THRESHOLD", "0.55")
)
LIVENESS_STRICT_THRESHOLD = float(
    os.environ.get("LIVENESS_STRICT_THRESHOLD", "0.85")
)
LIVENESS_EARLY_REJECT_CONFIDENCE = float(
    os.environ.get("LIVENESS_EARLY_REJECT_CONFIDENCE", "0.5")
)
LIVENESS_REAL_FAST_CONFIDENCE = float(
    os.environ.get("LIVENESS_REAL_FAST_CONFIDENCE", "0.72")
)
LIVENESS_HISTORY_SIZE = int(
    os.environ.get("LIVENESS_HISTORY_SIZE", "5")
)
LIVENESS_MIN_HISTORY = int(
    os.environ.get("LIVENESS_MIN_HISTORY", "1")  # Fast liveness confirmation (1 frame minimum)
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
LIVENESS_SCORE_STD_THRESHOLD = float(
    os.environ.get("LIVENESS_SCORE_STD_THRESHOLD", "0.12")
)
LIVENESS_MIN_FACE_SIZE_PIXELS = int(
    os.environ.get("LIVENESS_MIN_FACE_SIZE_PIXELS", "64")
)
LIVENESS_MIN_FACE_AREA_RATIO = float(
    os.environ.get("LIVENESS_MIN_FACE_AREA_RATIO", "0.01")
)
LIVENESS_DECISION_DELAY_SECONDS = float(
    os.environ.get("LIVENESS_DECISION_DELAY_SECONDS", "2.5")
)
LIVENESS_SPOOF_COOLDOWN_SECONDS = float(
    os.environ.get("LIVENESS_SPOOF_COOLDOWN_SECONDS", "2.5")
)
LIVENESS_MAX_VERIFICATION_TIMEOUT_SECONDS = float(
    os.environ.get("LIVENESS_MAX_VERIFICATION_TIMEOUT_SECONDS", "5.5")
)
LIVENESS_FACE_MOTION_MIN_PIXELS = float(
    os.environ.get("LIVENESS_FACE_MOTION_MIN_PIXELS", "8.0")
)
LIVENESS_SCREEN_LAPLACIAN_MIN = float(
    os.environ.get("LIVENESS_SCREEN_LAPLACIAN_MIN", "35.0")
)
LIVENESS_SCREEN_CONTRAST_MIN = float(
    os.environ.get("LIVENESS_SCREEN_CONTRAST_MIN", "20.0")
)
LIVENESS_SCREEN_HIGHLIGHT_RATIO_MAX = float(
    os.environ.get("LIVENESS_SCREEN_HIGHLIGHT_RATIO_MAX", "0.08")
)
LIVENESS_SCREEN_BRIGHTNESS_MIN = float(
    os.environ.get("LIVENESS_SCREEN_BRIGHTNESS_MIN", "170.0")
)
LIVENESS_WEIGHTED_DECISION_ENABLED = os.environ.get(
    "LIVENESS_WEIGHTED_DECISION_ENABLED", "0"
) == "1"
LIVENESS_WEIGHTED_ACCEPT_THRESHOLD = float(
    os.environ.get("LIVENESS_WEIGHTED_ACCEPT_THRESHOLD", "0.72")
)
LIVENESS_NO_ENCODE_MARGIN = float(
    os.environ.get("LIVENESS_NO_ENCODE_MARGIN", "0.03")
)
SPOOF_HOLD_SECONDS = float(
    os.environ.get("SPOOF_HOLD_SECONDS", str(LIVENESS_SPOOF_COOLDOWN_SECONDS))
)
TRACK_STATE_PENDING_SECONDS = float(
    os.environ.get("TRACK_STATE_PENDING_SECONDS", str(LIVENESS_DECISION_DELAY_SECONDS))
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
BLUR_THRESHOLD = float(
    os.environ.get("BLUR_THRESHOLD", "6.0")
)  # Laplacian variance; below = too blurry (relaxed with CLAHE preprocessing)
BRIGHTNESS_THRESHOLD = 40.0        # Mean pixel brightness; below = too dark
MIN_FACE_SIZE_PIXELS = int(os.environ.get("MIN_FACE_SIZE_PIXELS", "36"))
MIN_FACE_AREA_RATIO = float(os.environ.get("MIN_FACE_AREA_RATIO", "0.005"))
MAX_REGISTRATION_IMAGES = 5        # Max images accepted per registration

# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------
UPLOAD_MAX_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ---------------------------------------------------------------------------
# Frame Processing
# ---------------------------------------------------------------------------
FRAME_PROCESS_WIDTH = 512          # Resize frames to this width before processing (reduced from 640 for 20% speedup)

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
MJPEG_TARGET_FPS = int(os.environ.get("MJPEG_TARGET_FPS", "15"))  # Reduced from 24 for faster processing

# ---------------------------------------------------------------------------
# Bounded Buffers
# ---------------------------------------------------------------------------
EVENT_BUFFER_MAX = 50              # Max attendance events in deque
LOG_BUFFER_MAX = 200               # Max log entries in deque

# ---------------------------------------------------------------------------
# YuNet Face Detector
# ---------------------------------------------------------------------------
YUNET_SCORE_THRESHOLD = float(
    os.environ.get("YUNET_SCORE_THRESHOLD", "0.62")
)  # Detection confidence threshold (lower = more detections)
YUNET_NMS_THRESHOLD = float(
    os.environ.get("YUNET_NMS_THRESHOLD", "0.23")
)  # Non-maximum suppression threshold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    os.environ.get("OCCLUDED_RECOGNITION_THRESHOLD_DELTA", "0.03")
)  # Only relax occluded faces slightly (mask/cap reduces quality)
OCCLUDED_MIN_DISTANCE_GAP = float(
    os.environ.get("OCCLUDED_MIN_DISTANCE_GAP", "0.10")
)  # Require larger gap when face is occluded
OCCLUDED_MIN_CONFIDENCE = float(
    os.environ.get("OCCLUDED_MIN_CONFIDENCE", "0.5")
)
OCCLUDED_DISABLE_INCREMENTAL_LEARNING = (
    os.environ.get("OCCLUDED_DISABLE_INCREMENTAL_LEARNING", "1") == "1"
)


# ---------------------------------------------------------------------------
# Advanced Anti-Spoofing Features
# ---------------------------------------------------------------------------
ENABLE_ADVANCED_LIVENESS = os.environ.get("ENABLE_ADVANCED_LIVENESS", "1") == "1"
ENABLE_CHALLENGE_RESPONSE = os.environ.get("ENABLE_CHALLENGE_RESPONSE", "0") == "1"
ENABLE_TEXTURE_ANALYSIS = os.environ.get("ENABLE_TEXTURE_ANALYSIS", "1") == "1"

# Liveness signal fusion weights
LIVENESS_FUSION_WEIGHT_CNN = float(
    os.environ.get("LIVENESS_FUSION_WEIGHT_CNN", "0.4")
)
LIVENESS_FUSION_WEIGHT_BLINK = float(
    os.environ.get("LIVENESS_FUSION_WEIGHT_BLINK", "0.2")
)
LIVENESS_FUSION_WEIGHT_MOTION = float(
    os.environ.get("LIVENESS_FUSION_WEIGHT_MOTION", "0.2")
)
LIVENESS_FUSION_WEIGHT_TEXTURE = float(
    os.environ.get("LIVENESS_FUSION_WEIGHT_TEXTURE", "0.15")
)
LIVENESS_FUSION_WEIGHT_CHALLENGE = float(
    os.environ.get("LIVENESS_FUSION_WEIGHT_CHALLENGE", "0.05")
)

# Texture analysis thresholds
TEXTURE_FLATNESS_THRESHOLD = float(
    os.environ.get("TEXTURE_FLATNESS_THRESHOLD", "0.7")
)
TEXTURE_LBP_RADIUS = int(
    os.environ.get("TEXTURE_LBP_RADIUS", "1")
)
TEXTURE_LBP_POINTS = int(
    os.environ.get("TEXTURE_LBP_POINTS", "8")
)

# Challenge-response settings
CHALLENGE_RESPONSE_TIMEOUT_SECONDS = float(
    os.environ.get("CHALLENGE_RESPONSE_TIMEOUT_SECONDS", "10.0")
)
CHALLENGE_BLINK_EAR_THRESHOLD = float(
    os.environ.get("CHALLENGE_BLINK_EAR_THRESHOLD", "0.21")
)
CHALLENGE_SMILE_MOUTH_THRESHOLD = float(
    os.environ.get("CHALLENGE_SMILE_MOUTH_THRESHOLD", "0.3")
)
CHALLENGE_MOVE_MOTION_THRESHOLD = float(
    os.environ.get("CHALLENGE_MOVE_MOTION_THRESHOLD", "8.0")
)

# ---------------------------------------------------------------------------
# Vector Search (Face Matching Acceleration)
# ---------------------------------------------------------------------------
ENABLE_VECTOR_SEARCH = os.environ.get("ENABLE_VECTOR_SEARCH", "1") == "1"
VECTOR_SEARCH_BACKEND = os.environ.get(
    "VECTOR_SEARCH_BACKEND", "faiss"
)  # Options: "faiss", "mongodb_atlas", "hybrid"
FAISS_INDEX_TYPE = os.environ.get(
    "FAISS_INDEX_TYPE", "IVFFlat"
)  # Options: "Flat", "IVFFlat", "HNSW"
FAISS_INDEX_NLIST = int(os.environ.get("FAISS_INDEX_NLIST", "50"))
FAISS_INDEX_NPROBE = int(os.environ.get("FAISS_INDEX_NPROBE", "10"))
FAISS_INDEX_BATCH_SIZE = int(
    os.environ.get("FAISS_INDEX_BATCH_SIZE", "100")
)
VECTOR_SEARCH_K = int(os.environ.get("VECTOR_SEARCH_K", "5"))
VECTOR_SEARCH_INDEX_PATH = os.environ.get(
    "VECTOR_SEARCH_INDEX_PATH",
    None,  # Will be set to {BASE_DIR}/data/faiss_index.bin after BASE_DIR is defined
)

# ---------------------------------------------------------------------------
# Dashboard Analytics
# ---------------------------------------------------------------------------
ENABLE_ANALYTICS = os.environ.get("ENABLE_ANALYTICS", "1") == "1"
ANALYTICS_CACHE_SECONDS = float(
    os.environ.get("ANALYTICS_CACHE_SECONDS", "300.0")
)
ANALYTICS_AGGREGATION_MAX_PIPELINE_LENGTH = int(
    os.environ.get("ANALYTICS_AGGREGATION_MAX_PIPELINE_LENGTH", "50")
)
LATE_ARRIVAL_CUTOFF_TIME = os.environ.get(
    "LATE_ARRIVAL_CUTOFF_TIME", "09:00:00"
)  # Time after which arrival is marked as "late"


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

# Camera reconnection with exponential backoff
CAMERA_RECONNECT_INITIAL_DELAY_SECONDS = float(
    os.environ.get("CAMERA_RECONNECT_INITIAL_DELAY_SECONDS", "1.0")
)
CAMERA_RECONNECT_MAX_DELAY_SECONDS = float(
    os.environ.get("CAMERA_RECONNECT_MAX_DELAY_SECONDS", "30.0")
)
CAMERA_RECONNECT_MAX_ATTEMPTS = int(
    os.environ.get("CAMERA_RECONNECT_MAX_ATTEMPTS", "12")
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
STUDENT_SAMPLE_DIR = os.path.join(UPLOAD_DIR, "student_app")

# ---------------------------------------------------------------------------
# Student Portal App
# ---------------------------------------------------------------------------
STUDENT_APP_HOST = os.environ.get("STUDENT_APP_HOST", "0.0.0.0")
STUDENT_APP_PORT = int(os.environ.get("STUDENT_APP_PORT", "5001"))
STUDENT_MIN_CAPTURE_IMAGES = int(os.environ.get("STUDENT_MIN_CAPTURE_IMAGES", "3"))
STUDENT_MAX_CAPTURE_IMAGES = int(os.environ.get("STUDENT_MAX_CAPTURE_IMAGES", "5"))
STUDENT_AUTO_APPROVE_SCORE = float(os.environ.get("STUDENT_AUTO_APPROVE_SCORE", "85"))
STUDENT_PENDING_SCORE = float(os.environ.get("STUDENT_PENDING_SCORE", "60"))
STUDENT_PORTAL_BASE_URL = os.environ.get(
    "STUDENT_PORTAL_BASE_URL",
    f"http://localhost:{STUDENT_APP_PORT}/student",
).rstrip("/")

# Student portal session cookies (for Flask)
# SECURITY: These settings enforce secure session handling
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "1") == "1"  # Enforce HTTPS in production
SESSION_COOKIE_HTTPONLY = os.environ.get("SESSION_COOKIE_HTTPONLY", "1") == "1"  # Prevent JS access (CSRF protection)
SESSION_COOKIE_SAMESITE = os.environ.get("SESSION_COOKIE_SAMESITE", "Strict")  # Prevent cross-site requests

# Session timeout configuration (in seconds)
PERMANENT_SESSION_LIFETIME = int(os.environ.get("PERMANENT_SESSION_LIFETIME", "1800"))  # 30 minutes for students
ADMIN_SESSION_LIFETIME = int(os.environ.get("ADMIN_SESSION_LIFETIME", "3600"))  # 60 minutes for admins

# ---------------------------------------------------------------------------
# SocketIO / CORS
# ---------------------------------------------------------------------------
SOCKETIO_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "SOCKETIO_CORS_ORIGINS",
        "http://localhost:*,http://127.0.0.1:*,http://192.168.*",
    ).split(",")
    if origin.strip()
]
# In development mode, allow all origins (be more permissive)
if os.environ.get("APP_DEBUG", "0") == "1":
    SOCKETIO_CORS_ORIGINS = "*"

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

# ---------------------------------------------------------------------------
# Dynamic Recognition Threshold (image-quality adaptive)
# ---------------------------------------------------------------------------
DYNAMIC_THRESHOLD_ENABLED = os.environ.get("DYNAMIC_THRESHOLD_ENABLED", "1") == "1"
DYNAMIC_THRESHOLD_BLUR_PENALTY = float(
    os.environ.get("DYNAMIC_THRESHOLD_BLUR_PENALTY", "0.05")
)  # Reduce threshold when face region is blurry
DYNAMIC_THRESHOLD_DARK_PENALTY = float(
    os.environ.get("DYNAMIC_THRESHOLD_DARK_PENALTY", "0.03")
)  # Reduce threshold when face region is dark
DYNAMIC_THRESHOLD_LOW_CONTRAST_PENALTY = float(
    os.environ.get("DYNAMIC_THRESHOLD_LOW_CONTRAST_PENALTY", "0.03")
)  # Reduce threshold when face region has low contrast
DYNAMIC_THRESHOLD_MAX_PENALTY = float(
    os.environ.get("DYNAMIC_THRESHOLD_MAX_PENALTY", "0.10")
)

# ---------------------------------------------------------------------------
# Multi-Frame Embedding Smoothing
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW = int(os.environ.get("SMOOTHING_WINDOW", "5"))
SMOOTHING_MIN_FRAMES = int(os.environ.get("SMOOTHING_MIN_FRAMES", "2"))

# ---------------------------------------------------------------------------
# Blink Detection (Anti-Spoofing supplement)
# ---------------------------------------------------------------------------
BLINK_DETECTION_ENABLED = os.environ.get("BLINK_DETECTION_ENABLED", "1") == "1"
BLINK_LANDMARK_REFRESH_SECONDS = float(
    os.environ.get("BLINK_LANDMARK_REFRESH_SECONDS", "5.0")
)
BLINK_EAR_THRESHOLD = float(
    os.environ.get("BLINK_EAR_THRESHOLD", "0.21")
)  # Eye Aspect Ratio below this = eye closed
BLINK_CONSEC_FRAMES = int(
    os.environ.get("BLINK_CONSEC_FRAMES", "2")
)  # Consecutive low-EAR frames to register a blink
BLINK_REQUIRED_COUNT = int(
    os.environ.get("BLINK_REQUIRED_COUNT", "1")
)  # Minimum blinks expected within a track lifetime for liveness bonus

# ---------------------------------------------------------------------------
# Image Preprocessing (CLAHE)
# ---------------------------------------------------------------------------
PREPROCESSING_CLAHE_ENABLED = os.environ.get("PREPROCESSING_CLAHE_ENABLED", "1") == "1"
PREPROCESSING_CLAHE_CLIP = float(
    os.environ.get("PREPROCESSING_CLAHE_CLIP", "2.0")
)
PREPROCESSING_CLAHE_GRID = int(
    os.environ.get("PREPROCESSING_CLAHE_GRID", "8")
)

# ---------------------------------------------------------------------------
# Performance Tuning (real-time pipeline optimisation)
# ---------------------------------------------------------------------------
PERF_FRAME_SCALE = float(
    os.environ.get("PERF_FRAME_SCALE", "1.0")
)  # Extra scale factor on the processing frame (e.g. 0.75 = 75% of FRAME_PROCESS_WIDTH)

PERF_MAX_FACES = int(
    os.environ.get("PERF_MAX_FACES", "5")
)  # Cap on simultaneous tracked faces; new detections ignored when at limit

PERF_RECOGNITION_INTERVAL = int(
    os.environ.get("PERF_RECOGNITION_INTERVAL", "2")
)  # Re-attempt recognition on existing unidentified tracks every N detection cycles (reduced from 3)

PERF_ANTISPOOF_INTERVAL = int(
    os.environ.get("PERF_ANTISPOOF_INTERVAL", "1")
)  # Re-evaluate liveness on uncertain tracks every N detection cycles (reduced from 3 for faster liveness confirmation)

PERF_RECOGNITION_COOLDOWN_FRAMES = int(
    os.environ.get("PERF_RECOGNITION_COOLDOWN_FRAMES", "30")
)  # After a track is recognised, skip re-recognition for this many frames

PERF_USE_KCF_TRACKER = (
    os.environ.get("PERF_USE_KCF_TRACKER", "0") == "1"
)  # Opt-in to faster KCF tracker (less accurate than default CSRT)

PERF_JPEG_QUALITY = int(
    os.environ.get("PERF_JPEG_QUALITY", "60")  # Reduced from 80 for faster encoding
)  # JPEG encode quality (0-100); lower = faster encode + smaller payload

# PHASE 1: Reliability & Fault Tolerance
# ---------------------------------------------------------------------------

# Frame Timeout Detection
CAMERA_FRAME_TIMEOUT_SECONDS = float(
    os.environ.get("CAMERA_FRAME_TIMEOUT_SECONDS", "5.0")
)  # Force reconnect if no frame received for this duration
CAMERA_FRAME_TIMEOUT_CHECK_INTERVAL_SECONDS = float(
    os.environ.get("CAMERA_FRAME_TIMEOUT_CHECK_INTERVAL_SECONDS", "1.0")
)  # How often to check for frame timeouts

# Frame Queue Overflow & Dropping
FRAME_QUEUE_MAX_DEPTH = int(
    os.environ.get("FRAME_QUEUE_MAX_DEPTH", "50")
)  # Drop frames if pipeline queue depth exceeds this
FRAME_DROP_ENABLED = os.environ.get("FRAME_DROP_ENABLED", "1") == "1"  # Enable frame dropping on overload

# Graceful Degradation Under Load
GRACEFUL_DEGRADATION_ENABLED = os.environ.get(
    "GRACEFUL_DEGRADATION_ENABLED", "1"
) == "1"  # Enable auto-throttling when CPU/memory high
GRACEFUL_DEGRADATION_CPU_THRESHOLD = float(
    os.environ.get("GRACEFUL_DEGRADATION_CPU_THRESHOLD", "80.0")
)  # Disable expensive ops if CPU > this %
GRACEFUL_DEGRADATION_MEMORY_THRESHOLD = float(
    os.environ.get("GRACEFUL_DEGRADATION_MEMORY_THRESHOLD", "85.0")
)  # Disable expensive ops if memory > this %
GRACEFUL_DEGRADATION_DETECTION_INTERVAL_MAX = int(
    os.environ.get("GRACEFUL_DEGRADATION_DETECTION_INTERVAL_MAX", "15")
)  # Max detection interval during degradation
GRACEFUL_DEGRADATION_DISABLE_ANTISPOOF = os.environ.get(
    "GRACEFUL_DEGRADATION_DISABLE_ANTISPOOF", "1"
) == "1"  # Skip anti-spoof when CPU high

# ---------------------------------------------------------------------------
# PHASE 2: Metrics & Observability
# ---------------------------------------------------------------------------

# Slow Frame Logging
SLOW_FRAME_THRESHOLD_MS = float(
    os.environ.get("SLOW_FRAME_THRESHOLD_MS", "200.0")
)  # Threshold for frame processing warning (lowered to encourage optimization)  # Log warning if frame processing exceeds this (milliseconds)

# ---------------------------------------------------------------------------
# PHASE 3: Recognition Confidence & False Positive Control
# ---------------------------------------------------------------------------

# Top-2 Similarity Margin (already configured as RECOGNITION_MIN_DISTANCE_GAP = 0.08)
# Explicitly set margin for top-2 check (higher = stricter, rejects more ambiguous matches)
RECOGNITION_TOP2_SIMILARITY_MARGIN = float(
    os.environ.get("RECOGNITION_TOP2_SIMILARITY_MARGIN", "0.05")
)  # Margin between top match and 2nd-best (cosine similarity)

# Composed Confidence Score Weights
COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT = float(
    os.environ.get("COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT", "0.5")
)  # Weight of recognition score in final confidence
COMPOSED_CONFIDENCE_LIVENESS_WEIGHT = float(
    os.environ.get("COMPOSED_CONFIDENCE_LIVENESS_WEIGHT", "0.3")
)  # Weight of liveness score in final confidence
COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT = float(
    os.environ.get("COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT", "0.2")
)  # Weight of frame consistency score in final confidence

# ---------------------------------------------------------------------------
# Configuration Validation
# ---------------------------------------------------------------------------

def validate_configuration():
    """
    Validate critical configuration parameters for consistency and valid ranges.
    
    Raises:
        ValueError: If any configuration parameter is invalid
        
    Returns:
        dict: Validation results with any warnings
    """
    warnings = []
    errors = []
    
    # Phase 1: Reliability checks
    if CAMERA_FRAME_TIMEOUT_SECONDS < 1.0 or CAMERA_FRAME_TIMEOUT_SECONDS > 30.0:
        errors.append(
            f"CAMERA_FRAME_TIMEOUT_SECONDS must be 1.0-30.0, got {CAMERA_FRAME_TIMEOUT_SECONDS}"
        )
    
    if FRAME_QUEUE_MAX_DEPTH < 10 or FRAME_QUEUE_MAX_DEPTH > 500:
        errors.append(f"FRAME_QUEUE_MAX_DEPTH must be 10-500, got {FRAME_QUEUE_MAX_DEPTH}")
    
    if GRACEFUL_DEGRADATION_CPU_THRESHOLD < 50 or GRACEFUL_DEGRADATION_CPU_THRESHOLD > 95:
        errors.append(
            f"GRACEFUL_DEGRADATION_CPU_THRESHOLD must be 50-95, got {GRACEFUL_DEGRADATION_CPU_THRESHOLD}"
        )
    
    if GRACEFUL_DEGRADATION_MEMORY_THRESHOLD < 60 or GRACEFUL_DEGRADATION_MEMORY_THRESHOLD > 95:
        errors.append(
            f"GRACEFUL_DEGRADATION_MEMORY_THRESHOLD must be 60-95, got {GRACEFUL_DEGRADATION_MEMORY_THRESHOLD}"
        )
    
    # Phase 2: Metrics checks
    if SLOW_FRAME_THRESHOLD_MS < 50 or SLOW_FRAME_THRESHOLD_MS > 500:
        warnings.append(
            f"SLOW_FRAME_THRESHOLD_MS is {SLOW_FRAME_THRESHOLD_MS}ms; "
            "recommend 33ms for 60 FPS or 100ms for 30 FPS"
        )
    
    # Phase 3: Confidence checks
    if RECOGNITION_TOP2_SIMILARITY_MARGIN < 0.01 or RECOGNITION_TOP2_SIMILARITY_MARGIN > 0.20:
        errors.append(
            f"RECOGNITION_TOP2_SIMILARITY_MARGIN must be 0.01-0.20, got {RECOGNITION_TOP2_SIMILARITY_MARGIN}"
        )
    
    # Validate composed confidence weights sum to > 0
    total_weight = (
        COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT
        + COMPOSED_CONFIDENCE_LIVENESS_WEIGHT
        + COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT
    )
    if total_weight <= 0:
        errors.append(
            f"Composed confidence weights must sum > 0, got {total_weight}. "
            f"Recognition={COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT}, "
            f"Liveness={COMPOSED_CONFIDENCE_LIVENESS_WEIGHT}, "
            f"Consistency={COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT}"
        )
    
    if any(w < 0 for w in [
        COMPOSED_CONFIDENCE_RECOGNITION_WEIGHT,
        COMPOSED_CONFIDENCE_LIVENESS_WEIGHT,
        COMPOSED_CONFIDENCE_CONSISTENCY_WEIGHT,
    ]):
        errors.append("Composed confidence weights cannot be negative")
    
    # Raise errors if any found
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  ✗ {e}" for e in errors)
        raise ValueError(error_msg)
    
    return {
        "status": "ok",
        "warnings": warnings,
        "validated_parameters": 10,
    }


# Run validation on module load
try:
    _config_validation = validate_configuration()
except ValueError as e:
    import logging
    logging.warning("Configuration validation warning: %s", str(e))


# ---------------------------------------------------------------------------
# Post-Initialization Config Setup
# ---------------------------------------------------------------------------
# Set VECTOR_SEARCH_INDEX_PATH if not explicitly configured
if VECTOR_SEARCH_INDEX_PATH is None:
    VECTOR_SEARCH_INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")

# Ensure data directory exists
_data_dir = os.path.dirname(VECTOR_SEARCH_INDEX_PATH)
if not os.path.exists(_data_dir):
    os.makedirs(_data_dir, exist_ok=True)
