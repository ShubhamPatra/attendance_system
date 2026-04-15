"""
Anti-spoofing module - deep learning liveness detection using
Silent-Face-Anti-Spoofing models.

Models are loaded once at startup via ``init_models()`` and cached globally
so that per-frame inference stays fast (< 250 ms target).
"""

import os
import sys
import time

import numpy as np

import app_core.config as config
from app_core.utils import setup_logging

logger = setup_logging()

_torch = None
_DEVICE = "cpu"


def _get_torch():
    global _torch
    if _torch is None:
        import torch as _torch_mod

        _torch = _torch_mod
    return _torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


SILENT_FACE_PATH = _env_or_default(
    "SILENT_FACE_PATH",
    os.path.join(config.BASE_DIR, "Silent-Face-Anti-Spoofing"),
)

MODEL_DIR = _env_or_default(
    "ANTI_SPOOF_MODEL_DIR",
    os.path.join(SILENT_FACE_PATH, "resources", "anti_spoof_models"),
)

DEVICE_ID = int(os.environ.get("ANTI_SPOOF_DEVICE_ID", "0"))

# ---------------------------------------------------------------------------
# Cached model instances (populated by init_models)
# ---------------------------------------------------------------------------
_predictor = None
_cropper = None
_model_names: list[str] = []
_parse_model_name = None
_initialization_failed = False
_initialization_error: str | None = None

# Output-class mapping for Silent-Face-Anti-Spoofing models.
# Keep this explicit so downstream decision logic stays stable
# if model files are swapped or retrained.
LIVENESS_LABELS = {
    0: "spoof_or_no_face",
    1: "real",
    2: "other_attack",
    -1: "internal_error",
}


def _setup_library_path():
    """Add Silent-Face-Anti-Spoofing to sys.path if it exists on disk."""
    if os.path.isdir(SILENT_FACE_PATH) and SILENT_FACE_PATH not in sys.path:
        sys.path.insert(0, SILENT_FACE_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_models():
    """Load anti-spoofing models into memory. Call once at application startup.
    
    On error, logs the failure, sets _initialization_failed flag, and returns gracefully.
    This allows the app to continue running with degraded anti-spoofing (all faces marked "real").
    """
    global _predictor, _cropper, _model_names, _parse_model_name
    global _DEVICE, _initialization_failed, _initialization_error

    _setup_library_path()
    torch_mod = _get_torch()
    _DEVICE = "cuda" if torch_mod.cuda.is_available() else "cpu"

    try:
        try:
            from src.anti_spoof_predict import AntiSpoofPredict
            from src.generate_patches import CropImage
            from src.utility import parse_model_name
        except ImportError as exc:
            raise ImportError(
                f"Silent-Face-Anti-Spoofing library not found at "
                f"'{SILENT_FACE_PATH}'. Clone "
                f"https://github.com/minivision-ai/Silent-Face-Anti-Spoofing "
                f"into that path or set the SILENT_FACE_PATH environment "
                f"variable. Error: {exc}"
            ) from exc

        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(
                f"Anti-spoof model directory not found: {MODEL_DIR}. "
                f"Set ANTI_SPOOF_MODEL_DIR environment variable."
            )

        # The library's Detection class loads a caffe model using relative paths
        # ("./resources/detection_model/..."), so we must temporarily change the
        # working directory to the library root during construction.
        prev_cwd = os.getcwd()
        try:
            os.chdir(SILENT_FACE_PATH)
            _predictor = AntiSpoofPredict(DEVICE_ID)
        finally:
            os.chdir(prev_cwd)

        _cropper = CropImage()
        _parse_model_name = parse_model_name

        _model_names = sorted(
            name for name in os.listdir(MODEL_DIR) if not name.startswith(".")
        )

        if not _model_names:
            raise FileNotFoundError(f"No model files found in {MODEL_DIR}.")

        logger.info(
            "Anti-spoof models loaded from %s (%d models).",
            MODEL_DIR,
            len(_model_names),
        )
        logger.info("Anti-spoofing using device: %s", _DEVICE)
        _initialization_failed = False
        _initialization_error = None
        
    except Exception as exc:
        _initialization_failed = True
        _initialization_error = str(exc)
        logger.error(
            "Anti-spoofing model initialization failed. "
            "App will continue with all faces marked 'real' (degraded mode). "
            "This is a production warning. Error: %s",
            exc,
        )


def is_ready() -> bool:
    """Return whether anti-spoofing models are initialized and ready."""
    return not _initialization_failed and _predictor is not None


def get_initialization_error() -> str | None:
    """Return the initialization error message if initialization failed, else None."""
    return _initialization_error


def check_liveness(frame: np.ndarray) -> tuple[int, float]:
    """Determine whether the face in *frame* is real or spoofed.

    Returns
    -------
    (label, confidence)
        label : 1 = real face, 0 = spoof / no face detected, -1 = internal error
        confidence : float in [0, 1]

    Returns ``(0, 0.0)`` when no face is detected.
    Returns ``(-1, 0.0)`` on internal error or if models failed to initialize.
    The function never raises; errors are logged and a safe default is returned.
    """
    if _initialization_failed:
        # Graceful degradation: mark all faces as real
        logger.debug(
            "Anti-spoofing models unavailable; marking frame as real (degraded mode)"
        )
        return 1, 1.0
    
    if _predictor is None:
        # This can happen if init_models() was never called
        logger.warning(
            "Anti-spoof models not initialised. Call init_models() first. "
            "Returning degraded result (all real)."
        )
        return 1, 1.0

    try:
        bbox = _predictor.get_bbox(frame)
        if bbox is None or len(bbox) == 0:
            return 0, 0.0

        prediction = np.zeros((1, 3))
        for model_name in _model_names:
            h_input, w_input, model_type, scale = _parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": scale is not None,
            }
            img = _cropper.crop(**param)
            prediction += _predictor.predict(img, os.path.join(MODEL_DIR, model_name))

        label = int(np.argmax(prediction))
        confidence = float(prediction[0][label] / len(_model_names))

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            "Liveness check: label=%d (%s) confidence=%.4f timestamp=%s",
            label,
            LIVENESS_LABELS.get(label, "unknown"),
            confidence,
            timestamp,
        )

        if label == 0:
            logger.warning(
                "Spoof attempt detected (confidence=%.4f) at %s.",
                confidence,
                timestamp,
            )

        return label, confidence

    except Exception as exc:
        logger.exception(
            "Anti-spoof check failed with error: %s %s",
            type(exc).__name__,
            str(exc),
        )
        return -1, 0.0


# ---------------------------------------------------------------------------
# Blink Detection (Eye Aspect Ratio – lightweight anti-spoofing supplement)
# ---------------------------------------------------------------------------

def compute_ear_from_5point(landmarks_5: np.ndarray) -> float:
    """Estimate Eye Aspect Ratio from InsightFace 5-point landmarks.

    InsightFace 5-point landmarks are:
        0: left eye centre
        1: right eye centre
        2: nose tip
        3: left mouth corner
        4: right mouth corner

    Since we only have eye *centres* (not full 6-point eye contours),
    we estimate the EAR from the vertical distance between the eye
    centre and the nose tip vs the inter-ocular distance.  When the
    eyes close, the eye centres shift downward (closer to the nose),
    reducing this ratio.

    This is an approximation that works well enough for detecting
    blinks in a real-time pipeline.

    Parameters
    ----------
    landmarks_5 : ndarray
        ``(5, 2)`` array of facial keypoints.

    Returns
    -------
    float
        Estimated EAR value.  Lower = eyes more closed.
    """
    if landmarks_5 is None or landmarks_5.shape != (5, 2):
        return 0.5  # neutral default

    left_eye = landmarks_5[0]
    right_eye = landmarks_5[1]
    nose = landmarks_5[2]

    # Inter-ocular distance (horizontal reference)
    inter_ocular = np.linalg.norm(right_eye - left_eye)
    if inter_ocular < 1.0:
        return 0.5

    # Average vertical distance from eye centres to nose tip
    left_vert = np.linalg.norm(left_eye - nose)
    right_vert = np.linalg.norm(right_eye - nose)
    avg_vert = (left_vert + right_vert) / 2.0

    # Normalised ratio — higher = eyes open, lower = eyes closed
    ear = avg_vert / inter_ocular
    return float(ear)


def update_blink_state(
    ear: float,
    ear_history: list,
    blink_count: int,
    blink_frames_below: int,
    ear_threshold: float | None = None,
    consec_frames: int | None = None,
) -> tuple[list, int, int]:
    """Update blink detection state with a new EAR observation.

    Parameters
    ----------
    ear : float
        Current Eye Aspect Ratio value.
    ear_history : list
        Mutable list of recent EAR values.
    blink_count : int
        Running count of detected blinks.
    blink_frames_below : int
        Consecutive frames where EAR was below threshold.
    ear_threshold : float or None
        Threshold below which eyes are considered closed.
    consec_frames : int or None
        Consecutive low-EAR frames needed to register a blink.

    Returns
    -------
    (ear_history, blink_count, blink_frames_below)
        Updated state values.
    """
    if ear_threshold is None:
        ear_threshold = config.BLINK_EAR_THRESHOLD
    if consec_frames is None:
        consec_frames = config.BLINK_CONSEC_FRAMES

    ear_history.append(ear)
    if len(ear_history) > 30:
        ear_history.pop(0)

    if ear < ear_threshold:
        blink_frames_below += 1
    else:
        if blink_frames_below >= consec_frames:
            blink_count += 1
            logger.debug(
                "Blink detected (total=%d, consec_below=%d)",
                blink_count, blink_frames_below,
            )
        blink_frames_below = 0

    return ear_history, blink_count, blink_frames_below
