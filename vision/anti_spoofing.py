"""
Anti-spoofing module - liveness detection using Silent-Face-Anti-Spoofing library.
Models are loaded once at startup and cached globally for fast inference.
"""

import os
import sys
import time
import numpy as np
import cv2

import core.config as config
from core.utils import setup_logging

logger = setup_logging()

# Configuration
SILENT_FACE_PATH = os.path.join(config.BASE_DIR, "Silent-Face-Anti-Spoofing")
MODEL_DIR = os.environ.get(
    "ANTI_SPOOF_MODEL_DIR",
    os.path.join(SILENT_FACE_PATH, "resources", "anti_spoof_models"),
)

# Cached model instances
_predictor = None
_cropper = None
_model_names = []
_parse_model_name = None
_is_ready = False
_initialization_error = None

LIVENESS_LABELS = {
    0: "spoof_or_no_face",
    1: "real",
    2: "other_attack",
    -1: "internal_error",
}


def _setup_library_path():
    """Add Silent-Face-Anti-Spoofing to sys.path."""
    if os.path.isdir(SILENT_FACE_PATH) and SILENT_FACE_PATH not in sys.path:
        sys.path.insert(0, SILENT_FACE_PATH)


def init_models():
    """Load anti-spoofing models into memory. Call once at application startup."""
    global _predictor, _cropper, _model_names, _parse_model_name
    global _is_ready, _initialization_error
    
    _setup_library_path()
    
    try:
        # Import library
        from src.anti_spoof_predict import AntiSpoofPredict
        from src.generate_patches import CropImage
        from src.utility import parse_model_name
        
        # Verify model directory
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(f"Anti-spoof model directory not found: {MODEL_DIR}")
        
        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')])
        if not model_files:
            raise FileNotFoundError(f"No .pth model files found in {MODEL_DIR}")
        
        # Initialize predictor (must be in SILENT_FACE_PATH directory context)
        prev_cwd = os.getcwd()
        try:
            os.chdir(SILENT_FACE_PATH)
            _predictor = AntiSpoofPredict(0)
        finally:
            os.chdir(prev_cwd)
        
        _cropper = CropImage()
        _parse_model_name = parse_model_name
        _model_names = model_files
        
        _is_ready = True
        _initialization_error = None
        logger.info(f"✓ Anti-spoof models loaded from {MODEL_DIR} ({len(model_files)} models)")
        logger.info(f"✓ Silent-Face-Anti-Spoofing library initialized successfully")
        
    except ImportError as exc:
        logger.error(
            f"Silent-Face-Anti-Spoofing library import failed: {exc}. "
            f"Ensure it's installed at {SILENT_FACE_PATH}"
        )
        _is_ready = False
        _initialization_error = str(exc)
        
    except Exception as exc:
        logger.error(
            f"Anti-spoofing initialization failed: {exc}. "
            f"App will continue with all faces marked 'real' (degraded mode)."
        )
        _is_ready = False
        _initialization_error = str(exc)


def is_ready() -> bool:
    """Return whether anti-spoofing models are initialized and ready."""
    return _is_ready and _predictor is not None


def get_initialization_error() -> str | None:
    """Return the initialization error message if initialization failed, else None."""
    return _initialization_error


def _is_face_too_small(
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> bool:
    """Return True when the face crop or bbox is too small for reliable inference."""
    h, w = frame.shape[:2]
    min_pixels = max(1, int(config.LIVENESS_MIN_FACE_SIZE_PIXELS))
    if h < min_pixels or w < min_pixels:
        return True
    if face_bbox is None:
        return False

    _, _, bw, bh = face_bbox
    if bw < min_pixels or bh < min_pixels:
        return True

    frame_area = max(1, h * w)
    bbox_area = max(1, int(bw * bh))
    return (bbox_area / frame_area) < config.LIVENESS_MIN_FACE_AREA_RATIO


def _compute_frame_heuristics(frame: np.ndarray) -> dict:
    """Return cheap quality and screen-spoof heuristics for a face crop."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    highlight_ratio = float(np.mean(gray >= 245))
    return {
        "frame_shape": (h, w),
        "face_area": int(h * w),
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "laplacian_var": laplacian_var,
        "saturation_mean": saturation_mean,
        "highlight_ratio": highlight_ratio,
        "contrast_proxy": brightness_std,
        "screen_suspicious": bool(
            mean_brightness >= config.LIVENESS_SCREEN_BRIGHTNESS_MIN
            and (
                laplacian_var <= config.LIVENESS_SCREEN_LAPLACIAN_MIN
                or brightness_std <= config.LIVENESS_SCREEN_CONTRAST_MIN
                or highlight_ratio >= config.LIVENESS_SCREEN_HIGHLIGHT_RATIO_MAX
            )
        ),
    }


def analyze_liveness_frame(
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> dict:
    """Return a full liveness analysis result and frame heuristics.

    Performs an early size rejection before the model runs so tiny
    distant faces or postage-stamp crops do not waste inference budget.
    """
    heuristics = _compute_frame_heuristics(frame)
    heuristics["too_small"] = _is_face_too_small(frame, face_bbox)

    if heuristics["too_small"]:
        return {
            "label": 0,
            "confidence": 0.0,
            "model_confidence": 0.0,
            "early_reject": True,
            "reject_reason": "face_too_small",
            **heuristics,
        }

    label, confidence = check_liveness(frame, face_bbox=face_bbox)
    return {
        "label": label,
        "confidence": confidence,
        "model_confidence": confidence,
        "early_reject": confidence < config.LIVENESS_EARLY_REJECT_CONFIDENCE,
        "reject_reason": "low_confidence" if confidence < config.LIVENESS_EARLY_REJECT_CONFIDENCE else None,
        **heuristics,
    }


def check_liveness(
    frame: np.ndarray,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[int, float]:
    """Determine whether the face in *frame* is real or spoofed.
    
    The *frame* passed by the camera pipeline is already an adaptive
    face crop (from ``_adaptive_liveness_crop``), so we skip the
    expensive RetinaFace ``get_bbox`` call and synthesize a bbox from
    the crop dimensions instead.
    
    Returns:
        (label, confidence) where:
        - label: 1 = real face, 0 = spoof/no face, -1 = error
        - confidence: float in [0, 1]
    """
    if not _is_ready:
        # Graceful degradation: mark all faces as real
        return 1, 1.0
    
    try:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return 0, 0.0

        bbox = face_bbox
        if bbox is None and _predictor is not None and hasattr(_predictor, "get_bbox"):
            bbox = _predictor.get_bbox(frame)
            if bbox in (None, [], ()):  # Preserve legacy no-face semantics.
                return 0, 0.0

        if _is_face_too_small(frame, bbox):
            logger.debug("Early reject: face crop too small for liveness inference")
            return 0, 0.0
        
        # The frame IS the face crop — synthesize a bbox covering the
        # entire crop with a small margin inset, avoiding the expensive
        # RetinaFace detection that would redundantly re-locate the face.
        if bbox is None:
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            bbox = [margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y]
        
        # Run prediction with all models
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
        
        # Get consensus label
        label = int(np.argmax(prediction))
        confidence = float(prediction[0][label] / len(_model_names))
        if confidence < config.LIVENESS_EARLY_REJECT_CONFIDENCE:
            logger.debug("Early reject: low model confidence %.4f", confidence)
        
        if label == 0:
            logger.debug(f"Spoof detected (confidence={confidence:.4f})")
        
        return label, confidence
        
    except Exception as exc:
        logger.debug(f"Anti-spoofing check failed: {exc}")
        # Graceful degradation
        return 1, 1.0


def compute_ear_from_5point(landmarks_5: np.ndarray) -> float:
    """Compute Eye Aspect Ratio (EAR) from 5-point facial landmarks.
    
    For 5-point landmarks (left_eye, right_eye, nose, mouth_left, mouth_right),
    compute a simplified eye openness metric.
    
    Args:
        landmarks_5: numpy array of shape (5, 2) with [x, y] coordinates
        
    Returns:
        float: Eye Aspect Ratio value (higher = eyes more open)
    """
    try:
        if landmarks_5 is None or len(landmarks_5) < 2:
            return 1.0  # Default to open if invalid
        
        # 5-point landmarks: [left_eye, right_eye, nose, mouth_left, mouth_right]
        left_eye = landmarks_5[0]
        right_eye = landmarks_5[1]
        
        # Calculate vertical and horizontal distances for each eye
        # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        # Simplified: use distance from eye to nose as vertical component
        
        nose = landmarks_5[2]
        
        # Vertical distances: eye to nose
        left_eye_vertical = abs(left_eye[1] - nose[1])
        right_eye_vertical = abs(right_eye[1] - nose[1])
        
        # Horizontal distance between eyes
        horizontal_distance = abs(left_eye[0] - right_eye[0])
        
        if horizontal_distance < 1e-6:
            return 1.0  # Avoid division by zero
        
        # EAR calculation
        ear = (left_eye_vertical + right_eye_vertical) / (2 * max(horizontal_distance, 1e-6))
        
        return float(ear)
        
    except Exception as e:
        logger.debug(f"Error computing EAR: {e}")
        return 1.0  # Default to open if calculation fails


def update_blink_state(
    ear: float,
    ear_history: list,
    blink_count: int,
    blink_frames_below: int,
) -> tuple[list, int, int]:
    """Update blink detection state based on current Eye Aspect Ratio.
    
    Detects blinks by tracking when EAR drops below threshold for consecutive frames.
    
    Args:
        ear: Current Eye Aspect Ratio value
        ear_history: List of recent EAR values
        blink_count: Number of blinks detected so far
        blink_frames_below: Consecutive frames with EAR below threshold
        
    Returns:
        Tuple of (updated_ear_history, updated_blink_count, updated_blink_frames_below)
    """
    try:
        # Get thresholds from config
        ear_threshold = config.BLINK_EAR_THRESHOLD
        consec_frames = config.BLINK_CONSEC_FRAMES
        max_history = 15  # Keep last N EAR values
        
        # Add current EAR to history
        ear_history.append(ear)
        if len(ear_history) > max_history:
            ear_history.pop(0)
        
        # Update blink state
        if ear < ear_threshold:
            blink_frames_below += 1
        else:
            # EAR returned above threshold
            if blink_frames_below >= consec_frames:
                blink_count += 1
                logger.debug(
                    "Blink detected (total=%d, consec_below=%d)",
                    blink_count, blink_frames_below,
                )
            blink_frames_below = 0
        
        return ear_history, blink_count, blink_frames_below
        
    except Exception as e:
        logger.debug(f"Error updating blink state: {e}")
        return ear_history, blink_count, blink_frames_below
