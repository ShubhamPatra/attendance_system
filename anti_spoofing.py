"""
Anti-spoofing module – deep learning liveness detection using
Silent-Face-Anti-Spoofing models.

Models are loaded once at startup via ``init_models()`` and cached globally
so that per-frame inference stays fast (< 250 ms target).
"""

import os
import sys
import time

import numpy as np

import config
from utils import setup_logging

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
    """Load anti-spoofing models into memory.  Call once at application startup."""
    global _predictor, _cropper, _model_names, _parse_model_name

    global _DEVICE

    _setup_library_path()
    torch_mod = _get_torch()
    _DEVICE = "cuda" if torch_mod.cuda.is_available() else "cpu"

    try:
        from src.anti_spoof_predict import AntiSpoofPredict
        from src.generate_patches import CropImage
        from src.utility import parse_model_name
    except ImportError as exc:
        raise ImportError(
            f"Silent-Face-Anti-Spoofing library not found at "
            f"'{SILENT_FACE_PATH}'.  Clone "
            f"https://github.com/minivision-ai/Silent-Face-Anti-Spoofing "
            f"into that path or set the SILENT_FACE_PATH environment "
            f"variable.  Error: {exc}"
        ) from exc

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Anti-spoof model directory not found: {MODEL_DIR}.  "
            f"Set ANTI_SPOOF_MODEL_DIR environment variable."
        )

    # The library's Detection class loads a caffe model using relative paths
    # ("./resources/detection_model/..."), so we must temporarily change the
    # working directory to the library root during construction.
    # init_models() must run during single-threaded startup before worker
    # threads begin, because os.chdir affects the entire process.
    prev_cwd = os.getcwd()
    try:
        os.chdir(SILENT_FACE_PATH)
        _predictor = AntiSpoofPredict(DEVICE_ID)
    finally:
        os.chdir(prev_cwd)

    _cropper = CropImage()
    _parse_model_name = parse_model_name

    _model_names = sorted(
        name for name in os.listdir(MODEL_DIR)
        if not name.startswith(".")
    )

    if not _model_names:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}.")

    logger.info(
        "Anti-spoof models loaded from %s (%d models).",
        MODEL_DIR,
        len(_model_names),
    )
    logger.info("Anti-spoofing using device: %s", _DEVICE)


def check_liveness(frame: np.ndarray) -> tuple[int, float]:
    """Determine whether the face in *frame* is real or spoofed.

    Returns
    -------
    (label, confidence)
        label : 1 = real face, 0 = spoof / no face detected, -1 = internal error
        confidence : float in [0, 1]

    Returns ``(0, 0.0)`` when no face is detected.
    Returns ``(-1, 0.0)`` on internal error.
    The function never raises; errors are logged and a safe default is
    returned so the system does not crash.
    """
    if _predictor is None:
        raise RuntimeError(
            "Anti-spoof models not initialised.  Call init_models() first."
        )

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
            prediction += _predictor.predict(
                img, os.path.join(MODEL_DIR, model_name)
            )

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
