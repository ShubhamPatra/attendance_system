"""
Image preprocessing module – lighting normalization, quality assessment,
and dynamic threshold computation.

Provides CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
L-channel of LAB colour space to normalise luminance while preserving
colour fidelity.  Also exposes a quality-assessment helper used by the
recognition pipeline to dynamically adjust matching thresholds.
"""

from __future__ import annotations

import cv2
import numpy as np

import core.config as config
from core.utils import setup_logging

logger = setup_logging()

# Reusable CLAHE object (thread-safe – stateless after creation)
_clahe: cv2.CLAHE | None = None


def _get_clahe() -> cv2.CLAHE:
    """Return a cached CLAHE instance configured from application settings."""
    global _clahe
    if _clahe is None:
        grid = max(1, config.PREPROCESSING_CLAHE_GRID)
        _clahe = cv2.createCLAHE(
            clipLimit=config.PREPROCESSING_CLAHE_CLIP,
            tileGridSize=(grid, grid),
        )
    return _clahe


# ---------------------------------------------------------------------------
# CLAHE lighting normalisation
# ---------------------------------------------------------------------------

def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE histogram equalisation on the luminance channel.

    Converts to LAB, equalises L, converts back to BGR.  If CLAHE is
    disabled via config the image is returned unchanged.

    Parameters
    ----------
    face_bgr : ndarray
        Face crop in BGR colour order.

    Returns
    -------
    ndarray
        Preprocessed BGR image (same shape and dtype as input).
    """
    if not config.PREPROCESSING_CLAHE_ENABLED:
        return face_bgr

    if face_bgr is None or face_bgr.size == 0:
        return face_bgr

    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _get_clahe().apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Image quality assessment
# ---------------------------------------------------------------------------

def assess_image_quality(face_bgr: np.ndarray) -> dict:
    """Compute quality metrics for a face crop.

    Returns
    -------
    dict
        ``blur_score``  : float – Laplacian variance (higher = sharper).
        ``brightness``  : float – Mean intensity of the grayscale image.
        ``contrast``    : float – Std-dev of the grayscale image.
        ``is_blurry``   : bool  – True when below ``BLUR_THRESHOLD``.
        ``is_dark``     : bool  – True when below ``BRIGHTNESS_THRESHOLD``.
        ``is_bright``   : bool  – True when above ``BRIGHTNESS_MAX``.
        ``is_low_contrast`` : bool – True when contrast < 35.
    """
    if face_bgr is None or face_bgr.size == 0:
        return {
            "blur_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "is_blurry": True,
            "is_dark": True,
            "is_bright": False,
            "is_low_contrast": True,
        }

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "is_blurry": blur_score < config.BLUR_THRESHOLD,
        "is_dark": brightness < config.BRIGHTNESS_THRESHOLD,
        "is_bright": brightness > config.BRIGHTNESS_MAX,
        "is_low_contrast": contrast < 35.0,
    }


# ---------------------------------------------------------------------------
# Dynamic threshold adjustment
# ---------------------------------------------------------------------------

def compute_dynamic_threshold(
    base_threshold: float,
    quality: dict,
) -> float:
    """Adjust recognition threshold based on image quality.

    For cosine-similarity matching the threshold represents the *minimum*
    similarity required.  Poor-quality images (blurry, dark, low contrast)
    cause the threshold to be **lowered** — i.e. we become slightly more
    lenient, acknowledging that embeddings from degraded images are noisier.

    The combined penalty is capped at 0.10 to prevent the threshold from
    dropping too far and admitting false positives.

    Parameters
    ----------
    base_threshold : float
        The configured ``RECOGNITION_THRESHOLD``.
    quality : dict
        Output of :func:`assess_image_quality`.

    Returns
    -------
    float
        The adjusted threshold, never less than ``base_threshold - 0.10``.
    """
    if not config.DYNAMIC_THRESHOLD_ENABLED:
        return base_threshold

    penalty = 0.0

    if quality.get("is_blurry", False):
        penalty += config.DYNAMIC_THRESHOLD_BLUR_PENALTY

    if quality.get("is_dark", False):
        penalty += config.DYNAMIC_THRESHOLD_DARK_PENALTY

    if quality.get("is_low_contrast", False):
        penalty += config.DYNAMIC_THRESHOLD_LOW_CONTRAST_PENALTY

    # Cap the total penalty
    max_penalty = max(0.0, float(config.DYNAMIC_THRESHOLD_MAX_PENALTY))
    penalty = min(penalty, max_penalty)

    adjusted = base_threshold - penalty
    if penalty > 0:
        logger.debug(
            "Dynamic threshold: base=%.3f penalty=%.3f adjusted=%.3f "
            "blur=%.1f bright=%.1f contrast=%.1f",
            base_threshold, penalty, adjusted,
            quality.get("blur_score", 0),
            quality.get("brightness", 0),
            quality.get("contrast", 0),
        )
    return adjusted
