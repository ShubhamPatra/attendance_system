"""
Recognition module – face alignment and encoding for the camera pipeline.

Provides eye-based affine alignment before encoding to improve accuracy.
Registration (single-image) still uses :func:`face_engine.generate_encoding`
directly, which operates on the HOG detector path.
"""

import math

import cv2
import face_recognition
import numpy as np

import app_core.config as config
from app_core.utils import setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Face quality gate
# ---------------------------------------------------------------------------

def check_face_quality_gate(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> tuple[bool, str]:
    """Check blur and brightness of the detected face ROI.

    Returns (True, "") if acceptable, (False, reason) if rejected.
    Uses config.BLUR_THRESHOLD for minimum Laplacian variance and
    config.BRIGHTNESS_THRESHOLD / config.BRIGHTNESS_MAX for brightness range [40, 250].
    """
    x, y, w, h = bbox_xywh
    fh, fw = frame_bgr.shape[:2]
    # Clamp coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False, "Empty face ROI"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Blur check
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < config.BLUR_THRESHOLD:
        return False, f"Face too blurry (sharpness {variance:.0f}, min {config.BLUR_THRESHOLD:.0f})"

    # Brightness check
    brightness = float(np.mean(gray))
    if brightness < config.BRIGHTNESS_THRESHOLD:
        return False, f"Face too dark (brightness {brightness:.0f}, min {config.BRIGHTNESS_THRESHOLD:.0f})"
    if brightness > config.BRIGHTNESS_MAX:
        return False, f"Face too bright (brightness {brightness:.0f}, max {config.BRIGHTNESS_MAX:.0f})"

    return True, ""


# ---------------------------------------------------------------------------
# Face alignment
# ---------------------------------------------------------------------------

def align_face(
    rgb_image: np.ndarray,
    face_location: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Align a face so that the eyes are horizontally level.

    Parameters
    ----------
    rgb_image : ndarray
        Full RGB image (as used by ``face_recognition``).
    face_location : tuple
        ``(top, right, bottom, left)`` face bounding box.

    Returns
    -------
    ndarray or None
        Aligned RGB face crop, or *None* if landmarks cannot be found.
    """
    landmarks_list = face_recognition.face_landmarks(
        rgb_image, face_locations=[face_location],
    )
    if not landmarks_list:
        return None

    landmarks = landmarks_list[0]
    left_eye_pts = landmarks.get("left_eye")
    right_eye_pts = landmarks.get("right_eye")
    if not left_eye_pts or not right_eye_pts:
        return None

    # Centre of each eye
    le = np.mean(left_eye_pts, axis=0)
    re = np.mean(right_eye_pts, axis=0)

    # Angle between eyes
    dy = re[1] - le[1]
    dx = re[0] - le[0]
    angle = math.degrees(math.atan2(dy, dx))

    # Rotate around the midpoint between the eyes
    eye_centre = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)
    h, w = rgb_image.shape[:2]
    M = cv2.getRotationMatrix2D(eye_centre, angle, 1.0)
    aligned = cv2.warpAffine(
        rgb_image, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Crop the face region from the rotated image
    top, right, bottom, left = face_location
    # Add small padding (5 %) to avoid cutting cheeks/chin
    pad_h = int((bottom - top) * 0.05)
    pad_w = int((right - left) * 0.05)
    top = max(0, top - pad_h)
    left = max(0, left - pad_w)
    bottom = min(h, bottom + pad_h)
    right = min(w, right + pad_w)

    crop = aligned[top:bottom, left:right]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop)


# ---------------------------------------------------------------------------
# Camera-path encoding (quality gate → align → encode)
# ---------------------------------------------------------------------------

def encode_face(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Generate a 128-D face encoding for a detected face in a camera frame.

    Performs a quality gate (blur + brightness) check on the face ROI before
    attempting alignment and encoding. Returns None if quality is insufficient.

    Parameters
    ----------
    frame_bgr : ndarray
        Full camera frame (BGR).
    bbox_xywh : tuple
        ``(x, y, w, h)`` of the detected face in the frame.

    Returns
    -------
    ndarray or None
        128-D encoding, or *None* if encoding fails.
    """
    encoding, _ = encode_face_with_reason(frame_bgr, bbox_xywh)
    return encoding


def encode_face_with_reason(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, str]:
    """Generate encoding and return a rejection reason when unavailable."""
    # Quality gate (can be bypassed for diagnosis)
    if config.BYPASS_QUALITY_GATE:
        logger.debug("Quality gate BYPASSED by config flag.")
    else:
        ok, reason = check_face_quality_gate(frame_bgr, bbox_xywh)
        if not ok:
            logger.info("Face quality gate rejected: %s", reason)
            return None, reason

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x, y, w, h = bbox_xywh

    # face_recognition format: (top, right, bottom, left)
    face_loc = (y, x + w, y + h, x)

    # Attempt alignment — this is critical for encoding quality
    aligned_crop = align_face(rgb, face_loc)
    if aligned_crop is not None and aligned_crop.size > 0:
        # Encode the aligned crop (the whole crop is the face)
        h_c, w_c = aligned_crop.shape[:2]
        crop_loc = [(0, w_c, h_c, 0)]
        encodings = face_recognition.face_encodings(aligned_crop, crop_loc)
        if encodings:
            logger.debug(
                "Encoding generated via alignment: shape=%s",
                encodings[0].shape,
            )
            return encodings[0], ""
        # Alignment produced a crop but encoding failed — do NOT fall through
        logger.warning(
            "Aligned crop produced no encoding for bbox=%s; rejecting to avoid low-quality match.",
            bbox_xywh,
        )
        return None, "Face alignment succeeded but encoding generation failed. Please try again."

    # Fallback: encode directly from the full frame WITHOUT alignment only as last resort
    logger.debug("Face alignment unavailable; attempting direct encoding without alignment.")
    encodings = face_recognition.face_encodings(rgb, [face_loc])
    if encodings:
        logger.debug(
            "Encoding generated via fallback (no alignment): shape=%s [LOWER QUALITY]",
            encodings[0].shape,
        )
        return encodings[0], ""

    logger.warning("encode_face: no encoding produced for bbox=%s", bbox_xywh)
    return None, "No facial encoding could be produced. Keep face front-facing and unobstructed."
