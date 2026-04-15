"""
Recognition module – face alignment and encoding for the camera pipeline.

Supports two alignment strategies:

* **ArcFace 5-point alignment** (preferred): Uses 5 facial keypoints
  (2 eyes · nose · 2 mouth corners) from InsightFace to compute an
  affine warp into the standard 112×112 ArcFace template.
* **Legacy dlib eye-centre alignment**: Uses ``face_recognition``
  68-landmark eye centres to de-rotate the face.  Only active when the
  embedding backend is set to ``dlib``.

Registration (single-image) still uses :func:`face_engine.generate_encoding`
directly, which internally calls the appropriate backend.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import app_core.config as config
from app_core.utils import setup_logging
from app_vision.preprocessing import (
    assess_image_quality,
    preprocess_face,
)

logger = setup_logging()

# ---------------------------------------------------------------------------
# Standard ArcFace 112×112 destination template (5-point)
# ---------------------------------------------------------------------------
# Source: InsightFace reference alignment targets
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


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

    face_w = x2 - x1
    face_h = y2 - y1
    frame_area = max(1, fw * fh)
    face_area = face_w * face_h
    face_area_ratio = face_area / frame_area

    if face_w < config.MIN_FACE_SIZE_PIXELS or face_h < config.MIN_FACE_SIZE_PIXELS:
        return (
            False,
            f"Face too small (size {face_w}x{face_h}, min {config.MIN_FACE_SIZE_PIXELS}px)",
        )
    if face_area_ratio < config.MIN_FACE_AREA_RATIO:
        return (
            False,
            f"Face too small in frame (area ratio {face_area_ratio:.3f}, min {config.MIN_FACE_AREA_RATIO:.3f})",
        )

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
# ArcFace 5-point landmark alignment (preferred)
# ---------------------------------------------------------------------------

def align_face_arcface(
    image_bgr: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: int = 112,
) -> np.ndarray | None:
    """Warp a face to the standard ArcFace 112×112 template using 5
    facial keypoints.

    Parameters
    ----------
    image_bgr : ndarray
        Full image in BGR colour order.
    landmarks_5 : ndarray
        ``(5, 2)`` array of keypoints: left-eye, right-eye, nose,
        left-mouth, right-mouth.
    output_size : int
        Output face chip size (default 112 for ArcFace).

    Returns
    -------
    ndarray or None
        Aligned face chip (BGR, 112×112), or *None* on failure.
    """
    if landmarks_5 is None or landmarks_5.shape != (5, 2):
        return None

    src_pts = landmarks_5.astype(np.float32)
    dst_pts = ARCFACE_DST.copy()

    # Scale destination template if output_size != 112
    if output_size != 112:
        scale = output_size / 112.0
        dst_pts = dst_pts * scale

    # Estimate similarity transform (rotation + uniform scale + translation)
    tform, _ = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.LMEDS,
    )
    if tform is None:
        return None

    aligned = cv2.warpAffine(
        image_bgr,
        tform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if aligned.size == 0:
        return None
    return aligned


# ---------------------------------------------------------------------------
# Legacy dlib eye-centre alignment (fallback)
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
    try:
        import face_recognition
    except ImportError:
        logger.warning("face_recognition not available for legacy alignment.")
        return None

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
# Camera-path encoding (quality gate → preprocess → align → encode)
# ---------------------------------------------------------------------------

def encode_face(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    landmarks_5: np.ndarray | None = None,
) -> np.ndarray | None:
    """Generate a face embedding for a detected face in a camera frame.

    Performs a quality gate (blur + brightness) check on the face ROI before
    attempting alignment and encoding. Returns None if quality is insufficient.

    Parameters
    ----------
    frame_bgr : ndarray
        Full camera frame (BGR).
    bbox_xywh : tuple
        ``(x, y, w, h)`` of the detected face in the frame.
    landmarks_5 : ndarray or None
        Optional ``(5, 2)`` keypoints for ArcFace alignment.

    Returns
    -------
    ndarray or None
        Face embedding vector, or *None* if encoding fails.
    """
    encoding, _ = encode_face_with_reason(frame_bgr, bbox_xywh, landmarks_5)
    return encoding


def encode_face_with_reason(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    landmarks_5: np.ndarray | None = None,
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

    backend_name = config.EMBEDDING_BACKEND

    if backend_name == "arcface":
        return _encode_arcface(frame_bgr, bbox_xywh, landmarks_5)
    else:
        return _encode_dlib(frame_bgr, bbox_xywh)


# ---------------------------------------------------------------------------
# ArcFace encoding path
# ---------------------------------------------------------------------------

def _encode_arcface(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    landmarks_5: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str]:
    """Generate embedding via InsightFace ArcFace model.

    **Fast path** (Strategy 1): When 5-point landmarks are available,
    align the face to 112×112 and run *only* the recognition model.
    This is ~50× faster than running full detection on CPU.

    **Slow path** (Strategy 2): Run InsightFace detection+recognition
    on a padded ROI.  Only used when landmarks are unavailable.
    """
    from app_vision.face_engine import get_embedding_backend, get_arcface_backend

    backend = get_embedding_backend()
    x, y, w, h = bbox_xywh
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    face_roi = frame_bgr[y1:y2, x1:x2]

    if face_roi.size == 0:
        return None, "Empty face ROI for ArcFace encoding."

    # ── Strategy 1 (fast): Pre-aligned 112×112 → recognition model only ──
    if landmarks_5 is not None:
        aligned = align_face_arcface(frame_bgr, landmarks_5)
        if aligned is not None:
            aligned = preprocess_face(aligned)
            try:
                af_backend = get_arcface_backend()
                embedding = af_backend.generate_from_aligned(aligned)
                logger.debug(
                    "ArcFace encoding via FAST aligned path: dim=%d",
                    len(embedding),
                )
                return embedding, ""
            except Exception as exc:
                logger.debug("Fast aligned encoding failed, trying detection: %s", exc)

    # ── Strategy 2: InsightFace detect + align on padded ROI ──
    try:
        pad = int(max(w, h) * 0.25)
        px1 = max(0, x - pad)
        py1 = max(0, y - pad)
        px2 = min(fw, x + w + pad)
        py2 = min(fh, y + h + pad)
        padded_roi = frame_bgr[py1:py2, px1:px2]
        padded_roi = preprocess_face(padded_roi)
        embedding = backend.generate(padded_roi)
        logger.debug(
            "ArcFace encoding via padded ROI detection: dim=%d",
            len(embedding),
        )
        return embedding, ""
    except (ValueError, RuntimeError) as exc:
        logger.debug("ArcFace padded ROI encoding failed: %s", exc)

    # ── Strategy 3: Full-frame fallback (slowest) ──
    try:
        full_preprocessed = preprocess_face(frame_bgr)
        embedding = backend.generate(full_preprocessed)
        logger.debug(
            "ArcFace encoding via full-frame fallback: dim=%d [LOWER QUALITY]",
            len(embedding),
        )
        return embedding, ""
    except (ValueError, RuntimeError) as exc:
        logger.warning("ArcFace full-frame encoding also failed: %s", exc)
        return None, f"ArcFace encoding failed: {exc}"


# ---------------------------------------------------------------------------
# Legacy dlib encoding path
# ---------------------------------------------------------------------------

def _encode_dlib(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, str]:
    """Generate 128-D face encoding via dlib / face_recognition."""
    try:
        import face_recognition
    except ImportError:
        return None, "face_recognition library not available."

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
