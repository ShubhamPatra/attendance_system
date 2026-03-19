"""
Pipeline module – face detection (YuNet), tracking, motion detection,
and track association logic.

Extracted from camera.py to separate detection/tracking concerns from
camera capture and event management.
"""

import math

import cv2
import numpy as np

import config
from utils import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# Module-level YuNet detector (loaded once at startup)
# ---------------------------------------------------------------------------
_yunet_detector: cv2.FaceDetectorYN | None = None


def init_yunet(model_path: str, input_width: int = 640) -> None:
    """Load the YuNet face detector from an ONNX file.

    Must be called once at application startup (e.g. in ``create_app``).
    The detector is stored in a module-level variable so subsequent calls
    to :func:`detect_faces_yunet` reuse the same instance.
    """
    global _yunet_detector
    # Input size will be adjusted per-frame, but we initialise with a
    # reasonable default so the object is ready.
    _yunet_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (input_width, input_width),
        config.YUNET_SCORE_THRESHOLD,
        config.YUNET_NMS_THRESHOLD,
    )
    logger.info("YuNet face detector loaded from %s", model_path)


# ---------------------------------------------------------------------------
# OpenCV tracker factory (handles API differences across versions)
# ---------------------------------------------------------------------------

def create_tracker():
    """Return a new object tracker from the best available OpenCV backend.

    Priority: CSRT (contrib) > MIL (built-in).
    """
    for factory in (
        lambda: cv2.TrackerCSRT.create(),
        lambda: cv2.legacy.TrackerCSRT.create(),
    ):
        try:
            return factory()
        except (AttributeError, cv2.error):
            pass
    return cv2.TrackerMIL.create()


# ---------------------------------------------------------------------------
# FaceTrack – one tracked face with cached identity
# ---------------------------------------------------------------------------

class FaceTrack:
    """Lightweight object that pairs an OpenCV tracker with a recognition
    result so that identity is computed once and reused across frames."""

    __slots__ = (
        "track_id", "tracker", "bbox",
        "identity", "liveness", "is_spoof", "is_unknown",
        "frames_missing",
    )

    def __init__(self, track_id: int, frame: np.ndarray,
                 bbox_xywh: tuple[int, int, int, int]):
        self.track_id = track_id
        self.tracker = create_tracker()
        self.bbox = bbox_xywh                       # (x, y, w, h)
        self.tracker.init(frame, self.bbox)
        self.identity: tuple | None = None          # (student_id, name, confidence)
        self.liveness: tuple[int, float] = (0, 0.0)
        self.is_spoof: bool = False
        self.is_unknown: bool = False
        self.frames_missing: int = 0

    def update(self, frame: np.ndarray) -> bool:
        """Advance the tracker by one frame.  Returns *True* on success."""
        ok, box = self.tracker.update(frame)
        if ok:
            self.bbox = tuple(int(v) for v in box)
            self.frames_missing = 0
        else:
            self.frames_missing += 1
        return ok

    def tlbr(self) -> tuple[int, int, int, int]:
        """Return bounding box as (top, left, bottom, right)."""
        x, y, w, h = self.bbox
        return y, x, y + h, x + w

    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou(box_a: tuple, box_b: tuple) -> float:
    """Compute Intersection-over-Union between two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def centroid_distance(box_a: tuple, box_b: tuple) -> float:
    """Euclidean distance between centres of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    cx_a = ax + aw / 2
    cy_a = ay + ah / 2
    cx_b = bx + bw / 2
    cy_b = by + bh / 2
    return math.hypot(cx_a - cx_b, cy_a - cy_b)


# ---------------------------------------------------------------------------
# Motion detection (pure function)
# ---------------------------------------------------------------------------

def detect_motion(
    prev_gray: np.ndarray | None,
    frame: np.ndarray,
    threshold: int = config.MOTION_THRESHOLD,
) -> tuple[bool, np.ndarray]:
    """Return ``(motion_detected, current_gray)`` for the given *frame*.

    *prev_gray* is the blurred grayscale image from the previous call
    (or ``None`` on the first invocation, which always returns True).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_gray is None:
        return True, gray
    delta = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    changed = cv2.countNonZero(thresh)
    return changed >= threshold, gray


# ---------------------------------------------------------------------------
# YuNet face detection
# ---------------------------------------------------------------------------

def detect_faces_yunet(
    frame: np.ndarray,
    process_width: int = config.FRAME_PROCESS_WIDTH,
) -> list[tuple[int, int, int, int]]:
    """Run YuNet on *frame* and return ``(x, y, w, h)`` boxes in
    *frame*-coordinate space.

    The frame is internally resized to *process_width* for detection,
    and results are scaled back.
    """
    if _yunet_detector is None:
        raise RuntimeError(
            "YuNet detector not loaded.  Call pipeline.init_yunet() first."
        )

    h_orig, w_orig = frame.shape[:2]
    scale = process_width / w_orig
    new_w = process_width
    new_h = int(h_orig * scale)
    small = cv2.resize(frame, (new_w, new_h))

    _yunet_detector.setInputSize((new_w, new_h))
    _, detections = _yunet_detector.detect(small)

    boxes: list[tuple[int, int, int, int]] = []
    if detections is None:
        return boxes

    inv_scale = 1.0 / scale
    for det in detections:
        x = int(det[0] * inv_scale)
        y = int(det[1] * inv_scale)
        w = int(det[2] * inv_scale)
        h = int(det[3] * inv_scale)
        # Clamp to frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_orig - x)
        h = min(h, h_orig - y)
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))

    return boxes


# ---------------------------------------------------------------------------
# Detection + track association
# ---------------------------------------------------------------------------

def detect_and_associate(
    frame: np.ndarray,
    tracks: list[FaceTrack],
    process_width: int = config.FRAME_PROCESS_WIDTH,
    iou_threshold: float = config.IOU_THRESHOLD,
    centroid_dist_threshold: float = config.CENTROID_DISTANCE_THRESHOLD,
) -> list[tuple[int, int, int, int]]:
    """Run face detection and return bounding boxes for **new** faces
    that do not match any existing track.

    A detection matches a track when **either**:
    * ``IoU > iou_threshold``
    * ``centroid distance < centroid_dist_threshold``

    Matched tracks have their ``frames_missing`` counter reset.
    Returns ``(x, y, w, h)`` boxes in original-frame coordinates.
    """
    det_boxes = detect_faces_yunet(frame, process_width)
    new_boxes: list[tuple[int, int, int, int]] = []

    for det_box in det_boxes:
        matched = False
        # Check against existing tracks
        for trk in tracks:
            if (iou(det_box, trk.bbox) > iou_threshold
                    or centroid_distance(det_box, trk.bbox) < centroid_dist_threshold):
                trk.frames_missing = 0
                matched = True
                break
        # Check against already-accepted new boxes (avoid duplicates)
        if not matched:
            for accepted in new_boxes:
                if (iou(det_box, accepted) > iou_threshold
                        or centroid_distance(det_box, accepted) < centroid_dist_threshold):
                    matched = True
                    break
        if not matched:
            new_boxes.append(det_box)

    return new_boxes
