"""
Overlay module – drawing bounding boxes, labels, and status text on frames.

Extracted from camera.py to separate UI rendering from pipeline logic.
"""

import time

import cv2
import numpy as np

import app_core.config as config


# ---------------------------------------------------------------------------
# Label background helper
# ---------------------------------------------------------------------------

def draw_label_bg(
    frame: np.ndarray,
    texts: list[tuple[str, tuple[int, int, int]]],
    left: int,
    top_y: int,
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw multiple lines of text with a dark semi-transparent background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 22
    padding = 6

    max_w = 0
    for txt, _ in texts:
        (tw, _), _ = cv2.getTextSize(txt, font, font_scale, thickness)
        max_w = max(max_w, tw)

    box_w = max_w + padding * 2
    box_h = len(texts) * line_height + padding * 2

    x1 = left
    y1 = top_y
    x2 = x1 + box_w
    y2 = y1 + box_h

    fh, fw = frame.shape[:2]
    x1 = max(0, min(x1, fw - 1))
    y1 = max(0, min(y1, fh - 1))
    x2 = max(x1 + 1, min(x2, fw))
    y2 = max(y1 + 1, min(y2, fh))

    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, roi, 0.45, 0, roi)

    for i, (txt, color) in enumerate(texts):
        ty = y1 + padding + (i + 1) * line_height - 4
        cv2.putText(
            frame, txt, (x1 + padding, ty),
            font, font_scale, color, thickness,
        )


# ---------------------------------------------------------------------------
# Per-track overlay
# ---------------------------------------------------------------------------

def draw_track_overlay(
    frame: np.ndarray,
    trk,
    seen_dict: dict,
    seen_lock,
    cooldown: float = config.RECOGNITION_COOLDOWN,
) -> None:
    """Draw bounding box and identity label for one tracked face.

    Parameters
    ----------
    trk : FaceTrack
        The track object.
    seen_dict : dict
        ``{student_id: monotonic_timestamp}`` cooldown map.
    seen_lock : threading.Lock
        Lock protecting *seen_dict*.
    cooldown : float
        Seconds within which a student is considered "already marked".
    """
    top, left, bottom, right = trk.tlbr()
    liveness_label, liveness_conf = trk.liveness
    ppe_state = getattr(trk, "ppe_state", "none")
    ppe_conf = float(getattr(trk, "ppe_confidence", 0.0))

    if trk.is_spoof:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        lines = [
            ("Spoof Detected!", (0, 0, 255)),
            (f"Confidence: {liveness_conf:.2f}", (0, 0, 255)),
        ]
        if config.DEBUG_MODE:
            lines.append((f"Liveness: {liveness_label}/{liveness_conf:.2f}", (0, 200, 200)))
        draw_label_bg(frame, lines, left, bottom + 6)
        return

    if trk.identity is not None:
        student_id, name, confidence = trk.identity
        color = (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        now = time.monotonic()
        with seen_lock:
            last_seen = seen_dict.get(student_id, 0)
        in_cooldown = (now - last_seen < cooldown)
        status = "Recently Marked" if in_cooldown else "ATTENDANCE MARKED"

        lines = [
            (name, (255, 255, 255)),
            (f"Confidence: {confidence:.2f}", (0, 255, 0)),
            (f"Liveness: {liveness_conf:.2f}", (0, 200, 0)),
            (f"PPE: {ppe_state} ({ppe_conf:.2f})", (0, 220, 220)),
            (status, (0, 255, 0) if not in_cooldown else (0, 200, 255)),
        ]
        if config.DEBUG_MODE:
            backend = config.EMBEDDING_BACKEND
            blinks = getattr(trk, "blink_count", 0)
            lines.append((f"Sim: {confidence:.4f} | Thr: {config.RECOGNITION_THRESHOLD:.2f} | {backend}", (0, 200, 200)))
            lines.append((f"Blinks: {blinks} | Frames: {len(getattr(trk, 'embedding_history', []))}", (0, 200, 200)))
        draw_label_bg(frame, lines, left, bottom + 6)
        return

    if trk.is_unknown:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        lines = [
            ("Unknown", (0, 0, 255)),
            (f"Liveness: {liveness_conf:.2f}", (0, 200, 0)),
            (f"PPE: {ppe_state} ({ppe_conf:.2f})", (0, 220, 220)),
        ]
        reason = getattr(trk, "quality_reason", "")
        if reason:
            trimmed = reason if len(reason) <= 52 else reason[:49] + "..."
            lines.append((trimmed, (0, 180, 255)))
        if config.DEBUG_MODE:
            state = getattr(trk, "state", "unknown")
            blinks = getattr(trk, "blink_count", 0)
            frames = len(getattr(trk, "embedding_history", []))
            lines.append((f"State: {state}", (0, 200, 200)))
            lines.append((f"L: {liveness_label}/{liveness_conf:.2f} | Blinks: {blinks}", (0, 200, 200)))
            lines.append((f"Frames: {frames}/{config.SMOOTHING_MIN_FRAMES}", (0, 200, 200)))
        draw_label_bg(frame, lines, left, bottom + 6)
        return

    # Track exists but recognition hasn't run yet (e.g., liveness uncertain)
    cv2.rectangle(frame, (left, top), (right, bottom), (200, 200, 200), 1)
    pending = getattr(trk, "state", "liveness_pending")
    draw_label_bg(frame, [(f"{pending}...", (200, 200, 200))], left, bottom + 6)
