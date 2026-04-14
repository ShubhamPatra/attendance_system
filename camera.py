"""
Camera module -- threaded webcam capture and Detect-Track-Recognize pipeline.

Architecture
------------
* **Every frame**: update lightweight OpenCV CSRT trackers and draw overlays
  from previously stored identities (~5-20 ms per tracked face).
* **Every DETECTION_INTERVAL frames (with motion) or every
  NO_MOTION_DETECTION_INTERVAL frames (without motion)**: run YuNet face
  detection to discover *new* faces.
* **Only for new tracks**: run anti-spoof + aligned encoding + recognition
  once, then store the result inside the track so it is never recomputed.

Detection, tracking, and overlay responsibilities are delegated to
``pipeline``, ``recognition``, and ``overlay`` helper modules.

Multi-camera support: multiple Camera instances can run concurrently
(one per device index).
"""

import collections
import os
import threading
import time

import cv2
import numpy as np

import config
import database
from overlay import draw_track_overlay
from performance import tracker
from pipeline import FaceTrack, centroid_distance, detect_and_associate_detailed, detect_motion, iou
from utils import setup_logging

logger = setup_logging()

# -- SocketIO reference (set by app.py after init) --------------------------
_socketio = None
_lazy_face_engine = None
_lazy_recognition = None
_lazy_anti_spoofing = None
_lazy_ppe_detection = None


def _face_engine_module():
    global _lazy_face_engine
    if _lazy_face_engine is None:
        import face_engine as _mod
        _lazy_face_engine = _mod
    return _lazy_face_engine


def _recognition_module():
    global _lazy_recognition
    if _lazy_recognition is None:
        import recognition as _mod
        _lazy_recognition = _mod
    return _lazy_recognition


def _anti_spoofing_module():
    global _lazy_anti_spoofing
    if _lazy_anti_spoofing is None:
        import anti_spoofing as _mod
        _lazy_anti_spoofing = _mod
    return _lazy_anti_spoofing


def _ppe_detection_module():
    global _lazy_ppe_detection
    if _lazy_ppe_detection is None:
        import ppe_detection as _mod
        _lazy_ppe_detection = _mod
    return _lazy_ppe_detection


def _encoding_cache():
    return _face_engine_module().encoding_cache


def check_liveness(*args, **kwargs):
    """Compatibility wrapper for tests and monkeypatching."""
    return _anti_spoofing_module().check_liveness(*args, **kwargs)


def encode_face(*args, **kwargs):
    """Compatibility wrapper for tests and monkeypatching."""
    return _recognition_module().encode_face(*args, **kwargs)


def detect_ppe(*args, **kwargs):
    """Compatibility wrapper for tests and monkeypatching."""
    return _ppe_detection_module().detect_ppe(*args, **kwargs)


def set_socketio(sio):
    """Store a reference to the Flask-SocketIO instance for emitting events."""
    global _socketio
    _socketio = sio


def _emit_event(event_name: str, data: dict):
    """Emit a SocketIO event if available (non-blocking, fire-and-forget)."""
    if _socketio is not None:
        try:
            _socketio.emit(event_name, data, namespace="/")
        except Exception as exc:
            logger.debug("SocketIO emit failed for %s: %s", event_name, exc)


def _resize_to_process_width(frame: np.ndarray) -> np.ndarray:
    """Resize *frame* to :data:`config.FRAME_PROCESS_WIDTH` maintaining
    aspect ratio.  Returns the original frame unchanged if it is already
    at or below the target width."""
    h, w = frame.shape[:2]
    if w <= config.FRAME_PROCESS_WIDTH:
        return frame
    scale = config.FRAME_PROCESS_WIDTH / w
    new_w = config.FRAME_PROCESS_WIDTH
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


class Camera:
    """Thread-safe webcam capture with face-recognition pipeline."""

    def __init__(self, source: int = 0):
        self._cap = cv2.VideoCapture(source)
        self._source = source
        self._frame = None
        self._frame_fresh = False
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

        # Latest attendance events (for UI overlay / SSE -- consumed on read)
        self._events: collections.deque[dict] = collections.deque(
            maxlen=config.EVENT_BUFFER_MAX,
        )
        self._events_lock = threading.Lock()

        # Persistent log buffer (kept for /logs page -- NOT cleared on read)
        self._log_buffer: collections.deque[dict] = collections.deque(
            maxlen=config.LOG_BUFFER_MAX,
        )
        self._log_lock = threading.Lock()

        # Per-student cooldown: student_id -> last recognized timestamp
        self._seen: dict = {}
        self._seen_lock = threading.Lock()

        # Unknown face snapshot cooldown
        self._last_unknown_save: float = 0

        # Last encoded JPEG for MJPEG stream when no fresh frame available
        self._last_jpeg: bytes | None = None
        self._jpeg_lock = threading.Lock()

        # Ensure only one thread mutates pipeline state at a time.
        self._process_lock = threading.Lock()

        # -- Detect-Track-Recognize state --
        self._tracks: list[FaceTrack] = []
        self._frame_count: int = 0
        self._next_track_id: int = 0
        self._prev_gray: np.ndarray | None = None

    @staticmethod
    def _track_priority(trk: FaceTrack) -> tuple[int, int, int]:
        """Return a sortable priority for keeping the best duplicate track."""
        match_conf = 0.0
        if trk.identity is not None:
            try:
                match_conf = float(trk.identity[2])
            except Exception:
                match_conf = 0.0
        return (
            1 if trk.identity is not None else 0,
            match_conf,
            -trk.frames_missing,
            trk.bbox[2] * trk.bbox[3],
        )

    @staticmethod
    def _tracks_are_duplicates(a: FaceTrack, b: FaceTrack) -> bool:
        """Return True when two tracks likely represent the same face."""
        overlap = iou(a.bbox, b.bbox)
        if overlap >= 0.2:
            return True

        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter > 0:
            min_area = max(1.0, min(aw * ah, bw * bh))
            if inter / min_area >= 0.6:
                return True

        # Fallback for shifted boxes with low IoU but near-identical centers/sizes.
        dist = centroid_distance(a.bbox, b.bbox)
        min_face_span = max(1.0, min(a.bbox[2], a.bbox[3], b.bbox[2], b.bbox[3]))
        if dist <= (1.0 * min_face_span):
            return True

        # Nested rectangles with similar centers are often duplicate trackers.
        acx, acy = ax + aw / 2.0, ay + ah / 2.0
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0
        a_contains_b_center = ax <= bcx <= ax + aw and ay <= bcy <= ay + ah
        b_contains_a_center = bx <= acx <= bx + bw and by <= acy <= by + bh
        if a_contains_b_center or b_contains_a_center:
            area_a = max(1.0, aw * ah)
            area_b = max(1.0, bw * bh)
            ratio = max(area_a, area_b) / min(area_a, area_b)
            if ratio <= 12.0:
                return True

        # Boxes that are fairly close and partially overlap are usually
        # duplicate trackers on a single face.
        max_span = max(a.bbox[2], a.bbox[3], b.bbox[2], b.bbox[3], 1)
        if overlap >= 0.1 and dist <= (0.9 * max_span):
            return True

        return False

    def _deduplicate_tracks(self):
        """Remove overlapping duplicate tracks so one face keeps one box."""
        if len(self._tracks) < 2:
            return

        ordered = sorted(
            self._tracks,
            key=self._track_priority,
            reverse=True,
        )
        kept: list[FaceTrack] = []

        for trk in ordered:
            if any(self._tracks_are_duplicates(trk, existing) for existing in kept):
                continue
            kept.append(trk)

        if len(kept) != len(self._tracks):
            logger.debug(
                "Deduplicated tracks: before=%d after=%d",
                len(self._tracks),
                len(kept),
            )
        self._tracks = kept

    def _effective_detection_interval(self, motion_detected: bool) -> int:
        """Compute dynamic detector cadence for better CPU/latency balance."""
        interval = config.DETECTION_INTERVAL
        interval = min(max(interval, config.DETECTION_INTERVAL_MIN), config.DETECTION_INTERVAL_MAX)

        active_tracks = len(self._tracks)
        if active_tracks == 0:
            return config.DETECTION_INTERVAL_MIN

        if motion_detected:
            interval -= 2
        else:
            interval += 2

        if active_tracks >= 3:
            interval -= 1
        elif active_tracks == 1:
            interval += 1

        return min(
            config.DETECTION_INTERVAL_MAX,
            max(config.DETECTION_INTERVAL_MIN, interval),
        )

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
        )
        self._thread.start()
        self._process_thread.start()
        logger.info("Camera %d started.", self._source)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._process_thread:
            self._process_thread.join(timeout=2)
        if self._cap.isOpened():
            self._cap.release()
        logger.info("Camera %d stopped.", self._source)

    def _capture_loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_fresh = True

    def _process_loop(self):
        min_interval = 1.0 / max(config.MJPEG_TARGET_FPS, 1)
        while self._running:
            start = time.monotonic()
            self.process_frame()
            elapsed = time.monotonic() - start
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    def get_raw_frame(self, consume: bool = False) -> np.ndarray | None:
        """Return last captured frame. If *consume* is True, only return
        the frame when it is fresh (not yet consumed) to avoid
        reprocessing the same image when the camera feed is stale."""
        with self._lock:
            if self._frame is None:
                return None
            if consume:
                if not self._frame_fresh:
                    return None
                self._frame_fresh = False
            return self._frame.copy()

    # -- events --------------------------------------------------------------

    def pop_events(self) -> list[dict]:
        with self._events_lock:
            events = list(self._events)
            self._events.clear()
            return events

    def _push_event(self, event: dict):
        with self._events_lock:
            self._events.append(event)
        # Also persist to log buffer
        self._push_log(event)
        # Emit via WebSocket
        _emit_event("attendance_event", event)

    # -- log buffer (persistent, non-destructive read) -----------------------

    def get_log_buffer(self) -> list[dict]:
        """Return the full log buffer (most recent last)."""
        with self._log_lock:
            return list(self._log_buffer)

    def _push_log(self, event: dict):
        log_entry = {
            **event,
            "time": time.strftime("%H:%M:%S"),
        }
        with self._log_lock:
            self._log_buffer.append(log_entry)
        # Emit via WebSocket
        _emit_event("log_event", log_entry)

    # -- snapshot helper -----------------------------------------------------

    def _save_unknown_snapshot(self, frame):
        """Save unknown face snapshot (non-blocking, with cooldown)."""
        now = time.monotonic()
        if now - self._last_unknown_save < config.UNKNOWN_FACE_COOLDOWN:
            return
        self._last_unknown_save = now

        def _save():
            try:
                os.makedirs(config.UNKNOWN_FACES_DIR, exist_ok=True)
                ts = time.strftime("%Y-%m-%d_%H-%M-%S")
                path = os.path.join(config.UNKNOWN_FACES_DIR, f"{ts}.jpg")
                cv2.imwrite(path, frame)
                logger.info("Unknown face snapshot saved: %s", path)
            except Exception as exc:
                logger.error("Failed to save unknown face snapshot: %s", exc)

        threading.Thread(target=_save, daemon=True).start()

    @staticmethod
    def _to_raw_bbox(
        bbox_xywh: tuple[int, int, int, int],
        sx: float,
        sy: float,
    ) -> tuple[int, int, int, int]:
        return (
            int(bbox_xywh[0] * sx),
            int(bbox_xywh[1] * sy),
            int(bbox_xywh[2] * sx),
            int(bbox_xywh[3] * sy),
        )

    @staticmethod
    def _adaptive_liveness_crop(
        frame: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Create an adaptive padded square crop for anti-spoof inference."""
        x, y, w, h = bbox_xywh
        fh, fw = frame.shape[:2]

        span = max(w, h, 1)
        dynamic_ratio = config.ANTI_SPOOF_PAD_RATIO_BASE + (50.0 / span)
        pad_ratio = min(config.ANTI_SPOOF_PAD_RATIO_MAX, dynamic_ratio)
        pad = int(max(config.ANTI_SPOOF_PAD_MIN_PIXELS, span * pad_ratio))

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(fw, x + w + pad)
        y2 = min(fh, y + h + pad)

        rw = max(1, x2 - x1)
        rh = max(1, y2 - y1)
        side = max(rw, rh)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        sx1 = int(max(0, cx - side / 2.0))
        sy1 = int(max(0, cy - side / 2.0))
        sx2 = int(min(fw, sx1 + side))
        sy2 = int(min(fh, sy1 + side))

        if sx2 <= sx1 or sy2 <= sy1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[sy1:sy2, sx1:sx2]

    def _evaluate_track_liveness(
        self,
        trk: FaceTrack,
        liveness_frame: np.ndarray,
        liveness_box: tuple[int, int, int, int],
    ) -> tuple[str, float]:
        """Evaluate liveness with temporal voting and spoof hold cooldown."""
        now = time.monotonic()
        temporal_override = False

        if config.BYPASS_ANTISPOOF:
            label, conf = 1, 1.0
        else:
            face_crop = self._adaptive_liveness_crop(liveness_frame, liveness_box)
            if face_crop.size == 0:
                label, conf = 0, 0.0
            else:
                label, conf = check_liveness(face_crop)

        if label == -1:
            tracker.record_liveness_event("liveness_error")

        trk.liveness_history.append((label, conf))
        if len(trk.liveness_history) > config.LIVENESS_HISTORY_SIZE:
            trk.liveness_history.pop(0)

        if now < trk.spoof_hold_until:
            temporal_override = True
            state = "spoof"
            score = max(conf, config.LIVENESS_SPOOF_CONFIDENCE_MIN)
        elif (
            label == 1
            and conf >= max(
                config.LIVENESS_CONFIDENCE_THRESHOLD,
                config.LIVENESS_REAL_FAST_CONFIDENCE,
            )
        ):
            state = "real"
            score = conf
        elif label not in (1, -1) and conf >= config.LIVENESS_STRONG_SPOOF_CONFIDENCE:
            state = "spoof"
            score = conf
            trk.spoof_hold_until = now + config.SPOOF_HOLD_SECONDS
        else:
            valid = [(l, c) for l, c in trk.liveness_history if l != -1]
            min_hist = max(1, config.LIVENESS_MIN_HISTORY)

            if len(valid) >= min_hist:
                real_scores = [
                    c for l, c in valid
                    if l == 1 and c >= config.LIVENESS_CONFIDENCE_THRESHOLD
                ]
                spoof_scores = [
                    c for l, c in valid
                    if l not in (1, -1) and c >= config.LIVENESS_SPOOF_CONFIDENCE_MIN
                ]
                weak_spoof_scores = [
                    c for l, c in valid
                    if l not in (1, -1)
                ]
                real_ratio = len(real_scores) / len(valid)
                spoof_ratio = len(spoof_scores) / len(valid)
                weak_spoof_ratio = len(weak_spoof_scores) / len(valid)

                if real_ratio >= config.LIVENESS_REAL_VOTE_RATIO and real_scores:
                    state = "real"
                    score = float(sum(real_scores) / len(real_scores))
                elif spoof_ratio >= config.LIVENESS_SPOOF_VOTE_RATIO and spoof_scores:
                    state = "spoof"
                    score = float(sum(spoof_scores) / len(spoof_scores))
                    trk.spoof_hold_until = now + config.SPOOF_HOLD_SECONDS
                elif (
                    weak_spoof_ratio >= config.LIVENESS_SPOOF_VOTE_RATIO
                    and weak_spoof_scores
                    and (sum(weak_spoof_scores) / len(weak_spoof_scores))
                    >= config.LIVENESS_SPOOF_WEAK_CONFIDENCE_MIN
                ):
                    state = "spoof"
                    score = float(sum(weak_spoof_scores) / len(weak_spoof_scores))
                    trk.spoof_hold_until = now + config.SPOOF_HOLD_SECONDS
                else:
                    state = "uncertain"
                    score = conf
            else:
                state = "uncertain"
                score = conf

        if temporal_override:
            tracker.record_liveness_event("temporal_override")

        if state == "real":
            trk.liveness = (1, score)
        elif state == "spoof":
            trk.liveness = (0, score)
        elif label == -1:
            trk.liveness = (-1, score)
        else:
            trk.liveness = (2, score)

        return state, score

    def _set_track_liveness_state(self, trk: FaceTrack, state: str, score: float):
        prev_state = trk.liveness_state
        trk.liveness_state = state

        if state == "real":
            trk.is_spoof = False
            if trk.identity is None:
                trk.is_unknown = False
                trk.state = "liveness_pending"
            return

        if state == "spoof":
            trk.is_spoof = True
            trk.is_unknown = False
            trk.state = "spoof"
            if prev_state != "spoof":
                tracker.record_liveness_event("spoof_true")
                tracker.record_liveness_event("spoof_detected")
                tracker.record_recognition(False, False)
                self._push_event({
                    "name": "Unknown",
                    "status": "spoof_detected",
                    "confidence": 0.0,
                    "liveness_confidence": round(score, 4),
                })
            return

        trk.is_spoof = False
        if trk.identity is None:
            trk.is_unknown = True
            trk.state = "liveness_pending"
        if prev_state != "uncertain":
            tracker.record_liveness_event("spoof_uncertain")
            tracker.record_liveness_event("liveness_uncertain")

    def _evaluate_track_ppe(
        self,
        trk: FaceTrack,
        frame: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
    ) -> tuple[str, float]:
        """Evaluate mask/cap state with temporal smoothing."""
        if not config.PPE_DETECTION_ENABLED:
            trk.ppe_state = "none"
            trk.ppe_confidence = 0.0
            return "none", 0.0

        try:
            result = detect_ppe(frame, bbox_xywh)
        except Exception:
            tracker.record_liveness_event("ppe_model_error")
            trk.ppe_state = "none"
            trk.ppe_confidence = 0.0
            return "none", 0.0

        state = str(result.get("state", "none"))
        confidence = float(result.get("confidence", 0.0))

        trk.ppe_history.append((state, confidence))
        if len(trk.ppe_history) > config.PPE_HISTORY_SIZE:
            trk.ppe_history.pop(0)

        valid = [
            (s, c)
            for s, c in trk.ppe_history
            if c >= config.PPE_MIN_CONFIDENCE
        ]

        if len(valid) >= max(1, config.PPE_MIN_HISTORY):
            state_scores: dict[str, list[float]] = {}
            for s, c in valid:
                state_scores.setdefault(s, []).append(c)

            best_state = "none"
            best_ratio = 0.0
            best_score = confidence
            total = float(len(valid))
            for s, scores in state_scores.items():
                ratio = len(scores) / total
                if ratio > best_ratio:
                    best_state = s
                    best_ratio = ratio
                    best_score = float(sum(scores) / len(scores))

            if best_ratio >= config.PPE_VOTE_RATIO:
                state = best_state
                confidence = best_score

        trk.ppe_state = state
        trk.ppe_confidence = confidence
        trk.ppe_updated_at = time.monotonic()

        if state == "mask":
            tracker.record_liveness_event("ppe_mask_detected")
        elif state == "cap":
            tracker.record_liveness_event("ppe_cap_detected")
        elif state == "both":
            tracker.record_liveness_event("ppe_both_detected")

        return state, confidence

    def _try_recognize_track(
        self,
        trk: FaceTrack,
        frame: np.ndarray,
        raw_frame: np.ndarray,
        raw_bbox_xywh: tuple[int, int, int, int],
        liveness_conf: float,
        ppe_state: str = "none",
        ppe_confidence: float = 0.0,
    ) -> None:
        """Run encoding+recognition for a track only when liveness is stable."""
        if liveness_conf < (config.LIVENESS_CONFIDENCE_THRESHOLD + config.LIVENESS_NO_ENCODE_MARGIN):
            tracker.record_liveness_event("no_encode_guard")
            trk.is_unknown = True
            trk.state = "liveness_pending"
            return

        encoding, quality_reason = _recognition_module().encode_face_with_reason(raw_frame, raw_bbox_xywh)
        if encoding is None:
            trk.is_unknown = True
            trk.state = "liveness_pending"
            trk.quality_reason = quality_reason
            tracker.record_liveness_event("quality_rejection")
            self._push_event({
                "name": "Unknown",
                "status": "quality_rejection",
                "confidence": 0.0,
                "details": quality_reason or "Face quality check failed (blurry, too dark, or too bright)",
            })
            tracker.record_recognition(False, False)
            return

        result = _face_engine_module().recognize_face(
            encoding,
            ppe_state=ppe_state,
            ppe_confidence=ppe_confidence,
        )
        if result is not None:
            student_id, name, confidence = result
            required_hits = max(1, int(config.RECOGNITION_CONFIRM_FRAMES))

            if getattr(trk, "candidate_student_id", None) == student_id:
                trk.candidate_hits = int(getattr(trk, "candidate_hits", 0)) + 1
                trk.candidate_best_confidence = max(
                    float(getattr(trk, "candidate_best_confidence", 0.0)),
                    float(confidence),
                )
            else:
                trk.candidate_student_id = student_id
                trk.candidate_name = name
                trk.candidate_hits = 1
                trk.candidate_best_confidence = float(confidence)

            if trk.candidate_hits < required_hits:
                trk.is_unknown = True
                trk.state = "liveness_pending"
                trk.quality_reason = (
                    f"Confirming identity ({trk.candidate_hits}/{required_hits})"
                )
                return

            final_confidence = float(trk.candidate_best_confidence)
            trk.identity = (student_id, name, final_confidence)
            trk.is_unknown = False
            trk.state = "recognized"
            trk.quality_reason = ""
            trk.candidate_student_id = None
            trk.candidate_name = ""
            trk.candidate_hits = 0
            trk.candidate_best_confidence = 0.0
            logger.info(
                "Track #%d: RECOGNIZED %s (confidence=%.4f)",
                trk.track_id, name, final_confidence,
            )
            self._handle_recognized(
                student_id, name, final_confidence, liveness_conf, frame,
                encoding=encoding,
                ppe_state=ppe_state,
                ppe_confidence=ppe_confidence,
            )
            return

        trk.candidate_student_id = None
        trk.candidate_name = ""
        trk.candidate_hits = 0
        trk.candidate_best_confidence = 0.0
        trk.is_unknown = True
        trk.state = "liveness_pending"
        trk.quality_reason = "Face not confidently matched to a single student."
        tracker.record_recognition(False, False)
        self._save_unknown_snapshot(raw_frame)

    # -- recognise a new face and create a track -----------------------------

    def _create_track(self, frame: np.ndarray,
                      bbox_xywh: tuple[int, int, int, int],
                      raw_frame: np.ndarray | None = None,
                      raw_bbox_xywh: tuple[int, int, int, int] | None = None) -> FaceTrack:
        """Run anti-spoof -> aligned encoding -> recognition for a
        newly-detected face and return a fully populated
        :class:`~pipeline.FaceTrack`."""
        tid = self._next_track_id
        self._next_track_id += 1
        trk = FaceTrack(tid, frame, bbox_xywh)
        trk.created_at = time.monotonic()
        trk.state = "detecting"

        liveness_frame = raw_frame if raw_frame is not None else frame
        liveness_box = raw_bbox_xywh if raw_bbox_xywh is not None else bbox_xywh
        state, liveness_conf = self._evaluate_track_liveness(
            trk,
            liveness_frame,
            liveness_box,
        )
        self._set_track_liveness_state(trk, state, liveness_conf)

        ppe_state, ppe_confidence = "none", 0.0
        if state == "real":
            ppe_state, ppe_confidence = self._evaluate_track_ppe(
                trk,
                liveness_frame,
                liveness_box,
            )

        if state == "real":
            encode_frame = raw_frame if raw_frame is not None else frame
            encode_box = raw_bbox_xywh if raw_bbox_xywh is not None else bbox_xywh
            self._try_recognize_track(
                trk,
                frame,
                encode_frame,
                encode_box,
                liveness_conf,
                ppe_state=ppe_state,
                ppe_confidence=ppe_confidence,
            )
        elif state == "uncertain":
            tracker.record_recognition(False, False)

        logger.info("New track #%d created at bbox=%s", tid, bbox_xywh)
        return trk

    def _handle_recognized(
        self,
        student_id,
        name: str,
        confidence: float,
        liveness_conf: float,
        frame: np.ndarray,
        encoding: np.ndarray | None = None,
        ppe_state: str = "none",
        ppe_confidence: float = 0.0,
    ):
        """Apply cooldown check, mark attendance, and perform incremental learning."""
        now = time.monotonic()
        with self._seen_lock:
            last_seen = self._seen.get(student_id, 0)
            if now - last_seen < config.RECOGNITION_COOLDOWN:
                tracker.record_recognition(True, True)
                return
            # Update timestamp IMMEDIATELY to prevent race condition where
            # another frame passes cooldown check before we mark attendance
            self._seen[student_id] = now

        student_doc = database.get_student_by_id(student_id)
        student_meta = {}
        if student_doc:
            student_meta = {
                "registration_number": student_doc.get("registration_number", ""),
                "semester": student_doc.get("semester", ""),
                "section": student_doc.get("section", ""),
            }

        marked = database.mark_attendance(student_id, confidence)

        self._push_event({
            "name": name,
            "status": "marked",
            "confidence": round(confidence, 4),
            "liveness_confidence": round(liveness_conf, 4),
            "ppe_state": ppe_state,
            "ppe_confidence": round(float(ppe_confidence), 4),
            "attendance_marked": marked,
            **student_meta,
        })
        tracker.record_recognition(True, True)

        if marked:
            logger.info(
                "Attendance marked: student=%s confidence=%.4f "
                "liveness_confidence=%.4f",
                name, confidence, liveness_conf,
            )

        if (
            encoding is not None
            and confidence >= config.INCREMENTAL_LEARNING_CONFIDENCE
            and liveness_conf >= config.INCREMENTAL_LEARNING_MIN_LIVENESS
            and (
                not config.OCCLUDED_DISABLE_INCREMENTAL_LEARNING
                or ppe_state == "none"
            )
        ):
            try:
                flat_enc, flat_idx, ids, _ = _encoding_cache().get_flat()
                if flat_enc is not None and flat_idx is not None and student_id in ids:
                    student_idx = ids.index(student_id)
                    student_rows = np.where(flat_idx == student_idx)[0]
                    if len(student_rows) > 0:
                        min_dist = float(
                            np.min(np.linalg.norm(flat_enc[student_rows] - encoding, axis=1))
                        )
                        # Increased threshold from 0.15 to 0.20 to reduce duplicate encodings
                        # while still skipping very similar faces
                        if min_dist < 0.20:
                            logger.debug(
                                "Incremental learning skipped for %s (duplicate: dist=%.4f < threshold 0.20)",
                                name,
                                min_dist,
                            )
                            return

                _face_engine_module().append_encoding(student_id, encoding)
                logger.info(
                    "Incremental learning: appended encoding for %s "
                    "(confidence=%.4f)",
                    name, confidence,
                )
            except Exception as exc:
                logger.error(
                    "Incremental learning failed for %s: %s", name, exc,
                )

    def process_frame(self) -> bytes | None:
        """Grab latest frame, update trackers, detect new faces, and return JPEG."""
        if not self._process_lock.acquire(blocking=False):
            return self.get_latest_jpeg()

        try:
            raw = self.get_raw_frame(consume=True)
            if raw is None:
                return self.get_latest_jpeg()

            frame = _resize_to_process_width(raw)
            h_raw, w_raw = raw.shape[:2]
            h_proc, w_proc = frame.shape[:2]
            sx = w_raw / w_proc if w_proc else 1.0
            sy = h_raw / h_proc if h_proc else 1.0

            start_time = time.perf_counter()
            self._frame_count += 1

            if self._frame_count > 1_000_000:
                self._frame_count = 0

            if self._frame_count % 1000 == 0:
                cutoff = time.monotonic() - (config.RECOGNITION_COOLDOWN * 2)
                with self._seen_lock:
                    self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

            surviving: list[FaceTrack] = []
            for trk in self._tracks:
                trk.update(frame)
                if trk.frames_missing <= config.TRACK_EXPIRATION_FRAMES:
                    surviving.append(trk)
                else:
                    trk.state = "expired"
                    tracker.record_liveness_event("tracker_expired")
                    logger.debug("Track #%d expired.", trk.track_id)
            self._tracks = surviving
            self._deduplicate_tracks()

            if config.BYPASS_MOTION_DETECTION:
                motion = True
            else:
                motion, self._prev_gray = detect_motion(self._prev_gray, frame)

            dynamic_interval = self._effective_detection_interval(motion)
            if self._frame_count % dynamic_interval == 0:
                run_detection = motion or (
                    self._frame_count % config.NO_MOTION_DETECTION_INTERVAL == 0
                )
                if not self._tracks:
                    run_detection = True

                if run_detection:
                    new_boxes, matched_track_indices = detect_and_associate_detailed(
                        frame,
                        self._tracks,
                    )
                    created_track_ids: set[int] = set()

                    stale_tracks: list[FaceTrack] = []
                    for idx, trk in enumerate(self._tracks):
                        if idx in matched_track_indices:
                            trk.detector_misses = 0
                        else:
                            trk.detector_misses += 1
                        if trk.detector_misses <= config.TRACK_DETECTOR_MISS_TOLERANCE:
                            stale_tracks.append(trk)
                        else:
                            trk.state = "expired"
                            tracker.record_liveness_event("detector_pruned")
                            if not config.DEBUG_MODE:
                                continue
                            logger.debug(
                                "Track #%d expired by detector miss tolerance (%d)",
                                trk.track_id,
                                trk.detector_misses,
                            )
                    self._tracks = stale_tracks

                    if config.DEBUG_MODE and new_boxes:
                        logger.debug(
                            "Frame %d: %d new face(s) detected, %d active track(s)",
                            self._frame_count,
                            len(new_boxes),
                            len(self._tracks),
                        )
                    for box in new_boxes:
                        raw_box = self._to_raw_bbox(box, sx, sy)
                        trk = self._create_track(
                            frame,
                            box,
                            raw_frame=raw,
                            raw_bbox_xywh=raw_box,
                        )
                        trk.detector_misses = 0
                        self._tracks.append(trk)
                        created_track_ids.add(trk.track_id)

                    for trk in self._tracks:
                        if trk.identity is not None:
                            continue
                        if trk.track_id in created_track_ids:
                            continue
                        raw_box = self._to_raw_bbox(trk.bbox, sx, sy)
                        state, liveness_conf = self._evaluate_track_liveness(
                            trk,
                            raw,
                            raw_box,
                        )
                        self._set_track_liveness_state(trk, state, liveness_conf)
                        if state == "real":
                            ppe_state, ppe_confidence = self._evaluate_track_ppe(
                                trk,
                                raw,
                                raw_box,
                            )
                            self._try_recognize_track(
                                trk,
                                frame,
                                raw,
                                raw_box,
                                liveness_conf,
                                ppe_state=ppe_state,
                                ppe_confidence=ppe_confidence,
                            )

                    self._deduplicate_tracks()

            for trk in self._tracks:
                draw_track_overlay(frame, trk, self._seen, self._seen_lock)

            status_text = f"Tracks: {len(self._tracks)}  FPS: {tracker.metrics().get('fps', 0):.1f}"
            cv2.putText(
                frame,
                status_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            if config.DEBUG_MODE:
                debug_lines = [
                    f"Frame: {self._frame_count}",
                    f"DetInt: {dynamic_interval}",
                    f"Threshold: {config.RECOGNITION_THRESHOLD:.2f}",
                    f"Cache: {_encoding_cache().size} students",
                ]
                if config.BYPASS_ANTISPOOF:
                    debug_lines.append("ANTISPOOF: OFF")
                if config.BYPASS_QUALITY_GATE:
                    debug_lines.append("QUALITY GATE: OFF")
                if config.BYPASS_MOTION_DETECTION:
                    debug_lines.append("MOTION GATE: OFF")
                for i, line in enumerate(debug_lines):
                    cv2.putText(
                        frame,
                        line,
                        (10, 40 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

            elapsed = time.perf_counter() - start_time
            tracker.record_frame_time(elapsed)

            if self._frame_count % 30 == 0:
                _emit_event("metrics_update", tracker.metrics())

            _, jpeg = cv2.imencode(".jpg", frame)
            result = jpeg.tobytes()
            with self._jpeg_lock:
                self._last_jpeg = result
            return result
        finally:
            self._process_lock.release()

    def get_latest_jpeg(self) -> bytes | None:
        """Return the most recently processed JPEG frame."""
        with self._jpeg_lock:
            return self._last_jpeg

    def diagnostics(self) -> dict:
        """Return runtime diagnostics for this camera instance."""
        with self._lock:
            frame_shape = self._frame.shape[:2] if self._frame is not None else None
        return {
            "source": self._source,
            "running": self._running,
            "frame_shape": frame_shape,
            "frame_count": self._frame_count,
            "active_tracks": len(self._tracks),
            "tracks": [
                {
                    "track_id": trk.track_id,
                    "state": trk.state,
                    "liveness_state": trk.liveness_state,
                    "liveness": trk.liveness,
                    "ppe_state": getattr(trk, "ppe_state", "none"),
                    "ppe_confidence": getattr(trk, "ppe_confidence", 0.0),
                    "frames_missing": trk.frames_missing,
                    "detector_misses": trk.detector_misses,
                    "has_identity": trk.identity is not None,
                    "quality_reason": trk.quality_reason,
                }
                for trk in self._tracks
            ],
        }


# ---------------------------------------------------------------------------
# Multi-camera management
# ---------------------------------------------------------------------------
_cameras: dict[int, Camera] = {}
_camera_viewers: dict[int, int] = {}
_cameras_lock = threading.Lock()


def get_camera(source: int = 0) -> Camera:
    """Return a Camera instance for the given device index, creating it
    if necessary."""
    global _cameras
    with _cameras_lock:
        if source not in _cameras:
            cam = Camera(source)
            cam.start()
            _cameras[source] = cam
        return _cameras[source]


def get_camera_if_running(source: int = 0) -> Camera | None:
    """Return a running Camera instance if available, else None."""
    with _cameras_lock:
        return _cameras.get(source)


def acquire_camera_stream(source: int = 0) -> Camera:
    """Acquire a camera for active video streaming.

    The caller must pair this with ``release_camera_stream`` when the
    stream disconnects.
    """
    global _camera_viewers
    with _cameras_lock:
        cam = _cameras.get(source)
        if cam is None:
            cam = Camera(source)
            cam.start()
            _cameras[source] = cam

        _camera_viewers[source] = _camera_viewers.get(source, 0) + 1
        return cam


def release_camera_stream(source: int = 0):
    """Release one active stream viewer; stop camera when last viewer leaves."""
    global _camera_viewers
    with _cameras_lock:
        if source not in _camera_viewers:
            return

        remaining = _camera_viewers[source] - 1
        if remaining > 0:
            _camera_viewers[source] = remaining
            return

        _camera_viewers.pop(source, None)
        cam = _cameras.pop(source, None)
        if cam is not None:
            cam.stop()


def get_all_cameras() -> dict[int, Camera]:
    """Return the dict of all active camera instances."""
    with _cameras_lock:
        return dict(_cameras)


def release_camera(source: int | None = None):
    """Release one camera (by index) or all cameras (if source is None)."""
    global _cameras
    with _cameras_lock:
        if source is not None:
            _camera_viewers.pop(source, None)
            cam = _cameras.pop(source, None)
            if cam is not None:
                cam.stop()
        else:
            _camera_viewers.clear()
            for cam in _cameras.values():
                cam.stop()
            _cameras.clear()


def get_camera_diagnostics() -> dict:
    """Return diagnostics for all managed cameras and stream viewers."""
    with _cameras_lock:
        cameras_snapshot = dict(_cameras)
        viewers_snapshot = dict(_camera_viewers)

    return {
        "active_cameras": len(cameras_snapshot),
        "viewers": viewers_snapshot,
        "cameras": {
            source: cam.diagnostics()
            for source, cam in cameras_snapshot.items()
        },
    }
