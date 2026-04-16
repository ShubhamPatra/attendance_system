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
import atexit
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

import cv2
import numpy as np

import core.config as config
import core.database as database
from vision.overlay import draw_track_overlay
from core.performance import tracker
from vision.pipeline import FaceTrack, centroid_distance, detect_and_associate_detailed, detect_motion, iou
from core.utils import setup_logging

logger = setup_logging()

# -- SocketIO reference (set by app.py after init) --------------------------
_socketio = None
_lazy_face_engine = None
_lazy_recognition = None
_lazy_anti_spoofing = None
_lazy_ppe_detection = None
_snapshot_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="unknown-snapshot")


@atexit.register
def _shutdown_snapshot_executor():
    _snapshot_executor.shutdown(wait=False, cancel_futures=True)


def _face_engine_module():
    global _lazy_face_engine
    if _lazy_face_engine is None:
        import vision.face_engine as _mod
        _lazy_face_engine = _mod
    return _lazy_face_engine


def _recognition_module():
    global _lazy_recognition
    if _lazy_recognition is None:
        import vision.recognition as _mod
        _lazy_recognition = _mod
    return _lazy_recognition


def _anti_spoofing_module():
    global _lazy_anti_spoofing
    if _lazy_anti_spoofing is None:
        import vision.anti_spoofing as _mod
        _lazy_anti_spoofing = _mod
    return _lazy_anti_spoofing


def _ppe_detection_module():
    global _lazy_ppe_detection
    if _lazy_ppe_detection is None:
        import vision.ppe_detection as _mod
        _lazy_ppe_detection = _mod
    return _lazy_ppe_detection


def _encoding_cache():
    return _face_engine_module().encoding_cache


def check_liveness(*args, **kwargs):
    """Compatibility wrapper for tests and monkeypatching."""
    return _anti_spoofing_module().check_liveness(*args, **kwargs)


def analyze_liveness_frame(*args, **kwargs):
    """Compatibility wrapper for tests and monkeypatching."""
    return _anti_spoofing_module().analyze_liveness_frame(*args, **kwargs)


def encode_face(*args, **kwargs):
    """Compatibility wrapper for tests and monkeypatching."""
    return _recognition_module().encode_face(*args, **kwargs)


def _compute_smoothed_embedding(embedding_history: list) -> np.ndarray | None:
    """Compute L2-normalised average of recent embeddings."""
    if not embedding_history:
        return None
    stacked = np.stack(embedding_history, axis=0)
    avg = np.mean(stacked, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg.astype(np.float32)


def _track_center_from_box(box_xywh: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box_xywh
    return x + (w / 2.0), y + (h / 2.0)


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float32)))


def detect_ppe(*args, **kwargs):
    """Compatibility wrapper for tests and monkeypatching."""
    return _ppe_detection_module().detect_ppe(*args, **kwargs)


def _extract_landmarks_5_from_bbox(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract InsightFace 5-point landmarks and embedding in ONE call.

    Runs InsightFace detection+recognition on a padded ROI around the
    track box, then maps keypoints back to the original frame coordinate
    system.

    Returns
    -------
    (landmarks_5, embedding) where either may be None on failure.
    Returning the embedding here avoids a redundant InsightFace inference
    in the encoding step.
    """
    if config.EMBEDDING_BACKEND != "arcface":
        return None, None

    x, y, w, h = bbox_xywh
    fh, fw = frame_bgr.shape[:2]

    pad = int(max(w, h) * 0.25)
    px1 = max(0, x - pad)
    py1 = max(0, y - pad)
    px2 = min(fw, x + w + pad)
    py2 = min(fh, y + h + pad)
    if px2 <= px1 or py2 <= py1:
        return None, None

    roi = frame_bgr[py1:py2, px1:px2]
    try:
        af = _face_engine_module().get_arcface_backend()
        faces = af.get_faces(roi)
    except Exception as exc:
        logger.debug("Could not extract ArcFace landmarks for blink tracking: %s", exc)
        return None, None

    try:
        faces = list(faces)
    except TypeError:
        faces = []

    if not faces:
        return None, None

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    # Extract embedding from the same face object (no extra inference!)
    embedding = None
    try:
        embedding = af.get_embedding_from_face(face)
    except Exception:
        pass

    kps = getattr(face, "kps", None)
    if kps is None:
        return None, embedding

    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        return None, embedding

    kps[:, 0] += px1
    kps[:, 1] += py1
    return kps, embedding


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
    """Resize *frame* to ``FRAME_PROCESS_WIDTH × PERF_FRAME_SCALE``
    maintaining aspect ratio.  Returns the original frame unchanged if
    it is already at or below the target width.

    ``PERF_FRAME_SCALE`` (default 1.0) provides an additional scaling
    knob on top of the base processing width — e.g. 0.75 makes the
    processing frame 75 % of ``FRAME_PROCESS_WIDTH``, reducing both
    detection and tracking cost.
    """
    target_w = int(config.FRAME_PROCESS_WIDTH * max(0.1, min(1.0, config.PERF_FRAME_SCALE)))
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / w
    new_w = target_w
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

        # Reconnection backoff tracking
        self._reconnect_attempts = 0
        self._last_reconnect_time = time.time()
        self._reconnect_backoff_delay = config.CAMERA_RECONNECT_INITIAL_DELAY_SECONDS

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

        # Per-track recognition cache: track_id -> {result, expires_at}
        self._track_identity_cache: dict[int, dict] = {}

        # Per-camera active session cache (short TTL to reduce DB lookups)
        self._active_session_cache: dict | None = None
        self._active_session_cached_at: float = 0.0

        # Unknown face snapshot cooldown
        self._last_unknown_save: float = 0
        self._last_unknown_track_id: int | None = None
        self._last_unknown_confidence: float = 0.0
        self._process_frame_times: collections.deque[float] = collections.deque(maxlen=120)

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

    def _get_track_cached_result(self, track_id: int):
        """Return cached recognition result for a track if entry is still valid."""
        entry = self._track_identity_cache.get(track_id)
        if not entry:
            return None, False
        if float(entry.get("expires_at", 0.0)) <= time.monotonic():
            self._track_identity_cache.pop(track_id, None)
            return None, False
        return entry.get("result"), True

    def _set_track_cached_result(self, track_id: int, result):
        """Store recognition result for a track with TTL-based expiry."""
        ttl = max(0.0, float(config.RECOGNITION_TRACK_CACHE_TTL_SECONDS))
        self._track_identity_cache[track_id] = {
            "result": result,
            "expires_at": time.monotonic() + ttl,
        }

        max_entries = max(1, int(config.RECOGNITION_TRACK_CACHE_MAX_ENTRIES))
        if len(self._track_identity_cache) <= max_entries:
            return

        now = time.monotonic()
        expired_ids = [
            tid
            for tid, item in self._track_identity_cache.items()
            if float(item.get("expires_at", 0.0)) <= now
        ]
        for tid in expired_ids:
            self._track_identity_cache.pop(tid, None)

        if len(self._track_identity_cache) > max_entries:
            overflow = len(self._track_identity_cache) - max_entries
            oldest = sorted(
                self._track_identity_cache.items(),
                key=lambda kv: float(kv[1].get("expires_at", 0.0)),
            )[:overflow]
            for tid, _ in oldest:
                self._track_identity_cache.pop(tid, None)

    def _record_stage_time(self, stage_name: str, start_time: float) -> None:
        tracker.record_stage_time(stage_name, time.perf_counter() - start_time)

    def _reset_track_verification(
        self,
        trk: FaceTrack,
        reason: str,
        keep_spoof_hold: bool = False,
        preserve_state: bool = False,
    ) -> None:
        """Reset verification-specific state without destroying the tracker."""
        trk.liveness_history = []
        trk.confidence_history = []
        trk.motion_history = []
        trk.face_center_history = []
        trk.screen_history = []
        trk.brightness_history = []
        trk.contrast_history = []
        trk.last_liveness_meta = {"reset_reason": reason}
        trk.verification_started_at = None
        trk.last_seen_at = time.monotonic()
        if not keep_spoof_hold:
            trk.spoof_hold_until = 0.0
        if not preserve_state:
            trk.is_unknown = True
            trk.is_spoof = False
            trk.liveness_state = "init"
            trk.state = "detecting"

    def _update_track_motion_history(
        self,
        trk: FaceTrack,
        bbox_xywh: tuple[int, int, int, int],
    ) -> float:
        now = time.monotonic()
        center_x, center_y = _track_center_from_box(bbox_xywh)
        trk.face_center_history.append((now, center_x, center_y))
        if len(trk.face_center_history) > max(5, int(config.LIVENESS_HISTORY_SIZE)):
            trk.face_center_history.pop(0)

        motion_px = 0.0
        if len(trk.face_center_history) >= 2:
            _, prev_x, prev_y = trk.face_center_history[-2]
            motion_px = float(np.hypot(center_x - prev_x, center_y - prev_y))
        trk.motion_history.append(motion_px)
        if len(trk.motion_history) > max(5, int(config.LIVENESS_HISTORY_SIZE)):
            trk.motion_history.pop(0)
        return motion_px

    def _track_motion_low(self, trk: FaceTrack) -> bool:
        recent = trk.motion_history[-3:] if trk.motion_history else []
        if not recent:
            return True
        return float(np.mean(recent)) < config.LIVENESS_FACE_MOTION_MIN_PIXELS

    def _screen_heuristics_allowed(self, trk: FaceTrack) -> bool:
        has_blink = getattr(trk, "blink_count", 0) > 0
        return (not has_blink) and self._track_motion_low(trk)

    def _weighted_liveness_score(
        self,
        model_conf: float,
        blink_score: float,
        motion_score: float,
        screen_penalty: float,
    ) -> float:
        model_weight = 0.7
        blink_weight = 0.15
        motion_weight = 0.15
        score = (model_weight * model_conf) + (blink_weight * blink_score) + (motion_weight * motion_score)
        score -= screen_penalty
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _track_priority(trk: FaceTrack) -> tuple[int, int, int]:
        """Return a sortable priority for keeping the best duplicate track."""
        match_conf = 0.0
        if trk.identity is not None:
            try:
                match_conf = float(trk.identity[2])
            except (ValueError, TypeError, IndexError):
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
        """Capture frames from camera with exponential backoff on read failures."""
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                # Reset backoff on successful read
                self._reconnect_attempts = 0
                self._reconnect_backoff_delay = config.CAMERA_RECONNECT_INITIAL_DELAY_SECONDS
                with self._lock:
                    self._frame = frame
                    self._frame_fresh = True
            else:
                # Handle read failure with exponential backoff
                self._reconnect_attempts += 1
                if self._reconnect_attempts == 1:
                    logger.warning(
                        "Camera %d read failed; starting exponential backoff reconnection.",
                        self._source,
                    )
                
                if self._reconnect_attempts >= config.CAMERA_RECONNECT_MAX_ATTEMPTS:
                    logger.error(
                        "Camera %d: max reconnection attempts (%d) reached; "
                        "reconnection will continue but please check device.",
                        self._source,
                        config.CAMERA_RECONNECT_MAX_ATTEMPTS,
                    )
                    # Keep trying but log warning less frequently
                    if self._reconnect_attempts % config.CAMERA_RECONNECT_MAX_ATTEMPTS == 0:
                        logger.warning(
                            "Camera %d still disconnected after %d attempts; "
                            "still retrying with %0.1f second interval.",
                            self._source,
                            self._reconnect_attempts,
                            self._reconnect_backoff_delay,
                        )
                else:
                    # Exponential backoff: double the delay, capped at max
                    self._reconnect_backoff_delay = min(
                        self._reconnect_backoff_delay * 2,
                        config.CAMERA_RECONNECT_MAX_DELAY_SECONDS,
                    )
                    logger.debug(
                        "Camera %d read attempt %d; backoff delay: %.1f seconds",
                        self._source,
                        self._reconnect_attempts,
                        self._reconnect_backoff_delay,
                    )
                
                time.sleep(self._reconnect_backoff_delay)

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
                logger.info(
                    "Unknown face snapshot saved: %s (track=%s confidence=%.4f)",
                    path,
                    self._last_unknown_track_id,
                    self._last_unknown_confidence,
                )
            except Exception as exc:
                logger.error("Failed to save unknown face snapshot: %s", exc)

        try:
            _snapshot_executor.submit(_save)
        except Exception as exc:
            logger.error("Failed to queue unknown face snapshot: %s", exc)

    # Timestamp for throttled idle-session cleanup (not every cache refresh)
    _idle_session_cleanup_at: float = 0.0

    def _get_active_session_for_camera(self, force_refresh: bool = False) -> dict | None:
        """Return active attendance session for this camera, if available."""
        now = time.monotonic()
        if (
            not force_refresh
            and (now - self._active_session_cached_at) < config.ATTENDANCE_SESSION_CACHE_SECONDS
        ):
            return self._active_session_cache

        camera_id = str(self._source)
        try:
            # Throttle idle-session cleanup to once per 60s instead of every
            # cache refresh (~2s), avoiding a MongoDB update_many in the hot path.
            if now - self._idle_session_cleanup_at >= 60.0:
                self._idle_session_cleanup_at = now
                database.auto_close_idle_attendance_sessions(
                    idle_seconds=config.ATTENDANCE_SESSION_IDLE_TIMEOUT_SECONDS
                )
            active = database.get_active_attendance_session(camera_id)
        except RuntimeError as exc:
            logger.error(
                "Could not resolve active attendance session for camera %s: %s",
                camera_id,
                exc,
            )
            active = None

        self._active_session_cache = active
        self._active_session_cached_at = now
        return active

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
        """Evaluate liveness with temporal voting and spoof hold cooldown.
        
        Decision logic (in order):
        1. If in spoof hold timeout: return "spoof"
        2. If strong real evidence (label=1, conf >= 0.72 fast threshold): return "real"
        3. If strong spoof evidence (label∈{0,2}, conf >= 0.85): return "spoof" + Set hold
        4. Use temporal voting on history (require 3+ frames):
           - Real vote: label=1 AND conf >= 0.55 (threshold); ratio >= 70% → "real"
           - Spoof vote: label∈{0,2} AND conf >= 0.60 (threshold); ratio >= 60% → "spoof"
           - Weak spoof vote: label∈{0,2}, any conf, if weak ratio >= 60% and avg conf >= 0.45 → "spoof"
        5. Default: "uncertain"
        """
        now = time.monotonic()
        trk.last_seen_at = now

        if trk.verification_started_at is None:
            trk.verification_started_at = now

        if config.BYPASS_ANTISPOOF:
            analysis = {
                "label": 1,
                "confidence": 1.0,
                "early_reject": False,
                "too_small": False,
                "screen_suspicious": False,
                "mean_brightness": 0.0,
                "contrast_proxy": 0.0,
            }
        else:
            face_crop = self._adaptive_liveness_crop(liveness_frame, liveness_box)
            if face_crop.size == 0:
                analysis = {
                    "label": 0,
                    "confidence": 0.0,
                    "early_reject": True,
                    "reject_reason": "empty_crop",
                    "too_small": True,
                    "screen_suspicious": False,
                    "mean_brightness": 0.0,
                    "contrast_proxy": 0.0,
                }
            else:
                analysis = analyze_liveness_frame(face_crop, face_bbox=liveness_box)

        label = int(analysis.get("label", 0))
        conf = float(analysis.get("confidence", 0.0))
        trk.last_liveness_meta = analysis

        if label == -1:
            tracker.record_liveness_event("liveness_error")

        if analysis.get("too_small") or conf < config.LIVENESS_EARLY_REJECT_CONFIDENCE:
            trk.liveness_history.append((0, conf))
            trk.confidence_history.append(conf)
            if len(trk.liveness_history) > config.LIVENESS_HISTORY_SIZE:
                trk.liveness_history.pop(0)
            if len(trk.confidence_history) > config.LIVENESS_HISTORY_SIZE:
                trk.confidence_history.pop(0)
            trk.spoof_hold_until = now + config.LIVENESS_SPOOF_COOLDOWN_SECONDS
            self._reset_track_verification(
                trk,
                analysis.get("reject_reason", "early_reject"),
                keep_spoof_hold=True,
                preserve_state=True,
            )
            tracker.record_liveness_event("liveness_early_reject")
            return "spoof", conf

        trk.liveness_history.append((label, conf))
        trk.confidence_history.append(conf)
        if len(trk.liveness_history) > config.LIVENESS_HISTORY_SIZE:
            trk.liveness_history.pop(0)
        if len(trk.confidence_history) > config.LIVENESS_HISTORY_SIZE:
            trk.confidence_history.pop(0)

        motion_px = self._update_track_motion_history(trk, liveness_box)
        trk.brightness_history.append(float(analysis.get("mean_brightness", 0.0)))
        trk.contrast_history.append(float(analysis.get("contrast_proxy", 0.0)))
        trk.screen_history.append(bool(analysis.get("screen_suspicious", False)))
        if len(trk.brightness_history) > config.LIVENESS_HISTORY_SIZE:
            trk.brightness_history.pop(0)
        if len(trk.contrast_history) > config.LIVENESS_HISTORY_SIZE:
            trk.contrast_history.pop(0)
        if len(trk.screen_history) > config.LIVENESS_HISTORY_SIZE:
            trk.screen_history.pop(0)

        score_std = _safe_std(trk.confidence_history)
        if score_std > config.LIVENESS_SCORE_STD_THRESHOLD:
            tracker.record_liveness_event("liveness_unstable")
            if config.DEBUG_MODE:
                logger.debug(
                    "Track #%d unstable liveness scores: std=%.4f threshold=%.4f",
                    trk.track_id,
                    score_std,
                    config.LIVENESS_SCORE_STD_THRESHOLD,
                )
            return "uncertain", conf

        verification_age = now - float(trk.verification_started_at or now)
        if verification_age > config.LIVENESS_MAX_VERIFICATION_TIMEOUT_SECONDS:
            tracker.record_liveness_event("liveness_timeout")
            self._reset_track_verification(trk, "verification_timeout")
            trk.verification_started_at = now
            return "uncertain", conf

        # Check spoof hold temporal override
        if now < trk.spoof_hold_until:
            state = "spoof"
            score = max(conf, config.LIVENESS_SPOOF_CONFIDENCE_MIN)
            tracker.record_liveness_event("temporal_override")
        # Check for strong spoof signal
        elif self._is_spoof_strong(label, conf):
            state = "spoof"
            score = conf
            trk.spoof_hold_until = now + config.LIVENESS_SPOOF_COOLDOWN_SECONDS
            self._reset_track_verification(
                trk,
                "spoof_detected",
                keep_spoof_hold=True,
                preserve_state=True,
            )
        elif analysis.get("screen_suspicious") and self._screen_heuristics_allowed(trk):
            state = "spoof"
            score = conf
            trk.spoof_hold_until = now + config.LIVENESS_SPOOF_COOLDOWN_SECONDS
            tracker.record_liveness_event("screen_spoof_suspected")
            self._reset_track_verification(
                trk,
                "screen_spoof",
                keep_spoof_hold=True,
                preserve_state=True,
            )
        # Otherwise, use temporal voting on history
        else:
            state, score = self._decide_liveness_from_history(trk)

        if state == "uncertain" and config.LIVENESS_WEIGHTED_DECISION_ENABLED:
            blink_score = 1.0 if getattr(trk, "blink_count", 0) > 0 else 0.0
            motion_score = min(1.0, motion_px / max(1.0, config.LIVENESS_FACE_MOTION_MIN_PIXELS))
            screen_penalty = 0.25 if analysis.get("screen_suspicious") and self._screen_heuristics_allowed(trk) else 0.0
            weighted = self._weighted_liveness_score(conf, blink_score, motion_score, screen_penalty)
            if weighted >= config.LIVENESS_WEIGHTED_ACCEPT_THRESHOLD and conf >= config.LIVENESS_STRICT_THRESHOLD:
                state = "real"
                score = weighted

        # Map decision to internal label
        if state == "real":
            trk.liveness = (1, score)
        elif state == "spoof":
            trk.liveness = (0, score)
        elif label == -1:
            trk.liveness = (-1, score)
        else:
            trk.liveness = (2, score)

        return state, score

    def _is_real_fast(self, label: int, conf: float) -> bool:
        """Fast path: Mark real if label=1 and confidence high (≥ strict threshold)."""
        return (
            label == 1
            and conf >= config.LIVENESS_STRICT_THRESHOLD
        )

    def _is_spoof_strong(self, label: int, conf: float) -> bool:
        """Strong spoof signal: label∈{0,2} and confidence high (≥0.85)."""
        return (
            label in (0, 2)
            and conf >= config.LIVENESS_STRONG_SPOOF_CONFIDENCE
        )

    def _decide_liveness_from_history(self, trk: FaceTrack) -> tuple[str, float]:
        """Use temporal voting on liveness history to make decision.
        
        Requires at least config.LIVENESS_MIN_HISTORY valid frames, stable
        confidence across the rolling window, and behavioral evidence.
        Returns (state, score) where state ∈ {"real", "spoof", "uncertain"}.
        """
        latest_label, latest_conf = trk.liveness_history[-1]
        
        # Filter out error markers
        valid_history = [
            (l, c) for l, c in trk.liveness_history if l != -1
        ]

        min_hist = max(1, config.LIVENESS_MIN_HISTORY)
        if len(valid_history) < min_hist:
            # Not enough history; return uncertain
            return "uncertain", latest_conf

        if (time.monotonic() - float(trk.verification_started_at or time.monotonic())) < config.LIVENESS_DECISION_DELAY_SECONDS:
            return "uncertain", latest_conf

        # Count votes: real vs spoof at different confidence thresholds
        real_votes = [
            c for l, c in valid_history
            if l == 1 and c >= config.LIVENESS_STRICT_THRESHOLD
        ]
        spoof_votes = [
            c for l, c in valid_history
            if l in (0, 2) and c >= config.LIVENESS_SPOOF_CONFIDENCE_MIN
        ]
        weak_spoof_votes = [
            c for l, c in valid_history
            if l in (0, 2)  # No confidence threshold
        ]

        total_valid = len(valid_history)
        real_ratio = len(real_votes) / total_valid
        spoof_ratio = len(spoof_votes) / total_valid
        weak_spoof_ratio = len(weak_spoof_votes) / total_valid

        score_std = _safe_std([c for _, c in valid_history])
        if score_std > config.LIVENESS_SCORE_STD_THRESHOLD:
            return "uncertain", latest_conf

        # Get blink evidence (blink count > 0 indicates natural movement/real face)
        has_blink_evidence = getattr(trk, "blink_count", 0) > 0
        has_motion_evidence = not self._track_motion_low(trk)

        # Decide based on vote ratios
        if real_ratio >= config.LIVENESS_REAL_VOTE_RATIO and real_votes and (has_blink_evidence or has_motion_evidence):
            # Majority voted real
            avg_score = float(sum(real_votes) / len(real_votes))
            if avg_score >= config.LIVENESS_STRICT_THRESHOLD:
                return "real", avg_score
            return "uncertain", avg_score
        elif spoof_ratio >= config.LIVENESS_SPOOF_VOTE_RATIO and spoof_votes:
            # Majority voted spoof (strong confidence)
            avg_score = float(sum(spoof_votes) / len(spoof_votes))
            return "spoof", avg_score
        elif (
            weak_spoof_ratio >= config.LIVENESS_SPOOF_VOTE_RATIO
            and weak_spoof_votes
            and (sum(weak_spoof_votes) / len(weak_spoof_votes))
            >= config.LIVENESS_SPOOF_WEAK_CONFIDENCE_MIN
        ):
            # Majority voted spoof even with weak confidence threshold
            avg_score = float(sum(weak_spoof_votes) / len(weak_spoof_votes))
            return "spoof", avg_score
        elif config.LIVENESS_WEIGHTED_DECISION_ENABLED:
            blink_score = 1.0 if has_blink_evidence else 0.0
            motion_score = 1.0 if has_motion_evidence else 0.0
            screen_penalty = 0.25 if any(trk.screen_history[-3:]) and self._screen_heuristics_allowed(trk) else 0.0
            weighted = self._weighted_liveness_score(latest_conf, blink_score, motion_score, screen_penalty)
            if weighted >= config.LIVENESS_WEIGHTED_ACCEPT_THRESHOLD:
                return "real", weighted
            return "uncertain", weighted
        else:
            # No clear verdict
            return "uncertain", latest_conf



    def _set_track_liveness_state(self, trk: FaceTrack, state: str, score: float):
        prev_state = trk.liveness_state
        trk.liveness_state = state

        if state == "real":
            trk.is_spoof = False
            trk.spoof_hold_until = 0.0
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
        except (RuntimeError, ValueError) as exc:
            logger.debug("PPE detection error: %s", exc)
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
        """Run encoding+recognition for a track with multi-frame smoothing.

        Instead of making a recognition decision on each individual
        frame's embedding, embeddings are accumulated in
        ``trk.embedding_history`` and averaged once enough frames have
        been collected (``SMOOTHING_MIN_FRAMES``).  The smoothed
        embedding is then matched against the cache.

        Blink detection via EAR is also evaluated here when landmarks
        are available.
        """
        if liveness_conf < (config.LIVENESS_CONFIDENCE_THRESHOLD + config.LIVENESS_NO_ENCODE_MARGIN):
            tracker.record_liveness_event("no_encode_guard")
            trk.is_unknown = True
            trk.state = "liveness_pending"
            return

        if liveness_conf < config.LIVENESS_STRICT_THRESHOLD:
            tracker.record_liveness_event("strict_liveness_guard")
            trk.is_unknown = True
            trk.state = "liveness_pending"
            return

        # Backward-compatible defaults for lightweight test doubles / legacy tracks.
        if not hasattr(trk, "embedding_history"):
            trk.embedding_history = []
        if not hasattr(trk, "candidate_student_id"):
            trk.candidate_student_id = None
        if not hasattr(trk, "candidate_name"):
            trk.candidate_name = ""
        if not hasattr(trk, "candidate_hits"):
            trk.candidate_hits = 0
        if not hasattr(trk, "candidate_best_confidence"):
            trk.candidate_best_confidence = 0.0
        if not hasattr(trk, "identity_prediction_buffer"):
            trk.identity_prediction_buffer = []
        if not hasattr(trk, "student_metadata"):
            trk.student_metadata = {}
        if not hasattr(trk, "ear_history"):
            trk.ear_history = []
        if not hasattr(trk, "blink_count"):
            trk.blink_count = 0
        if not hasattr(trk, "blink_frames_below"):
            trk.blink_frames_below = 0

        # Refresh 5-point landmarks for blink/EAR logic, but not every cycle.
        # The landmark extraction ALSO returns an embedding from the same
        # InsightFace call, so we reuse it to avoid a redundant encode pass.
        # PERF: Always extract on first call (landmarks_5 is None) even when
        # blink detection is off, because the pre-extracted embedding saves
        # us a separate InsightFace call in the encoding step below.
        now = time.monotonic()
        pre_extracted_embedding = None
        need_landmark_refresh = (
            getattr(trk, "landmarks_5", None) is None
            or (
                config.BLINK_DETECTION_ENABLED
                and (now - getattr(trk, "landmarks_5_updated_at", 0.0))
                >= config.BLINK_LANDMARK_REFRESH_SECONDS
            )
        )
        if need_landmark_refresh and config.EMBEDDING_BACKEND == "arcface":
            new_landmarks, pre_extracted_embedding = _extract_landmarks_5_from_bbox(raw_frame, raw_bbox_xywh)
            if new_landmarks is not None:
                trk.landmarks_5 = new_landmarks
                trk.landmarks_5_updated_at = now

        # -- Blink detection (runs every recognition cycle) --
        if (
            config.BLINK_DETECTION_ENABLED
            and getattr(trk, "landmarks_5", None) is not None
        ):
            from vision.anti_spoofing import (
                compute_ear_from_5point,
                update_blink_state,
            )
            ear = compute_ear_from_5point(trk.landmarks_5)
            trk.ear_history, trk.blink_count, trk.blink_frames_below = (
                update_blink_state(
                    ear,
                    trk.ear_history,
                    trk.blink_count,
                    trk.blink_frames_below,
                )
            )

        # -- Encoding --
        # Use the pre-extracted embedding from the landmark call if available.
        # This is the KEY performance optimisation: one InsightFace call
        # instead of two (landmark + encode).
        if pre_extracted_embedding is not None:
            encoding = pre_extracted_embedding
            quality_reason = ""
        else:
            landmarks = getattr(trk, "landmarks_5", None)
            encoding, quality_reason = _recognition_module().encode_face_with_reason(
                raw_frame, raw_bbox_xywh, landmarks,
            )
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

        # -- Multi-frame embedding accumulation --
        trk.embedding_history.append(encoding)
        if len(trk.embedding_history) > config.SMOOTHING_WINDOW:
            trk.embedding_history.pop(0)

        # Not enough frames accumulated yet — show progress
        smoothing_min = max(1, config.SMOOTHING_MIN_FRAMES)
        if len(trk.embedding_history) < smoothing_min:
            trk.is_unknown = True
            trk.state = "liveness_pending"
            trk.quality_reason = (
                f"Accumulating frames ({len(trk.embedding_history)}/{smoothing_min})"
            )
            return

        # Compute smoothed (averaged + L2-normalised) embedding
        smoothed = _compute_smoothed_embedding(trk.embedding_history)
        if smoothed is None:
            trk.is_unknown = True
            trk.state = "liveness_pending"
            return
        trk.smoothed_embedding = smoothed

        result, cache_hit = self._get_track_cached_result(trk.track_id)
        if not cache_hit:
            recognition_start = time.perf_counter()
            # -- Image quality for dynamic threshold --
            from vision.preprocessing import assess_image_quality as _assess_quality
            x, y, w, h = raw_bbox_xywh
            fh, fw = raw_frame.shape[:2]
            roi = raw_frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
            image_quality = _assess_quality(roi) if roi.size > 0 else None

            candidate_student_ids = []
            if trk.candidate_student_id is not None:
                candidate_student_ids.append(trk.candidate_student_id)
            for pred in trk.identity_prediction_buffer:
                sid = pred.get("student_id")
                if sid is not None:
                    candidate_student_ids.append(sid)
            if candidate_student_ids:
                # Preserve order while deduplicating.
                candidate_student_ids = list(dict.fromkeys(candidate_student_ids))

            result = _face_engine_module().recognize_face(
                smoothed,
                ppe_state=ppe_state,
                ppe_confidence=ppe_confidence,
                image_quality=image_quality,
                candidate_student_ids=candidate_student_ids,
            )
            self._record_stage_time("recognition", recognition_start)
            self._set_track_cached_result(trk.track_id, result)
        if result is not None:
            student_id, name, confidence = result

            window = max(3, int(config.RECOGNITION_STABILITY_WINDOW))
            required_hits = max(
                1,
                int(
                    min(
                        window,
                        max(config.RECOGNITION_STABILITY_MIN_HITS, config.RECOGNITION_CONFIRM_FRAMES),
                    )
                ),
            )
            trk.identity_prediction_buffer.append(
                {
                    "student_id": student_id,
                    "name": name,
                    "confidence": float(confidence),
                }
            )
            if len(trk.identity_prediction_buffer) > window:
                trk.identity_prediction_buffer.pop(0)

            counts: dict = {}
            confidence_buckets: dict = {}
            names: dict = {}
            for pred in trk.identity_prediction_buffer:
                sid = pred["student_id"]
                counts[sid] = counts.get(sid, 0) + 1
                confidence_buckets.setdefault(sid, []).append(float(pred["confidence"]))
                names[sid] = pred["name"]

            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            best_student_id, best_hits = ranked[0]
            best_name = names.get(best_student_id, name)
            best_confidence = max(confidence_buckets.get(best_student_id, [float(confidence)]))

            second_hits = ranked[1][1] if len(ranked) > 1 else 0
            if best_hits == second_hits and len(ranked) > 1:
                trk.is_unknown = True
                trk.state = "liveness_pending"
                trk.quality_reason = (
                    "Unstable identity predictions across recent frames"
                )
                return

            if best_hits < required_hits:
                trk.is_unknown = True
                trk.state = "liveness_pending"
                trk.quality_reason = (
                    f"Confirming identity ({best_hits}/{required_hits}) in recent frames"
                )
                return

            final_confidence = float(best_confidence)
            trk.identity = (best_student_id, best_name, final_confidence)
            # Cache student metadata to avoid repeated DB lookups in _handle_recognized
            try:
                student_doc = database.get_student_by_id(best_student_id)
                if student_doc:
                    trk.student_metadata = {
                        "registration_number": student_doc.get("registration_number", ""),
                        "semester": student_doc.get("semester", ""),
                        "section": student_doc.get("section", ""),
                    }
            except Exception as exc:
                logger.debug("Failed to cache student metadata for %s: %s", student_id, exc)
                trk.student_metadata = {}

            trk.is_unknown = False
            trk.state = "recognized"
            trk.quality_reason = ""
            trk.candidate_student_id = None
            trk.candidate_name = ""
            trk.candidate_hits = 0
            trk.candidate_best_confidence = 0.0
            trk.identity_prediction_buffer = []
            logger.info(
                "Track #%d: RECOGNIZED %s (confidence=%.4f, smoothed_frames=%d, blinks=%d)",
                trk.track_id, best_name, final_confidence,
                len(trk.embedding_history),
                getattr(trk, "blink_count", 0),
            )
            self._handle_recognized(
                best_student_id, best_name, final_confidence, liveness_conf, frame,
                track=trk,
                encoding=smoothed,  # use the smoothed embedding for incremental learning
                ppe_state=ppe_state,
                ppe_confidence=ppe_confidence,
            )
            return

        trk.candidate_student_id = None
        trk.candidate_name = ""
        trk.candidate_hits = 0
        trk.candidate_best_confidence = 0.0
        trk.identity_prediction_buffer = []
        trk.is_unknown = True
        trk.state = "liveness_pending"
        trk.quality_reason = "Face not confidently matched to a single student."
        self._last_unknown_track_id = trk.track_id
        self._last_unknown_confidence = float(getattr(trk, "candidate_best_confidence", 0.0))
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
        track=None,
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

        # Use cached metadata from track to avoid DB lookup on every recognition
        student_meta = {}
        if track and track.student_metadata:
            student_meta = track.student_metadata
        else:
            # Fallback: fetch if cache not available (shouldn't happen in normal flow)
            student_doc = database.get_student_by_id(student_id)
            if student_doc:
                student_meta = {
                    "registration_number": student_doc.get("registration_number", ""),
                    "semester": student_doc.get("semester", ""),
                    "section": student_doc.get("section", ""),
                }

        active_session = self._get_active_session_for_camera()
        if active_session is None:
            with self._seen_lock:
                self._seen.pop(student_id, None)
            tracker.record_recognition(True, False)
            self._push_event({
                "name": name,
                "status": "session_inactive",
                "error": "No active attendance session for this camera",
                "confidence": round(confidence, 4),
                "liveness_confidence": round(liveness_conf, 4),
                "attendance_marked": False,
                "camera_id": str(self._source),
                **student_meta,
            })
            return

        session_id = active_session.get("_id")
        try:
            marked = database.mark_attendance(
                student_id,
                confidence,
                session_id=session_id,
            )
            if marked and session_id is not None:
                database.touch_attendance_session(session_id)
        except RuntimeError as exc:
            # Circuit breaker is open or other fatal DB error
            logger.error(
                "Failed to mark attendance due to database circuit breaker or connection error: %s",
                exc,
            )
            marked = False
            self._push_event({
                "name": name,
                "status": "error",
                "error": "Database unavailable; attendance not marked",
                "confidence": round(confidence, 4),
                "liveness_confidence": round(liveness_conf, 4),
                "attendance_marked": False,
                **student_meta,
            })
            return

        self._push_event({
            "name": name,
            "status": "marked",
            "confidence": round(confidence, 4),
            "liveness_confidence": round(liveness_conf, 4),
            "ppe_state": ppe_state,
            "ppe_confidence": round(float(ppe_confidence), 4),
            "attendance_marked": marked,
            "session_id": str(session_id) if session_id is not None else None,
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
                        # Use cosine similarity for duplicate detection
                        query = encoding.astype(np.float32).flatten()
                        q_norm = np.linalg.norm(query)
                        if q_norm > 0:
                            query = query / q_norm
                        sims = flat_enc[student_rows] @ query
                        max_sim = float(np.max(sims))
                        # Skip if very similar embedding already exists (cosine > 0.95)
                        if max_sim > 0.95:
                            logger.debug(
                                "Incremental learning skipped for %s (duplicate: cosine_sim=%.4f > 0.95)",
                                name,
                                max_sim,
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
                self._track_identity_cache = {
                    tid: entry
                    for tid, entry in self._track_identity_cache.items()
                    if float(entry.get("expires_at", 0.0)) > time.monotonic()
                }

            surviving: list[FaceTrack] = []
            for trk in self._tracks:
                trk.update(frame)
                if trk.frames_missing <= config.TRACK_EXPIRATION_FRAMES:
                    surviving.append(trk)
                else:
                    trk.state = "expired"
                    self._reset_track_verification(trk, "track_expired")
                    self._track_identity_cache.pop(trk.track_id, None)
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
                detection_start = time.perf_counter()
                run_detection = motion or (
                    self._frame_count % config.NO_MOTION_DETECTION_INTERVAL == 0
                )
                if not self._tracks:
                    run_detection = True

                if run_detection:
                    # Cap new detections to respect PERF_MAX_FACES
                    remaining_slots = max(0, config.PERF_MAX_FACES - len(self._tracks))
                    new_boxes, matched_track_indices = detect_and_associate_detailed(
                        frame,
                        self._tracks,
                        max_new_faces=remaining_slots,
                    )
                    self._record_stage_time("detection", detection_start)
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
                            self._reset_track_verification(trk, "detector_pruned")
                            self._track_identity_cache.pop(trk.track_id, None)
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

                        if trk.spoof_hold_until and time.monotonic() < trk.spoof_hold_until:
                            trk.is_spoof = True
                            trk.state = "spoof"
                            continue

                        # -- Performance gating: skip expensive ops on most cycles --
                        trk.antispoof_cycle_count += 1
                        trk.recognition_cycle_count += 1

                        run_antispoof = (
                            trk.antispoof_cycle_count
                            % max(1, config.PERF_ANTISPOOF_INTERVAL) == 0
                        )
                        run_recognition = (
                            trk.recognition_cycle_count
                            % max(1, config.PERF_RECOGNITION_INTERVAL) == 0
                        )

                        raw_box = self._to_raw_bbox(trk.bbox, sx, sy)

                        if run_antispoof:
                            liveness_start = time.perf_counter()
                            state, liveness_conf = self._evaluate_track_liveness(
                                trk,
                                raw,
                                raw_box,
                            )
                            self._record_stage_time("liveness", liveness_start)
                            self._set_track_liveness_state(trk, state, liveness_conf)
                        else:
                            # Reuse cached liveness state
                            state = trk.liveness_state
                            liveness_conf = trk.liveness[1] if trk.liveness else 0.0

                        if state == "real" and run_recognition:
                            ppe_start = time.perf_counter()
                            ppe_state, ppe_confidence = self._evaluate_track_ppe(
                                trk,
                                raw,
                                raw_box,
                            )
                            self._record_stage_time("ppe", ppe_start)
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
                if config.PERF_FRAME_SCALE < 1.0:
                    debug_lines.append(f"SCALE: {config.PERF_FRAME_SCALE:.0%}")
                debug_lines.append(f"MAX_FACES: {config.PERF_MAX_FACES}")
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
            self._process_frame_times.append(elapsed)

            if self._frame_count % 30 == 0:
                _emit_event("metrics_update", tracker.metrics())

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.PERF_JPEG_QUALITY]
            _, jpeg = cv2.imencode(".jpg", frame, encode_params)
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


class CameraManager:
    """Manage camera lifecycle and diagnostics across sources."""

    def __init__(self):
        self._cameras: dict[int, Camera] = {}
        self._camera_viewers: dict[int, int] = {}
        self._lock = threading.Lock()

    def get(self, source: int = 0) -> Camera:
        with self._lock:
            if source not in self._cameras:
                cam = Camera(source)
                cam.start()
                self._cameras[source] = cam
            return self._cameras[source]

    def get_if_running(self, source: int = 0) -> Camera | None:
        with self._lock:
            return self._cameras.get(source)

    def acquire_stream(self, source: int = 0) -> Camera:
        with self._lock:
            cam = self._cameras.get(source)
            if cam is None:
                cam = Camera(source)
                cam.start()
                self._cameras[source] = cam
            self._camera_viewers[source] = self._camera_viewers.get(source, 0) + 1
            return cam

    def release_stream(self, source: int = 0):
        with self._lock:
            if source not in self._camera_viewers:
                return
            remaining = self._camera_viewers[source] - 1
            if remaining > 0:
                self._camera_viewers[source] = remaining
                return
            self._camera_viewers.pop(source, None)
            cam = self._cameras.pop(source, None)
            if cam is not None:
                cam.stop()

    def get_all(self) -> dict[int, Camera]:
        with self._lock:
            return dict(self._cameras)

    def release(self, source: int | None = None):
        with self._lock:
            if source is not None:
                self._camera_viewers.pop(source, None)
                cam = self._cameras.pop(source, None)
                if cam is not None:
                    cam.stop()
            else:
                self._camera_viewers.clear()
                for cam in self._cameras.values():
                    cam.stop()
                self._cameras.clear()

    def diagnostics(self) -> dict:
        with self._lock:
            cameras_snapshot = dict(self._cameras)
            viewers_snapshot = dict(self._camera_viewers)

        return {
            "active_cameras": len(cameras_snapshot),
            "viewers": viewers_snapshot,
            "cameras": {
                source: cam.diagnostics()
                for source, cam in cameras_snapshot.items()
            },
        }


_camera_manager = CameraManager()
_cameras = _camera_manager._cameras
_camera_viewers = _camera_manager._camera_viewers


def get_camera(source: int = 0) -> Camera:
    """Return a Camera instance for the given device index, creating it
    if necessary."""
    return _camera_manager.get(source)


def get_camera_if_running(source: int = 0) -> Camera | None:
    """Return a running Camera instance if available, else None."""
    return _camera_manager.get_if_running(source)


def acquire_camera_stream(source: int = 0) -> Camera:
    """Acquire a camera for active video streaming.

    The caller must pair this with ``release_camera_stream`` when the
    stream disconnects.
    """
    return _camera_manager.acquire_stream(source)


def release_camera_stream(source: int = 0):
    """Release one active stream viewer; stop camera when last viewer leaves."""
    _camera_manager.release_stream(source)


def get_all_cameras() -> dict[int, Camera]:
    """Return the dict of all active camera instances."""
    return _camera_manager.get_all()


def release_camera(source: int | None = None):
    """Release one camera (by index) or all cameras (if source is None)."""
    _camera_manager.release(source)


def get_camera_diagnostics() -> dict:
    """Return diagnostics for all managed cameras and stream viewers."""
    return _camera_manager.diagnostics()
