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
from anti_spoofing import check_liveness
from face_engine import encoding_cache, recognize_face, append_encoding
from overlay import draw_track_overlay
from performance import tracker
from pipeline import FaceTrack, detect_and_associate, detect_motion
from recognition import encode_face
from utils import setup_logging

logger = setup_logging()

# -- SocketIO reference (set by app.py after init) --------------------------
_socketio = None


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

        # -- Anti-spoof on per-face crop --
        liveness_frame = raw_frame if raw_frame is not None else frame
        liveness_box = raw_bbox_xywh if raw_bbox_xywh is not None else bbox_xywh

        if config.BYPASS_ANTISPOOF:
            # Bypass: treat every face as real
            liveness_label, liveness_conf = 1, 1.0
            logger.debug("Track #%d: anti-spoof BYPASSED by config.", tid)
        else:
            x, y, w, h = liveness_box
            fh, fw = liveness_frame.shape[:2]
            pad = int(max(w, h) * 0.2)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)
            face_crop = liveness_frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                liveness_label, liveness_conf = 0, 0.0
            else:
                liveness_label, liveness_conf = check_liveness(face_crop)

        if liveness_label == -1:
            logger.warning(
                "Track #%d: anti-spoof ERROR (proceeding in degraded mode). "
                "Will treat face as real assuming camera is operational.",
                tid,
            )
            liveness_label = 1
            liveness_conf = config.LIVENESS_CONFIDENCE_THRESHOLD

        trk.liveness = (liveness_label, liveness_conf)

        is_real = (
            liveness_label == 1
            and liveness_conf >= config.LIVENESS_CONFIDENCE_THRESHOLD
        )

        logger.debug(
            "Track #%d: liveness_label=%d liveness_conf=%.4f "
            "threshold=%.2f is_real=%s (0=spoof/no_face, 1=real, -1=error)",
            tid, liveness_label, liveness_conf,
            config.LIVENESS_CONFIDENCE_THRESHOLD, is_real,
        )

        if is_real:
            # -- Face encoding (with alignment) + recognition --
            encode_frame = raw_frame if raw_frame is not None else frame
            encode_box = raw_bbox_xywh if raw_bbox_xywh is not None else bbox_xywh
            encoding = encode_face(encode_frame, encode_box)
            if encoding is not None:
                logger.debug(
                    "Track #%d: encoding generated, shape=%s",
                    tid, encoding.shape,
                )
                result = recognize_face(encoding)
                if result is not None:
                    student_id, name, confidence = result
                    trk.identity = (student_id, name, confidence)
                    logger.info(
                        "Track #%d: RECOGNIZED %s (confidence=%.4f)",
                        tid, name, confidence,
                    )
                    self._handle_recognized(
                        student_id, name, confidence, liveness_conf, frame,
                        encoding=encoding,
                    )
                else:
                    trk.is_unknown = True
                    logger.debug(
                        "Track #%d: encoding produced but no match in cache "
                        "(threshold=%.4f)",
                        tid, config.RECOGNITION_THRESHOLD,
                    )
                    tracker.record_recognition(False, False)
                    self._save_unknown_snapshot(liveness_frame)
            else:
                trk.is_unknown = True
                logger.debug(
                    "Track #%d: encode_face returned None "
                    "(likely quality gate rejection or encoding failure)",
                    tid,
                )
                self._push_event({
                    "name": "Unknown",
                    "status": "quality_rejection",
                    "confidence": 0.0,
                    "details": "Face quality check failed (blurry, too dark, or too bright)",
                })
                tracker.record_recognition(False, False)
        else:
            if liveness_label == 0 and liveness_conf >= config.LIVENESS_CONFIDENCE_THRESHOLD:
                trk.is_spoof = True
                self._push_event({
                    "name": "Unknown",
                    "status": "spoof_detected",
                    "confidence": 0.0,
                    "liveness_confidence": round(liveness_conf, 4),
                })
                logger.warning(
                    "Track #%d: Spoof detected: liveness_label=%d "
                    "liveness_confidence=%.4f timestamp=%s",
                    tid, liveness_label, liveness_conf,
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                tracker.record_recognition(False, False)
            else:
                trk.is_unknown = True
                tracker.record_recognition(False, False)
                self._push_event({
                    "name": "Unknown",
                    "status": "liveness_uncertain",
                    "confidence": 0.0,
                    "liveness_confidence": round(liveness_conf, 4),
                })
                logger.debug(
                    "Track #%d: liveness uncertain (label=%d conf=%.4f) "
                    "— treating as unknown.",
                    tid, liveness_label, liveness_conf,
                )

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

        if (encoding is not None
                and confidence >= config.INCREMENTAL_LEARNING_CONFIDENCE):
            try:
                flat_enc, flat_idx, ids, _ = encoding_cache.get_flat()
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

                append_encoding(student_id, encoding)
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
                    logger.debug("Track #%d expired.", trk.track_id)
            self._tracks = surviving

            if self._frame_count % config.DETECTION_INTERVAL == 0:
                if config.BYPASS_MOTION_DETECTION:
                    run_detection = True
                else:
                    motion, self._prev_gray = detect_motion(self._prev_gray, frame)
                    run_detection = motion or (
                        self._frame_count % config.NO_MOTION_DETECTION_INTERVAL == 0
                    )
                if not self._tracks:
                    run_detection = True

                if run_detection:
                    new_boxes = detect_and_associate(frame, self._tracks)
                    if config.DEBUG_MODE and new_boxes:
                        logger.debug(
                            "Frame %d: %d new face(s) detected, %d active track(s)",
                            self._frame_count,
                            len(new_boxes),
                            len(self._tracks),
                        )
                    for box in new_boxes:
                        raw_box = (
                            int(box[0] * sx),
                            int(box[1] * sy),
                            int(box[2] * sx),
                            int(box[3] * sy),
                        )
                        trk = self._create_track(
                            frame,
                            box,
                            raw_frame=raw,
                            raw_bbox_xywh=raw_box,
                        )
                        self._tracks.append(trk)

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
                    f"Threshold: {config.RECOGNITION_THRESHOLD:.2f}",
                    f"Cache: {encoding_cache.size} students",
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


# ---------------------------------------------------------------------------
# Multi-camera management
# ---------------------------------------------------------------------------
_cameras: dict[int, Camera] = {}
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


def get_all_cameras() -> dict[int, Camera]:
    """Return the dict of all active camera instances."""
    with _cameras_lock:
        return dict(_cameras)


def release_camera(source: int | None = None):
    """Release one camera (by index) or all cameras (if source is None)."""
    global _cameras
    with _cameras_lock:
        if source is not None:
            cam = _cameras.pop(source, None)
            if cam is not None:
                cam.stop()
        else:
            for cam in _cameras.values():
                cam.stop()
            _cameras.clear()
