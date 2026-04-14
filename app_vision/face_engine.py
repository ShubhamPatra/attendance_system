"""
Face engine – encoding generation, in-memory cache, recognition.
"""

import threading

import cv2
import face_recognition
import numpy as np
import torch

import app_core.config as config
import app_core.database as database
from app_vision.pipeline import detect_faces_yunet
from app_vision.recognition import encode_face
from app_core.utils import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# GPU acceleration detection
# ---------------------------------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Face engine device: %s", _DEVICE)


# ---------------------------------------------------------------------------
# Encoding generation
# ---------------------------------------------------------------------------

def generate_encoding(image) -> np.ndarray:
    """Generate a 128-D face encoding from an image.

    *image* can be a numpy array (RGB) loaded via face_recognition.load_image_file
    or a file path string.

    Raises ValueError if no face or multiple faces are detected.
    """
    if isinstance(image, str):
        image = face_recognition.load_image_file(image)

    # Keep enrollment consistent with runtime path: detect + quality gate +
    # alignment + encoding from recognition.encode_face.
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        boxes = detect_faces_yunet(bgr_image, process_width=config.FRAME_PROCESS_WIDTH)
    except RuntimeError:
        boxes = []

    if len(boxes) == 1:
        encoding = encode_face(bgr_image, boxes[0])
        if encoding is not None:
            return encoding

    # Fallback for startup/tests where YuNet may not be initialised.
    locations = face_recognition.face_locations(image, model="hog")
    if len(locations) == 0:
        raise ValueError("No face detected in the image.")
    if len(locations) > 1:
        raise ValueError(
            f"Multiple faces detected ({len(locations)}). "
            "Please upload an image with exactly one face."
        )

    encodings = face_recognition.face_encodings(image, known_face_locations=locations)
    if not encodings:
        raise ValueError("Failed to generate face encoding from the image.")
    return encodings[0]


# ---------------------------------------------------------------------------
# Encoding cache (thread-safe)
# ---------------------------------------------------------------------------

class EncodingCache:
    """In-memory cache of all student face encodings from MongoDB.

    Each student may have multiple encodings (stored as a list).
    A flattened ``(N, 128)`` array and parallel student-index array are
    maintained for vectorised distance computation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ids: list = []
        self._names: list[str] = []
        self._encodings: list[list[np.ndarray]] = []  # list-of-lists
        # Flattened arrays for vectorised matching
        self._flat_encodings: np.ndarray | None = None  # (N, 128)
        self._flat_student_idx: np.ndarray | None = None  # (N,) → student index

    def _rebuild_flat(self):
        """Rebuild the flattened encoding matrix from per-student lists.

        Must be called while holding ``_lock``.
        """
        all_enc: list[np.ndarray] = []
        all_idx: list[int] = []
        for idx, enc_list in enumerate(self._encodings):
            for enc in enc_list:
                all_enc.append(enc)
                all_idx.append(idx)
        if all_enc:
            self._flat_encodings = np.array(all_enc, dtype=np.float64)
            self._flat_student_idx = np.array(all_idx, dtype=np.intp)
        else:
            self._flat_encodings = None
            self._flat_student_idx = None

    def load(self):
        """Load / reload all encodings from the database."""
        rows = database.get_student_encodings()
        with self._lock:
            self._ids = [r[0] for r in rows]
            self._names = [r[1] for r in rows]
            self._encodings = [r[2] for r in rows]  # each r[2] is a list
            self._rebuild_flat()
        logger.info("Encoding cache loaded: %d students.", len(rows))

    def refresh(self):
        """Alias for load — call after new registration."""
        self.load()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)

    def get_all(self) -> tuple[list, list[str], list[list[np.ndarray]]]:
        """Return (ids, names, encodings) snapshot."""
        with self._lock:
            return list(self._ids), list(self._names), list(self._encodings)

    def get_flat(self) -> tuple[np.ndarray | None, np.ndarray | None, list, list[str]]:
        """Return ``(flat_encodings, flat_student_idx, ids, names)`` snapshot
        for vectorised matching."""
        with self._lock:
            flat_enc = self._flat_encodings
            flat_idx = self._flat_student_idx
            return (
                flat_enc.copy() if flat_enc is not None else None,
                flat_idx.copy() if flat_idx is not None else None,
                list(self._ids),
                list(self._names),
            )

    def add_encoding_to_student(self, student_id, encoding: np.ndarray) -> None:
        """Add a new encoding to an existing student's in-memory list.

        This provides an immediate cache update without a full database
        reload.  If the number of encodings for the student exceeds
        ``config.MAX_ENCODINGS_PER_STUDENT``, the oldest encoding
        (index 0) is dropped.

        Parameters
        ----------
        student_id :
            The ``bson.ObjectId`` of the student whose encoding list
            should be updated.
        encoding : np.ndarray
            A 128-D face encoding vector.

        Raises
        ------
        ValueError
            If *student_id* is not present in the cache.
        """
        with self._lock:
            try:
                idx = self._ids.index(student_id)
            except ValueError:
                raise ValueError(
                    f"Student {student_id} not found in encoding cache."
                )

            self._encodings[idx].append(encoding)

            # Enforce the per-student encoding cap
            if len(self._encodings[idx]) > config.MAX_ENCODINGS_PER_STUDENT:
                self._encodings[idx] = self._encodings[idx][-config.MAX_ENCODINGS_PER_STUDENT:]

            self._rebuild_flat()


# Singleton cache instance
encoding_cache = EncodingCache()


def _display_confidence_from_distance(distance: float) -> float:
    """Map embedding distance to a smoother UI confidence score.

    This calibration improves readability of confidence shown to operators,
    while recognition acceptance still uses the raw ``1 - distance`` score.
    """
    alpha = max(0.1, float(config.RECOGNITION_CONFIDENCE_ALPHA))
    return float(np.clip(np.exp(-alpha * (distance ** 2)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognize_face(
    face_encoding: np.ndarray,
    threshold: float | None = None,
    ppe_state: str = "none",
    ppe_confidence: float = 0.0,
) -> tuple | None:
    """Compare *face_encoding* against the cache using vectorised NumPy.

    All stored encodings are pre-flattened into a single ``(N, 128)``
    matrix so that distance computation is a single
    ``np.linalg.norm(...)`` call.  For each student the **minimum**
    distance across all their encodings is used.

    Returns ``(student_id, name, confidence)`` for the best match
    whose distance is below *threshold*, or ``None`` if unknown.

    *confidence* is defined as ``1 − distance`` (higher = better).
    """
    if threshold is None:
        threshold = config.RECOGNITION_THRESHOLD

    min_conf = max(0.0, min(1.0, config.RECOGNITION_MIN_CONFIDENCE))
    min_gap = max(0.0, config.RECOGNITION_MIN_DISTANCE_GAP)

    occluded = (
        ppe_state in ("mask", "cap", "both")
        and ppe_confidence >= config.PPE_MIN_CONFIDENCE
    )
    if occluded:
        threshold = min(1.0, threshold + config.OCCLUDED_RECOGNITION_THRESHOLD_DELTA)
        min_gap = max(min_gap, config.OCCLUDED_MIN_DISTANCE_GAP)
        min_conf = max(min_conf, config.OCCLUDED_MIN_CONFIDENCE)

    flat_enc, flat_idx, ids, names = encoding_cache.get_flat()
    if flat_enc is None or len(flat_enc) == 0:
        logger.debug("recognize_face: encoding cache is empty — returning None")
        return None

    # Vectorised distance: one operation over all encodings
    distances = np.linalg.norm(flat_enc - face_encoding, axis=1)

    # Per-student minimum distance
    num_students = len(ids)
    min_dists = np.full(num_students, np.inf)
    np.minimum.at(min_dists, flat_idx, distances)

    best_idx = int(np.argmin(min_dists))
    best_dist = float(min_dists[best_idx])

    # Second-best student distance is used as an ambiguity guard.
    if len(min_dists) > 1:
        sorted_dists = np.sort(min_dists)
        second_best_dist = float(sorted_dists[1])
    else:
        second_best_dist = float("inf")

    distance_gap = second_best_dist - best_dist
    raw_confidence = float(1.0 - best_dist)
    display_confidence = _display_confidence_from_distance(best_dist)

    logger.debug(
        "recognize_face: best_match=%s best_dist=%.4f second_best=%.4f "
        "gap=%.4f threshold=%.4f conf_raw=%.4f conf_ui=%.4f min_conf=%.4f min_gap=%.4f "
        "ppe_state=%s ppe_conf=%.4f occluded=%s "
        "cache_students=%d cache_encodings=%d decision=%s",
        names[best_idx], best_dist, second_best_dist, distance_gap, threshold,
        raw_confidence, display_confidence, min_conf, min_gap,
        ppe_state, ppe_confidence, occluded,
        num_students, len(flat_enc),
        "MATCH"
        if (
            best_dist <= threshold
            and raw_confidence >= min_conf
            and distance_gap >= min_gap
        )
        else "REJECT",
    )

    if best_dist > threshold:
        return None

    if raw_confidence < min_conf:
        return None

    if distance_gap < min_gap:
        return None

    return ids[best_idx], names[best_idx], display_confidence


# ---------------------------------------------------------------------------
# Incremental learning
# ---------------------------------------------------------------------------

def append_encoding(student_id, new_encoding: np.ndarray) -> bool:
    """Persist a new encoding for an existing student and update the cache.

    This supports *incremental learning* — the system can accumulate
    additional face encodings over time to improve recognition accuracy
    under varying conditions (lighting, angle, etc.).

    The encoding is first written to the database via
    ``database.append_student_encoding`` and then incrementally reflected
    in the in-memory ``encoding_cache`` to avoid expensive full reloads.

    Parameters
    ----------
    student_id : bson.ObjectId
        The MongoDB ``_id`` of the student document.
    new_encoding : np.ndarray
        A 128-D face encoding vector to append.

    Returns
    -------
    bool
        ``True`` if the encoding was successfully appended and the
        cache refreshed, ``False`` if any error occurred.
    """
    try:
        database.append_student_encoding(student_id, new_encoding)
        encoding_cache.add_encoding_to_student(student_id, new_encoding)
        logger.info(
            "Appended new encoding for student %s and updated cache.",
            student_id,
        )
        return True
    except Exception:
        logger.exception(
            "Failed to append encoding for student %s.", student_id
        )
        return False
