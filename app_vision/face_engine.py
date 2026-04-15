"""
Face engine – embedding backends, in-memory cache, and recognition.

Supports interchangeable embedding backends via the ``EmbeddingBackend``
protocol.  The default backend is **ArcFace** (InsightFace), producing
512-D L2-normalised embeddings matched with cosine similarity.

The legacy **dlib** backend (128-D, Euclidean distance) is still available
by setting ``EMBEDDING_BACKEND=dlib`` in the environment.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
import torch

import app_core.config as config
import app_core.database as database
from app_vision.pipeline import detect_faces_yunet
from app_vision.recognition import encode_face
from app_core.performance import tracker
from app_vision.preprocessing import (
    assess_image_quality,
    compute_dynamic_threshold,
    preprocess_face,
)
from app_core.utils import setup_logging

logger = setup_logging()

# ---------------------------------------------------------------------------
# GPU acceleration detection / embedding backend selection
# ---------------------------------------------------------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Face engine device: %s", _DEVICE)


def _get_onnxruntime_providers() -> list[str]:
    """Return preferred ONNX Runtime providers, with safe CPU fallback."""
    if not config.ENABLE_GPU_PROVIDERS:
        return ["CPUExecutionProvider"]

    preferred = [
        provider.strip()
        for provider in str(config.ONNXRT_PROVIDER_PRIORITY).split(",")
        if provider.strip()
    ]
    if not preferred:
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if _DEVICE != "cuda":
        return [provider for provider in preferred if provider != "CUDAExecutionProvider"] or ["CPUExecutionProvider"]
    return preferred


class EmbeddingBackend(Protocol):
    """Interface for interchangeable face embedding backends."""

    name: str

    def generate(self, image) -> np.ndarray:
        """Generate a face embedding from *image*."""


# ---------------------------------------------------------------------------
# ArcFace (InsightFace) backend
# ---------------------------------------------------------------------------

class ArcFaceEmbeddingBackend:
    """InsightFace ArcFace backend producing 512-D L2-normalised embeddings."""

    name: str = "arcface"

    def __init__(self):
        self._app = None  # Lazy-loaded FaceAnalysis
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        """Lazy-load InsightFace model on first use."""
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            try:
                from insightface.app import FaceAnalysis

                ctx_id = 0 if _DEVICE == "cuda" else -1
                det_size = config.ARCFACE_DET_SIZE
                providers = _get_onnxruntime_providers()

                def _load_with_providers(provider_list: list[str]):
                    app = FaceAnalysis(
                        name=config.ARCFACE_MODEL_NAME,
                        allowed_modules=["detection", "recognition"],
                        providers=provider_list,
                    )
                    app.prepare(
                        ctx_id=ctx_id,
                        det_size=(det_size, det_size),
                    )
                    return app

                try:
                    self._app = _load_with_providers(providers)
                    logger.info(
                        "InsightFace ArcFace model loaded: model=%s det_size=%d ctx_id=%d providers=%s",
                        config.ARCFACE_MODEL_NAME,
                        det_size,
                        ctx_id,
                        providers,
                    )
                except Exception as exc:
                    if providers == ["CPUExecutionProvider"]:
                        raise
                    logger.warning(
                        "InsightFace load failed with providers=%s; retrying on CPU: %s",
                        providers,
                        exc,
                    )
                    self._app = _load_with_providers(["CPUExecutionProvider"])
                    logger.info(
                        "InsightFace ArcFace model loaded on CPU fallback: model=%s det_size=%d ctx_id=%d",
                        config.ARCFACE_MODEL_NAME,
                        det_size,
                        ctx_id,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"InsightFace model loading failed: {exc}.  "
                    f"Install insightface and onnxruntime, or set "
                    f"EMBEDDING_BACKEND=dlib to use the legacy backend."
                ) from exc

    def generate(self, image) -> np.ndarray:
        """Generate a 512-D L2-normalised face embedding.

        *image* must be a BGR numpy array.  If it is a file path string
        it will be read with ``cv2.imread``.

        Raises ValueError if no face or multiple faces are detected.
        """
        self._ensure_loaded()

        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image file: {image}")

        # Preprocess for lighting normalisation
        image = preprocess_face(image)

        faces = self._app.get(image)
        if len(faces) == 0:
            raise ValueError("No face detected in the image.")
        if len(faces) > 1:
            # Pick the largest face by bounding box area
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            logger.debug(
                "Multiple faces detected (%d); using largest.", len(faces),
            )

        embedding = faces[0].normed_embedding  # 512-D, already L2-normalised
        return embedding.astype(np.float32)

    def generate_from_aligned(self, aligned_face_112: np.ndarray) -> np.ndarray:
        """Generate embedding from a pre-aligned 112×112 face chip.

        This skips InsightFace's internal face detection entirely and
        runs **only the recognition model**, making it ~50× faster than
        ``generate()`` on CPU.

        Parameters
        ----------
        aligned_face_112 : ndarray
            A 112×112 BGR face chip, already aligned to the ArcFace template.

        Returns
        -------
        ndarray
            512-D L2-normalised float32 embedding.
        """
        self._ensure_loaded()
        rec_model = None
        for task, model in self._app.models.items():
            if task == 'recognition':
                rec_model = model
                break
        if rec_model is None:
            raise RuntimeError("ArcFace recognition model not found in loaded modules.")

        # get_feat() handles preprocessing (blob creation) and ONNX inference
        embedding = rec_model.get_feat(aligned_face_112).flatten()
        # L2-normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def get_faces(self, image):
        """Run detection + recognition and return all InsightFace face objects.

        Useful for the pipeline when we need landmarks and embeddings
        simultaneously.
        """
        self._ensure_loaded()
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image file: {image}")
        image = preprocess_face(image)
        return self._app.get(image)

    def get_embedding_from_face(self, face) -> np.ndarray:
        """Extract the 512-D embedding from a pre-detected InsightFace face.

        Use this when you already have face objects from ``get_faces()``
        to avoid running detection again.
        """
        embedding = face.normed_embedding
        return embedding.astype(np.float32)

    def preload(self):
        """Eagerly load the model at startup instead of lazy-loading."""
        self._ensure_loaded()


# ---------------------------------------------------------------------------
# dlib (legacy) backend
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DlibEmbeddingBackend:
    """Legacy backend built on YuNet + dlib/face_recognition."""

    name: str = "dlib"

    def generate(self, image) -> np.ndarray:
        """Generate a 128-D face encoding using the dlib path."""
        try:
            import face_recognition
        except ImportError as exc:
            raise RuntimeError(
                "face_recognition / dlib not installed.  Install them or "
                "set EMBEDDING_BACKEND=arcface."
            ) from exc

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
# Backend registry
# ---------------------------------------------------------------------------

_arcface_backend = ArcFaceEmbeddingBackend()

_EMBEDDING_BACKENDS: dict[str, EmbeddingBackend] = {
    "arcface": _arcface_backend,
    "dlib": DlibEmbeddingBackend(),
}


def get_embedding_backend(name: str | None = None) -> EmbeddingBackend:
    """Return the configured embedding backend."""
    backend_name = (name or config.EMBEDDING_BACKEND or "arcface").strip().lower()
    backend = _EMBEDDING_BACKENDS.get(backend_name)
    if backend is None:
        raise ValueError(
            f"Unknown embedding backend '{backend_name}'. "
            f"Available: {', '.join(_EMBEDDING_BACKENDS.keys())}"
        )
    return backend


def get_embedding_backend_name() -> str:
    """Return the active embedding backend name."""
    return get_embedding_backend().name


def get_arcface_backend() -> ArcFaceEmbeddingBackend:
    """Return the ArcFace backend instance (for direct landmark access)."""
    return _arcface_backend


# ---------------------------------------------------------------------------
# Encoding generation
# ---------------------------------------------------------------------------

def generate_encoding(image) -> np.ndarray:
    """Generate a face embedding from an image.

    *image* can be a numpy array (RGB for dlib, BGR for ArcFace) loaded via
    ``face_recognition.load_image_file`` / ``cv2.imread``, or a file path
    string.

    Raises ValueError if no face or multiple faces are detected.
    """
    return get_embedding_backend().generate(image)


# ---------------------------------------------------------------------------
# Encoding cache (thread-safe)
# ---------------------------------------------------------------------------

class EncodingCache:
    """In-memory cache of all student face embeddings from MongoDB.

    Each student may have multiple encodings (stored as a list).
    A flattened ``(N, D)`` array and parallel student-index array are
    maintained for vectorised similarity computation, where D is the
    embedding dimensionality (512 for ArcFace, 128 for dlib).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ids: list = []
        self._names: list[str] = []
        self._encodings: list[list[np.ndarray]] = []  # list-of-lists
        # Flattened arrays for vectorised matching
        self._flat_encodings: np.ndarray | None = None  # (N, D)
        self._flat_student_idx: np.ndarray | None = None  # (N,) → student index
        # Map from student index to (start_row, end_row) in _flat_encodings
        self._student_flat_range: dict[int, tuple[int, int]] = {}
        self._student_prototypes: np.ndarray | None = None  # (S, D)
        self._prototype_student_idx: np.ndarray | None = None  # (S,) -> student index
        self._embedding_dim: int = config.EMBEDDING_DIM

    def _rebuild_flat(self):
        """Rebuild the flattened encoding matrix from per-student lists.

        Only encodings whose dimension matches the active backend's
        ``EMBEDDING_DIM`` are included.  Stale encodings (e.g. 128-D
        dlib vectors when running ArcFace 512-D) are silently skipped
        — run ``scripts/migrate_encodings.py`` to convert them.

        Must be called while holding ``_lock``.
        """
        target_dim = config.EMBEDDING_DIM
        all_enc: list[np.ndarray] = []
        all_idx: list[int] = []
        proto_enc: list[np.ndarray] = []
        proto_idx: list[int] = []
        skipped_students: list[str] = []
        self._student_flat_range.clear()
        
        for student_idx, enc_list in enumerate(self._encodings):
            matched = False
            start_row = len(all_enc)
            for enc in enc_list:
                if enc.shape[0] == target_dim:
                    all_enc.append(enc)
                    all_idx.append(student_idx)
                    matched = True
            
            if matched:
                self._student_flat_range[student_idx] = (start_row, len(all_enc))
                student_enc = np.array(all_enc[start_row:len(all_enc)], dtype=np.float32)
                proto = np.mean(student_enc, axis=0)
                proto_norm = np.linalg.norm(proto)
                if proto_norm > 0:
                    proto = proto / proto_norm
                proto_enc.append(proto.astype(np.float32))
                proto_idx.append(student_idx)
            elif enc_list:
                skipped_students.append(self._names[student_idx] if student_idx < len(self._names) else "?")
        
        if skipped_students:
            logger.warning(
                "Skipped %d student(s) with incompatible encoding dimensions "
                "(expected %d-D, run migrate_encodings.py to fix): %s",
                len(skipped_students), target_dim,
                ", ".join(skipped_students[:5])
                + (f" ... +{len(skipped_students)-5} more" if len(skipped_students) > 5 else ""),
            )
        if all_enc:
            self._flat_encodings = np.array(all_enc, dtype=np.float32)
            self._flat_student_idx = np.array(all_idx, dtype=np.intp)
            # L2-normalise rows for cosine similarity via dot product
            norms = np.linalg.norm(self._flat_encodings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # avoid division by zero
            self._flat_encodings /= norms
        else:
            self._flat_encodings = None
            self._flat_student_idx = None

        if proto_enc:
            self._student_prototypes = np.array(proto_enc, dtype=np.float32)
            self._prototype_student_idx = np.array(proto_idx, dtype=np.intp)
        else:
            self._student_prototypes = None
            self._prototype_student_idx = None

    def _normalize_encoding(self, encoding: np.ndarray) -> np.ndarray:
        """Return L2-normalized encoding vector."""
        enc = np.array(encoding, dtype=np.float32)
        norm = np.linalg.norm(enc, keepdims=True)
        norm = max(norm, 1e-10)  # avoid division by zero
        return (enc / norm).reshape(1, -1)  # Return as (1, D) for vstack

    def _append_encoding_to_flat(self, student_idx: int, encoding: np.ndarray) -> None:
        """Incrementally append a normalized encoding to the flat matrices.
        
        This is O(1) instead of O(N) when compared to full rebuild.
        Called while holding _lock.
        
        Must only be called when encoding dimension matches _embedding_dim.
        """
        if self._flat_encodings is None:
            # Cache is empty; rebuild is necessary
            self._rebuild_flat()
            return
        
        # Normalize the new encoding
        norm_enc = self._normalize_encoding(encoding)
        
        # Append to flat matrices
        self._flat_encodings = np.vstack([self._flat_encodings, norm_enc])
        self._flat_student_idx = np.append(self._flat_student_idx, student_idx)
        
        # Update student flat range
        old_start, old_end = self._student_flat_range.get(student_idx, (len(self._flat_encodings) - 1, len(self._flat_encodings)))
        if student_idx in self._student_flat_range:
            # Student already had encodings; extend their range
            self._student_flat_range[student_idx] = (old_start, len(self._flat_encodings))
        else:
            # New student entry
            self._student_flat_range[student_idx] = (len(self._flat_encodings) - 1, len(self._flat_encodings))


    def load(self):
        """Load / reload all encodings from the database."""
        rows = database.get_student_encodings()
        with self._lock:
            self._ids = [r[0] for r in rows]
            self._names = [r[1] for r in rows]
            self._encodings = [r[2] for r in rows]  # each r[2] is a list
            self._rebuild_flat()

            # Detect embedding dimension from loaded data
            if self._flat_encodings is not None:
                self._embedding_dim = self._flat_encodings.shape[1]

        logger.info(
            "Encoding cache loaded: %d students, dim=%d.",
            len(rows), self._embedding_dim,
        )

    def refresh(self):
        """Alias for load — call after new registration."""
        self.load()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._ids)

    @property
    def embedding_dim(self) -> int:
        with self._lock:
            return self._embedding_dim

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

    def get_stage1(self) -> tuple[np.ndarray | None, np.ndarray | None, list, list[str]]:
        """Return ``(student_prototypes, prototype_student_idx, ids, names)`` snapshot
        for stage-1 shortlist generation."""
        with self._lock:
            prototypes = self._student_prototypes
            prototype_idx = self._prototype_student_idx
            return (
                prototypes.copy() if prototypes is not None else None,
                prototype_idx.copy() if prototype_idx is not None else None,
                list(self._ids),
                list(self._names),
            )

    def get_student_ranges(self) -> dict[int, tuple[int, int]]:
        """Return a copy of student-index -> flat range mapping."""
        with self._lock:
            return dict(self._student_flat_range)

    def upsert_student(self, student_id, name: str, encodings: list[np.ndarray]) -> None:
        """Insert or replace a student's cached encodings in place.

        Use this after registration or recompute operations so the cache stays
        warm without a full database reload.
        """
        with self._lock:
            try:
                idx = self._ids.index(student_id)
            except ValueError:
                self._ids.append(student_id)
                self._names.append(name)
                self._encodings.append(list(encodings))
            else:
                self._names[idx] = name
                self._encodings[idx] = list(encodings)

            idx = self._ids.index(student_id)
            if len(self._encodings[idx]) > config.MAX_ENCODINGS_PER_STUDENT:
                self._encodings[idx] = self._encodings[idx][-config.MAX_ENCODINGS_PER_STUDENT:]
            self._rebuild_flat()

    def add_encoding_to_student(self, student_id, encoding: np.ndarray) -> None:
        """Add a new encoding to an existing student's in-memory list.

        This provides an immediate cache update without a full database
        reload.  If the number of encodings for the student exceeds
        ``config.MAX_ENCODINGS_PER_STUDENT``, the oldest encoding
        (index 0) is dropped and the cache is rebuilt (since flat indices change).

        O(1) when within encoding limit, O(N) only when limit exceeded (rare).

        Parameters
        ----------
        student_id :
            The ``bson.ObjectId`` of the student whose encoding list
            should be updated.
        encoding : np.ndarray
            A face embedding vector (512-D for ArcFace, 128-D for dlib).

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

            # Check encoding dimension matches current backend
            target_dim = config.EMBEDDING_DIM
            if encoding.shape[0] != target_dim:
                logger.warning(
                    "Attempted to add encoding with dim %d (expected %d) for student %s; skipping",
                    encoding.shape[0], target_dim, self._names[idx]
                )
                return

            self._encodings[idx].append(encoding)

            # Enforce the per-student encoding cap
            if len(self._encodings[idx]) > config.MAX_ENCODINGS_PER_STUDENT:
                # We're dropping an old encoding; flat indices will shift
                self._encodings[idx] = self._encodings[idx][-config.MAX_ENCODINGS_PER_STUDENT:]
                self._rebuild_flat()
            else:
                # Still under limit; do incremental append (O(1))
                self._append_encoding_to_flat(idx, encoding)



# Singleton cache instance
encoding_cache = EncodingCache()


# ---------------------------------------------------------------------------
# Confidence mapping
# ---------------------------------------------------------------------------

def _display_confidence_from_similarity(similarity: float) -> float:
    """Map cosine similarity score to a smoother UI confidence value.

    For cosine similarity, higher is better (range 0–1 for normalised
    embeddings).  This mapping provides a visually calibrated score.
    """
    alpha = max(0.1, float(config.RECOGNITION_CONFIDENCE_ALPHA))
    # Sigmoid-style mapping: centres around 0.5 similarity
    return float(np.clip(
        1.0 / (1.0 + np.exp(-alpha * (similarity - 0.35))),
        0.0,
        1.0,
    ))


# ---------------------------------------------------------------------------
# Recognition (cosine similarity)
# ---------------------------------------------------------------------------

def recognize_face(
    face_encoding: np.ndarray,
    threshold: float | None = None,
    ppe_state: str = "none",
    ppe_confidence: float = 0.0,
    image_quality: dict | None = None,
    candidate_student_ids: list | None = None,
) -> tuple | None:
    """Compare *face_encoding* against the cache using cosine similarity.

    All stored encodings are pre-flattened and L2-normalised into a single
    ``(N, D)`` matrix so that similarity computation is a single
    matrix-vector dot product.  For each student the **maximum**
    similarity across all their encodings is used.

    Returns ``(student_id, name, confidence)`` for the best match
    whose similarity is above *threshold*, or ``None`` if unknown.

    *confidence* is a UI-friendly score derived from cosine similarity.
    """
    if threshold is None:
        threshold = config.RECOGNITION_THRESHOLD

    # Apply dynamic threshold if quality metrics available
    if image_quality is not None:
        threshold = compute_dynamic_threshold(threshold, image_quality)

    min_conf = max(0.0, min(1.0, config.RECOGNITION_MIN_CONFIDENCE))
    min_gap = max(0.0, config.RECOGNITION_MIN_DISTANCE_GAP)

    occluded = (
        ppe_state in ("mask", "cap", "both")
        and ppe_confidence >= config.PPE_MIN_CONFIDENCE
    )
    if occluded:
        threshold = max(0.0, threshold - config.OCCLUDED_RECOGNITION_THRESHOLD_DELTA)
        min_gap = max(min_gap, config.OCCLUDED_MIN_DISTANCE_GAP)
        min_conf = max(min_conf, config.OCCLUDED_MIN_CONFIDENCE)

    flat_enc, flat_idx, ids, names = encoding_cache.get_flat()
    student_ranges = encoding_cache.get_student_ranges()
    stage1_prototypes, stage1_idx, _, _ = encoding_cache.get_stage1()
    if flat_enc is None or len(flat_enc) == 0:
        logger.debug("recognize_face: encoding cache is empty — returning None")
        return None

    # L2-normalise the query embedding
    query = face_encoding.astype(np.float32).flatten()
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    num_students = len(ids)
    candidate_student_indices: set[int] = set()

    if candidate_student_ids:
        id_to_idx = {sid: idx for idx, sid in enumerate(ids)}
        for sid in candidate_student_ids:
            idx = id_to_idx.get(sid)
            if idx is not None:
                candidate_student_indices.add(idx)

    if (
        config.RECOGNITION_TWO_STAGE_ENABLED
        and not candidate_student_indices
        and stage1_prototypes is not None
        and stage1_idx is not None
        and len(stage1_prototypes) > 0
    ):
        stage1_sims = stage1_prototypes @ query
        top_k = max(1, int(config.RECOGNITION_STAGE1_TOP_K))
        min_similarity = float(config.RECOGNITION_STAGE1_MIN_SIMILARITY)
        margin = max(0.0, float(config.RECOGNITION_STAGE1_MARGIN))

        ranked = np.argsort(stage1_sims)[::-1]
        if len(ranked) > 0:
            best_stage1 = float(stage1_sims[ranked[0]])
            dynamic_floor = max(min_similarity, best_stage1 - margin)
        else:
            dynamic_floor = min_similarity

        for proto_pos in ranked[:top_k]:
            sim = float(stage1_sims[proto_pos])
            if sim < dynamic_floor:
                continue
            candidate_student_indices.add(int(stage1_idx[proto_pos]))

        logger.debug(
            "recognize_face stage1: top_k=%d selected=%d total_students=%d dynamic_floor=%.4f",
            top_k,
            len(candidate_student_indices),
            num_students,
            dynamic_floor,
        )

    use_candidate_filter = (
        config.RECOGNITION_TWO_STAGE_ENABLED
        and len(candidate_student_indices) >= max(1, int(config.RECOGNITION_STAGE2_MIN_CANDIDATES))
        and len(candidate_student_indices) < num_students
    )

    # Cosine similarity: dot product of L2-normalised vectors.
    # Stage-2 can be scoped to shortlisted candidates to reduce compute.
    if use_candidate_filter:
        candidate_rows: list[int] = []
        for student_idx in candidate_student_indices:
            row_range = student_ranges.get(student_idx)
            if row_range is None:
                continue
            start, end = row_range
            candidate_rows.extend(range(start, end))

        if candidate_rows:
            candidate_rows_arr = np.array(candidate_rows, dtype=np.intp)
            similarities = flat_enc[candidate_rows_arr] @ query
            local_flat_idx = flat_idx[candidate_rows_arr]
        else:
            similarities = flat_enc @ query
            local_flat_idx = flat_idx
    else:
        similarities = flat_enc @ query
        local_flat_idx = flat_idx

    # Per-student maximum similarity
    max_sims = np.full(num_students, -1.0, dtype=np.float64)
    np.maximum.at(max_sims, local_flat_idx, similarities)

    best_idx = int(np.argmax(max_sims))
    best_sim = float(max_sims[best_idx])

    # Second-best student similarity for ambiguity guard
    if len(max_sims) > 1:
        sorted_sims = np.sort(max_sims)[::-1]  # descending
        second_best_sim = float(sorted_sims[1])
    else:
        second_best_sim = -1.0

    similarity_gap = best_sim - second_best_sim
    display_confidence = _display_confidence_from_similarity(best_sim)

    logger.debug(
        "recognize_face: best_match=%s best_sim=%.4f second_best=%.4f "
        "gap=%.4f threshold=%.4f conf_ui=%.4f min_conf=%.4f min_gap=%.4f "
        "ppe_state=%s ppe_conf=%.4f occluded=%s "
        "cache_students=%d cache_encodings=%d candidate_filtered=%s decision=%s",
        names[best_idx], best_sim, second_best_sim, similarity_gap, threshold,
        display_confidence, min_conf, min_gap,
        ppe_state, ppe_confidence, occluded,
        num_students, len(flat_enc),
        use_candidate_filter,
        "MATCH"
        if (
            best_sim >= threshold
            and best_sim >= min_conf
            and similarity_gap >= min_gap
        )
        else "REJECT",
    )

    if best_sim < threshold:
        return None

    if best_sim < min_conf:
        return None

    if similarity_gap < min_gap:
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
        A face embedding vector to append.

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
        # Catch all exceptions from database and cache operations to ensure graceful degradation.
        # Specific errors: PyMongo OperationFailure/DuplicateKeyError, cache KeyError, etc.
        logger.exception(
            "Failed to append encoding for student %s.", student_id
        )
        return False
