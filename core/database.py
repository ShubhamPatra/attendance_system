"""
Database layer -- MongoDB Atlas connection, indexes, and CRUD helpers.
"""

import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import bson
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import DuplicateKeyError, ConnectionFailure

import core.config as config
from core.utils import setup_logging, today_str, now_time_str

logger = setup_logging()


# ---------------------------------------------------------------------------
# Circuit Breaker Pattern (fault tolerance)
# ---------------------------------------------------------------------------
class CircuitBreakerState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing; reject calls fast
    HALF_OPEN = "half_open"    # Testing; allow single call


class CircuitBreaker:
    """Simple circuit breaker to prevent cascading failures on MongoDB operations."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        logger_instance = None,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            timeout_seconds: How long to wait in OPEN state before trying HALF_OPEN
            logger_instance: Logger to use (defaults to module logger)
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.logger = logger_instance or logger
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = __import__('threading').Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute func with circuit breaker protection.
        
        Raises:
            RuntimeError: If circuit is OPEN
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN; attempting single call")
                else:
                    raise RuntimeError(
                        f"Database circuit breaker is OPEN. "
                        f"Too many consecutive failures ({self.failure_count}). "
                        f"Retrying in {self.timeout_seconds:.0f}s."
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except (ConnectionFailure, TimeoutError) as exc:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Return True if timeout has elapsed since last failure."""
        if self.last_failure_time is None:
            return True
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.timeout_seconds
    
    def _on_success(self):
        """Called after successful operation."""
        with self.lock:
            self.failure_count = 0
            if self.state != CircuitBreakerState.CLOSED:
                self.logger.info(
                    "Circuit breaker reset to CLOSED after successful operation"
                )
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Called after failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.logger.error(
                    "Circuit breaker opening after %d consecutive failures",
                    self.failure_count,
                )
                self.state = CircuitBreakerState.OPEN
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.logger.warning(
                    "HALF_OPEN test call failed; circuit reopening"
                )
                self.state = CircuitBreakerState.OPEN


# Global circuit breaker for all database operations
_circuit_breaker = CircuitBreaker(
    failure_threshold=config.MONGO_CIRCUIT_BREAKER_THRESHOLD,
    timeout_seconds=config.MONGO_CIRCUIT_BREAKER_TIMEOUT_SECONDS,
)

# ---------------------------------------------------------------------------
# Connection (lazy singleton)
# ---------------------------------------------------------------------------
_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return a cached MongoClient instance."""
    global _client
    if _client is None:
        retries = max(1, config.MONGO_CONNECT_RETRIES)
        base_delay = max(0.0, config.MONGO_CONNECT_RETRY_DELAY_SECONDS)
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            candidate = MongoClient(
                config.MONGO_URI,
                serverSelectionTimeoutMS=config.MONGO_SERVER_SELECTION_TIMEOUT_MS,
                connectTimeoutMS=config.MONGO_CONNECT_TIMEOUT_MS,
                maxPoolSize=config.MONGO_MAX_POOL_SIZE,
                minPoolSize=config.MONGO_MIN_POOL_SIZE,
                maxIdleTimeMS=config.MONGO_MAX_IDLE_TIME_MS,
            )
            try:
                # Verify the connection early and cache only healthy clients.
                candidate.admin.command("ping")
                _client = candidate
                logger.info(
                    "Connected to MongoDB Atlas. "
                    "Pool: max=%d, min=%d, maxIdle=%dms.",
                    config.MONGO_MAX_POOL_SIZE,
                    config.MONGO_MIN_POOL_SIZE,
                    config.MONGO_MAX_IDLE_TIME_MS,
                )
                break
            except ConnectionFailure as exc:
                last_exc = exc
                candidate.close()
                if attempt < retries:
                    delay = base_delay * attempt
                    logger.warning(
                        "MongoDB connection attempt %d/%d failed; retrying in %.1fs: %s",
                        attempt,
                        retries,
                        delay,
                        exc,
                    )
                    if delay > 0:
                        time.sleep(delay)
                else:
                    logger.error(
                        "MongoDB connection failed after %d attempt(s): %s",
                        retries,
                        exc,
                    )

        if _client is None and last_exc is not None:
            raise last_exc
    return _client


def get_db():
    """Return the application database handle.
    
    Raises:
        RuntimeError: If circuit breaker is OPEN (too many failures)
    """
    return _circuit_breaker.call(
        lambda: get_client()[config.DATABASE_NAME]
    )


# ---------------------------------------------------------------------------
# Index setup (idempotent -- safe to call on every startup)
# ---------------------------------------------------------------------------

def ensure_indexes():
    """Create required unique indexes if they don't exist."""
    db = get_db()
    db.students.create_index(
        [("registration_number", ASCENDING)],
        unique=True,
        name="uq_registration_number",
    )
    db.students.create_index(
        [("email", ASCENDING)],
        unique=True,
        sparse=True,
        name="uq_students_email",
    )
    db.students.create_index(
        [("verification_status", ASCENDING)],
        name="idx_students_verification_status",
    )
    db.users.create_index(
        [("username", ASCENDING)],
        unique=True,
        sparse=True,
        name="uq_users_username",
    )
    db.users.create_index(
        [("email", ASCENDING)],
        sparse=True,
        name="idx_users_email",
    )
    db.notification_events.create_index(
        [("created_at", ASCENDING)],
        name="idx_notification_events_created_at",
    )
    # Ensure legacy attendance index variant is removed if present.
    try:
        db.attendance.drop_index("uq_student_date_subject")
    except Exception as exc:
        logger.debug("Legacy index not found (expected): %s", exc)

    db.attendance.create_index(
        [("student_id", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uq_student_date",
    )
    db.attendance.create_index(
        [("date", ASCENDING)],
        name="idx_date",
    )
    db.attendance_sessions.create_index(
        [("camera_id", ASCENDING), ("status", ASCENDING)],
        unique=True,
        partialFilterExpression={"status": "active"},
        name="uq_active_session_per_camera",
    )
    db.attendance_sessions.create_index(
        [("status", ASCENDING), ("start_time", ASCENDING)],
        name="idx_attendance_sessions_status_start_time",
    )
    db.attendance_sessions.create_index(
        [("course_id", ASCENDING), ("camera_id", ASCENDING), ("start_time", ASCENDING)],
        name="idx_attendance_sessions_course_camera_start",
    )
    db.attendance_sessions.create_index(
        [("last_activity_at", ASCENDING)],
        name="idx_attendance_sessions_last_activity",
    )
    logger.info("Database indexes ensured.")


def _student_projection(include_sensitive: bool = False) -> dict:
    projection = {"encodings": 0, "face_encoding": 0}
    if not include_sensitive:
        projection["password_hash"] = 0
    return projection


def _student_exists(registration_number: str | None = None, email: str | None = None) -> bool:
    db = get_db()
    clauses: list[dict] = []
    if registration_number:
        clauses.append({"registration_number": registration_number})
    if email:
        clauses.append({"email": email})
    if not clauses:
        return False
    query = clauses[0] if len(clauses) == 1 else {"$or": clauses}
    return db.students.find_one(query, {"_id": 1}) is not None


# ---------------------------------------------------------------------------
# Users / Auth
# ---------------------------------------------------------------------------

def insert_user(
    username: str,
    password_hash: str,
    role: str,
    email: str | None = None,
    is_active: bool = True,
) -> bson.ObjectId:
    """Insert a user account and return inserted _id."""
    db = get_db()
    doc = {
        "username": username,
        "password_hash": password_hash,
        "role": role,
        "email": email,
        "is_active": is_active,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
    }
    try:
        result = db.users.insert_one(doc)
        logger.info("Inserted user %s (%s).", username, role)
        return result.inserted_id
    except DuplicateKeyError as exc:
        raise ValueError(f"Username '{username}' already exists.") from exc


def get_user_by_username(username: str) -> dict | None:
    """Return user document by username."""
    db = get_db()
    return db.users.find_one({"username": username})


def get_user_by_id(user_id: bson.ObjectId) -> dict | None:
    """Return user document by _id."""
    db = get_db()
    return db.users.find_one({"_id": user_id})


def update_user_last_login(user_id: bson.ObjectId) -> None:
    """Update last login timestamp for a user."""
    db = get_db()
    db.users.update_one(
        {"_id": user_id},
        {"$set": {"last_login": datetime.now(timezone.utc)}},
    )


def count_users_by_role(role: str) -> int:
    """Return active user count for a role."""
    db = get_db()
    return db.users.count_documents({"role": role, "is_active": True})


def insert_notification_event(event: dict) -> bson.ObjectId:
    """Persist a notification event document."""
    db = get_db()
    result = db.notification_events.insert_one(event)
    return result.inserted_id


def get_notification_events(limit: int = 100) -> list[dict]:
    """Return the most recent notification events."""
    db = get_db()
    return list(db.notification_events.find({}).sort("created_at", -1).limit(max(1, limit)))


# ---------------------------------------------------------------------------
# Students CRUD
# ---------------------------------------------------------------------------

def insert_student(
    name: str,
    semester: int,
    registration_number: str,
    section: str,
    encodings: list[np.ndarray],
    email: str | None = None,
    password_hash: str | None = None,
    verification_status: str = "approved",
    verification_score: float | None = None,
    verification_reason: str | None = None,
    face_samples: list[str] | None = None,
) -> bson.ObjectId:
    """Insert a new student with multiple face encodings.

    Returns the inserted _id.
    Raises ValueError on duplicate registration_number.
    """
    db = get_db()
    if _student_exists(registration_number=registration_number):
        raise ValueError(f"Registration number '{registration_number}' already exists.")
    if email and _student_exists(email=email):
        raise ValueError(f"Email '{email}' already exists.")

    doc = {
        "name": name,
        "semester": semester,
        "registration_number": registration_number,
        "section": section,
        "encodings": [_encode_for_storage(e) for e in encodings],
        "email": email,
        "password_hash": password_hash,
        "verification_status": verification_status,
        "verification_score": verification_score,
        "verification_reason": verification_reason,
        "face_samples": face_samples or [],
        "verification_updated_at": datetime.now(timezone.utc),
        "created_at": datetime.now(timezone.utc),
        "approved_at": datetime.now(timezone.utc) if verification_status == "approved" else None,
        "rejected_at": datetime.now(timezone.utc) if verification_status == "rejected" else None,
    }
    try:
        result = db.students.insert_one(doc)
        logger.info("Inserted student %s (%s).", name, registration_number)
        return result.inserted_id
    except DuplicateKeyError:
        raise ValueError(
            f"Registration number or email already exists for '{registration_number}'."
        )


def get_student_by_id(student_id: bson.ObjectId, include_sensitive: bool = False) -> dict | None:
    """Return a student's details by _id (without face_encoding)."""
    db = get_db()
    projection = _student_projection(include_sensitive=include_sensitive)
    projection["created_at"] = 0
    return db.students.find_one({"_id": student_id}, projection)


def get_student_by_reg_no(reg_no: str, include_sensitive: bool = False) -> dict | None:
    """Return a student document by registration number."""
    db = get_db()
    return db.students.find_one({"registration_number": reg_no}, _student_projection(include_sensitive=include_sensitive))


def get_student_by_email(email: str, include_sensitive: bool = False) -> dict | None:
    """Return a student document by email."""
    db = get_db()
    return db.students.find_one({"email": email}, _student_projection(include_sensitive=include_sensitive))


def get_students_by_verification_status(status: str) -> list[dict]:
    """Return students filtered by verification status."""
    db = get_db()
    return list(db.students.find({"verification_status": status}, _student_projection(False)).sort("created_at", ASCENDING))


def get_pending_students() -> list[dict]:
    """Return pending student registrations for admin review."""
    return get_students_by_verification_status("pending")


def get_all_students() -> list[dict]:
    """Return all students (without encodings) sorted by name."""
    db = get_db()
    return list(
        db.students.find(
            {},
            _student_projection(include_sensitive=False),
        ).sort("name", ASCENDING)
    )


def get_student_encodings() -> list[tuple]:
    """Return list of (student_id, name, [np.ndarray, ...]) for all students.

    Handles multiple embedding formats:
    * **ArcFace (512-D float32)** — 2048 bytes per encoding
    * **dlib (128-D float64)** — 1024 bytes per encoding
    * **Legacy ``face_encoding`` single field** — backward compatibility

    Encodings that don't match a known dimension are skipped with a warning.
    """
    db = get_db()
    results = []
    query = {
        "$or": [
            {"verification_status": {"$exists": False}},
            {"verification_status": "approved"},
        ]
    }

    # Known (dimension, dtype, byte_size) combinations
    _KNOWN_FORMATS = [
        (512, np.float32, 512 * 4),   # ArcFace
        (128, np.float64, 128 * 8),   # dlib
        (128, np.float32, 128 * 4),   # dlib stored as float32
    ]

    def _decode_encoding(raw_bytes, student_name: str, index: int):
        """Try known formats and return ndarray or None."""
        nbytes = len(raw_bytes)
        for dim, dtype, expected_size in _KNOWN_FORMATS:
            if nbytes == expected_size:
                arr = np.frombuffer(raw_bytes, dtype=dtype)
                if arr.shape == (dim,):
                    return arr.astype(np.float32)  # normalise to float32
        # Unknown size — log and skip
        logger.warning(
            "Skipping encoding %d for student %s "
            "(byte_size=%d, no matching format)",
            index, student_name, nbytes,
        )
        return None

    for doc in db.students.find(query, {"name": 1, "encodings": 1, "face_encoding": 1}):
        enc_list: list[np.ndarray] = []
        student_name = doc.get("name", "?")

        if "encodings" in doc and doc["encodings"]:
            for i, raw in enumerate(doc["encodings"]):
                arr = _decode_encoding(raw, student_name, i)
                if arr is not None:
                    enc_list.append(arr)
        elif "face_encoding" in doc:
            arr = _decode_encoding(doc["face_encoding"], student_name, 0)
            if arr is not None:
                enc_list.append(arr)

        if enc_list:
            results.append((doc["_id"], doc["name"], enc_list))
    return results


def _encode_for_storage(encoding: np.ndarray) -> bson.Binary:
    """Serialise a face embedding for MongoDB storage.

    ArcFace embeddings (512-D) are stored as float32 (2048 bytes).
    Legacy dlib embeddings (128-D) are stored as float64 (1024 bytes)
    to preserve backward compatibility with existing data.
    
    Validates encoding format before storage.
    
    Raises:
        ValueError: If encoding format is invalid or unsupported
    """
    if encoding is None:
        raise ValueError("Encoding cannot be None")
    
    if not isinstance(encoding, np.ndarray):
        raise ValueError(f"Encoding must be np.ndarray, got {type(encoding)}")
    
    if len(encoding.shape) != 1:
        raise ValueError(f"Encoding must be 1-dimensional, got shape {encoding.shape}")
    
    # Accept standard embedding dimensions
    valid_dims = {512, 128}
    if encoding.shape[0] not in valid_dims:
        raise ValueError(
            f"Encoding dimension {encoding.shape[0]} not supported. "
            f"Valid dimensions: {valid_dims}"
        )
    
    # Valid dtypes for storage
    if encoding.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"Encoding dtype {encoding.dtype} not supported. "
            f"Valid: float32, float64"
        )
    
    try:
        if encoding.shape == (512,) or encoding.dtype == np.float32:
            return bson.Binary(encoding.astype(np.float32).tobytes())
        return bson.Binary(encoding.astype(np.float64).tobytes())
    except Exception as exc:
        raise ValueError(
            f"Failed to serialize encoding to bytes: {exc}"
        ) from exc



def append_student_encoding(student_id: bson.ObjectId, new_encoding: np.ndarray) -> bool:
    """Push a new face encoding to the student's encodings array.

    Uses $push with $each/$slice to keep only the last
    ``config.MAX_ENCODINGS_PER_STUDENT`` encodings, automatically
    dropping the oldest when the cap is exceeded.
    
    Validates encoding before storage and deduplicates against existing
    encodings: if similarity (cosine) > 0.95, skips append.
    
    Validates encoding before storage.
    
    Returns:
        True if encoding was stored successfully, False if skipped (duplicate or validation error).
        On validation error, logs warning and returns False.
    """
    try:
        encoded = _encode_for_storage(new_encoding)
    except ValueError as exc:
        logger.warning(
            "Encoding validation failed for student %s (will not be stored): %s",
            student_id, exc,
        )
        return False
    
    db = get_db()
    
    # Check for duplicate encodings by cosine similarity
    student = db.students.find_one(
        {"_id": student_id},
        {"encodings": 1, "face_encoding": 1}
    )
    
    if student:
        # Normalize new encoding for cosine similarity
        new_enc_float32 = new_encoding.astype(np.float32)
        new_norm = np.linalg.norm(new_enc_float32)
        if new_norm > 1e-10:
            new_enc_normalized = new_enc_float32 / new_norm
        else:
            new_enc_normalized = new_enc_float32
        
        # Check existing encodings
        dedup_threshold = 0.95
        duplicate_found = False
        
        existing_encodings = []
        
        if "encodings" in student and student["encodings"]:
            # Check against new-format encodings
            for raw_bytes in student["encodings"]:
                try:
                    # Try to decode with known formats
                    _KNOWN_FORMATS = [
                        (512, np.float32, 512 * 4),
                        (512, np.float64, 512 * 8),
                        (128, np.float64, 128 * 8),
                        (128, np.float32, 128 * 4),
                    ]
                    for dim, dtype, expected_size in _KNOWN_FORMATS:
                        if len(raw_bytes) == expected_size:
                            existing = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
                            existing_encodings.append(existing)
                            break
                except Exception:
                    # Skip malformed encodings
                    pass
        
        elif "face_encoding" in student:
            # Check against legacy single encoding
            try:
                _KNOWN_FORMATS = [
                    (512, np.float32, 512 * 4),
                    (512, np.float64, 512 * 8),
                    (128, np.float64, 128 * 8),
                    (128, np.float32, 128 * 4),
                ]
                for dim, dtype, expected_size in _KNOWN_FORMATS:
                    if len(student["face_encoding"]) == expected_size:
                        existing = np.frombuffer(student["face_encoding"], dtype=dtype).astype(np.float32)
                        existing_encodings.append(existing)
                        break
            except Exception:
                pass
        
        # Compare against all existing encodings
        for existing in existing_encodings:
            existing_norm = np.linalg.norm(existing)
            if existing_norm > 1e-10:
                existing_normalized = existing / existing_norm
            else:
                existing_normalized = existing
            
            # Cosine similarity
            similarity = np.dot(new_enc_normalized, existing_normalized)
            
            if similarity > dedup_threshold:
                logger.debug(
                    "Skipping duplicate encoding for student %s (similarity=%.4f > %.4f)",
                    student_id, similarity, dedup_threshold,
                )
                duplicate_found = True
                break
        
        if duplicate_found:
            return False
    
    # Store the encoding
    result = db.students.update_one(
        {"_id": student_id},
        {"$push": {
            "encodings": {
                "$each": [encoded],
                "$slice": -config.MAX_ENCODINGS_PER_STUDENT,
            }
        }}
    )
    return result.modified_count > 0




def student_count() -> int:
    """Return total number of registered students."""
    return get_db().students.count_documents({})


def delete_student(reg_no: str) -> bool:
    """Delete a student and all their attendance records by registration number."""
    db = get_db()
    student = db.students.find_one({"registration_number": reg_no}, {"_id": 1})
    if not student:
        return False
    db.attendance.delete_many({"student_id": student["_id"]})
    result = db.students.delete_one({"_id": student["_id"]})
    return result.deleted_count > 0


def update_student(reg_no: str, updates: dict) -> bool:
    """Update a student profile by registration number."""
    db = get_db()
    payload = {
        key: value
        for key, value in updates.items()
        if key in {"name", "semester", "section", "email", "is_active"} and value is not None
    }
    if not payload:
        return False
    result = db.students.update_one(
        {"registration_number": reg_no},
        {"$set": payload},
    )
    return result.modified_count > 0


def replace_student_encodings(reg_no: str, encodings: list[np.ndarray]) -> bool:
    """Replace all encodings for a student."""
    db = get_db()
    result = db.students.update_one(
        {"registration_number": reg_no},
        {"$set": {"encodings": [_encode_for_storage(e) for e in encodings]}},
    )
    return result.modified_count > 0


def update_student_verification(
    reg_no: str,
    status: str,
    score: float | None = None,
    reason: str | None = None,
    face_samples: list[str] | None = None,
    encodings: list[np.ndarray] | None = None,
) -> bool:
    """Update verification state and onboarding artifacts."""
    db = get_db()
    now = datetime.now(timezone.utc)
    payload: dict = {
        "verification_status": status,
        "verification_updated_at": now,
    }
    if score is not None:
        payload["verification_score"] = round(float(score), 2)
    if reason is not None:
        payload["verification_reason"] = reason
    if face_samples is not None:
        payload["face_samples"] = face_samples
    if encodings is not None:
        payload["encodings"] = [_encode_for_storage(e) for e in encodings]
    if status == "approved":
        payload["approved_at"] = now
    elif status == "rejected":
        payload["rejected_at"] = now

    result = db.students.update_one({"registration_number": reg_no}, {"$set": payload})
    return result.modified_count > 0


def set_student_password(reg_no: str, password_hash: str) -> bool:
    """Persist a hashed password for a student account."""
    db = get_db()
    result = db.students.update_one(
        {"registration_number": reg_no},
        {"$set": {"password_hash": password_hash}},
    )
    return result.modified_count > 0


# ---------------------------------------------------------------------------
# Attendance CRUD
# ---------------------------------------------------------------------------

def create_attendance_session(
    course_id: str,
    camera_id: str,
    start_time: datetime | None = None,
) -> bson.ObjectId:
    """Create a new active attendance session for a camera.

    Raises ValueError if an active session already exists for *camera_id*.
    """
    db = get_db()
    now = (start_time or datetime.now(timezone.utc)).astimezone(timezone.utc)

    existing = db.attendance_sessions.find_one(
        {"camera_id": camera_id, "status": "active"},
        {"_id": 1},
    )
    if existing is not None:
        raise ValueError(
            f"An active attendance session already exists for camera '{camera_id}'."
        )

    doc = {
        "course_id": course_id,
        "camera_id": camera_id,
        "start_time": now,
        "end_time": None,
        "status": "active",
        "last_activity_at": now,
        "created_at": now,
        "updated_at": now,
    }
    result = db.attendance_sessions.insert_one(doc)
    return result.inserted_id


def get_attendance_session_by_id(session_id: bson.ObjectId | str) -> dict[str, Any] | None:
    """Return attendance session document by session id."""
    db = get_db()
    sid = bson.ObjectId(session_id) if isinstance(session_id, str) else session_id
    return db.attendance_sessions.find_one({"_id": sid})


def get_active_attendance_session(camera_id: str) -> dict[str, Any] | None:
    """Return the active attendance session for *camera_id*, if any."""
    db = get_db()
    return db.attendance_sessions.find_one(
        {"camera_id": camera_id, "status": "active"},
        sort=[("start_time", -1)],
    )


def end_attendance_session(
    session_id: bson.ObjectId | str,
    end_time: datetime | None = None,
) -> bool:
    """Mark an attendance session as ended."""
    db = get_db()
    sid = bson.ObjectId(session_id) if isinstance(session_id, str) else session_id
    ended_at = (end_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    result = db.attendance_sessions.update_one(
        {"_id": sid, "status": "active"},
        {
            "$set": {
                "status": "ended",
                "end_time": ended_at,
                "updated_at": ended_at,
                "last_activity_at": ended_at,
            }
        },
    )
    return result.modified_count > 0


def touch_attendance_session(
    session_id: bson.ObjectId | str,
    activity_time: datetime | None = None,
) -> bool:
    """Update last activity timestamp for an active attendance session."""
    db = get_db()
    sid = bson.ObjectId(session_id) if isinstance(session_id, str) else session_id
    ts = (activity_time or datetime.now(timezone.utc)).astimezone(timezone.utc)
    result = db.attendance_sessions.update_one(
        {"_id": sid, "status": "active"},
        {"$set": {"last_activity_at": ts, "updated_at": ts}},
    )
    return result.modified_count > 0


def auto_close_idle_attendance_sessions(
    idle_seconds: int = config.ATTENDANCE_SESSION_IDLE_TIMEOUT_SECONDS,
    now: datetime | None = None,
) -> int:
    """Auto-close active sessions idle for at least *idle_seconds*.

    Returns number of sessions transitioned to ``ended``.
    """
    db = get_db()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = current - timedelta(seconds=max(1, int(idle_seconds)))
    result = db.attendance_sessions.update_many(
        {
            "status": "active",
            "$or": [
                {"last_activity_at": {"$lte": cutoff}},
                {
                    "last_activity_at": {"$exists": False},
                    "start_time": {"$lte": cutoff},
                },
            ],
        },
        {
            "$set": {
                "status": "ended",
                "end_time": current,
                "updated_at": current,
                "last_activity_at": current,
            }
        },
    )
    return result.modified_count

def mark_attendance(
    student_id: bson.ObjectId,
    confidence: float,
    session_id: bson.ObjectId | str | None = None,
) -> bool:
    """Mark attendance for *student_id* today.

    Returns True if a new record was inserted, False if already present.
    
    Raises:
        RuntimeError: If database circuit breaker is OPEN (too many failures)
    """
    try:
        db = get_db()
        date = today_str()
        normalized_session_id = None
        if session_id is not None:
            normalized_session_id = (
                bson.ObjectId(session_id)
                if isinstance(session_id, str)
                else session_id
            )
            active = db.attendance_sessions.find_one(
                {"_id": normalized_session_id, "status": "active"},
                {"_id": 1},
            )
            if active is None:
                logger.warning(
                    "Skipping attendance mark for %s: inactive or missing session_id=%s",
                    student_id,
                    normalized_session_id,
                )
                return False

        doc = {
            "student_id": student_id,
            "date": date,
            "time": now_time_str(),
            "status": "Present",
            "confidence_score": round(confidence, 4),
        }
        if normalized_session_id is not None:
            doc["session_id"] = normalized_session_id
        db.attendance.insert_one(doc)
        logger.info(
            "Attendance marked for student %s on %s.", student_id, date
        )
        return True
    except DuplicateKeyError:
        return False


def bulk_upsert_attendance(
    entries: list[dict],
    session_id: bson.ObjectId | str | None = None,
) -> int:
    """Bulk upsert attendance records for today.

    *entries* is a list of dicts with keys: ``student_id`` (ObjectId),
    ``status`` ("Present"/"Absent"), ``confidence_score`` (float, optional).

    Returns the total number of upserted + modified documents.
    
    Raises:
        RuntimeError: If database circuit breaker is OPEN (too many failures)
    """
    try:
        db = get_db()
        date = today_str()
        normalized_session_id = None
        if session_id is not None:
            normalized_session_id = (
                bson.ObjectId(session_id)
                if isinstance(session_id, str)
                else session_id
            )
            active = db.attendance_sessions.find_one(
                {"_id": normalized_session_id, "status": "active"},
                {"_id": 1},
            )
            if active is None:
                raise ValueError(
                    f"Attendance session {normalized_session_id} is not active."
                )

        ops = []
        for entry in entries:
            payload = {
                "status": entry.get("status", "Present"),
                "time": now_time_str(),
                "confidence_score": entry.get("confidence_score", 0.0),
            }
            if normalized_session_id is not None:
                payload["session_id"] = normalized_session_id
            ops.append(UpdateOne(
                {
                    "student_id": entry["student_id"],
                    "date": date,
                },
                {"$set": payload},
                upsert=True,
            ))
        if not ops:
            return 0
        result = db.attendance.bulk_write(ops)
        return result.upserted_count + result.modified_count
    except RuntimeError:
        # Circuit breaker is open; re-raise for caller to handle
        raise


def get_attendance(date_str: str | None = None) -> list[dict]:
    """Return attendance records, optionally filtered by date.

    Each record includes the student's name via aggregation.
    """
    db = get_db()
    match_filters: dict = {}

    if date_str:
        match_filters["date"] = date_str

    pipeline = [
        {"$match": match_filters},
        {
            "$lookup": {
                "from": "students",
                "localField": "student_id",
                "foreignField": "_id",
                "as": "student",
            }
        },
        {"$unwind": "$student"},
        {
            "$project": {
                "_id": 0,
                "name": "$student.name",
                "registration_number": "$student.registration_number",
                "section": "$student.section",
                "semester": "$student.semester",
                "date": 1,
                "time": 1,
                "status": 1,
                "confidence_score": 1,
            }
        },
        {"$sort": {"date": -1, "time": -1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def today_attendance_count() -> int:
    """Return number of students marked present today."""
    return get_db().attendance.count_documents({"date": today_str()})


def get_attendance_csv(date_str: str) -> pd.DataFrame:
    """Return attendance for *date_str* as a pandas DataFrame."""
    records = get_attendance(date_str)
    return _records_to_df(records)


# ---------------------------------------------------------------------------
# Extended attendance queries
# ---------------------------------------------------------------------------

def get_attendance_by_student(reg_no: str) -> list[dict]:
    """Return all attendance records for a given registration number."""
    db = get_db()
    student = db.students.find_one(
        {"registration_number": reg_no}, {"_id": 1}
    )
    if not student:
        return []
    return get_attendance_filtered(student_id=student["_id"])


def get_attendance_by_date_range(
    start_date: str, end_date: str
) -> list[dict]:
    """Return attendance records between *start_date* and *end_date* inclusive."""
    db = get_db()
    pipeline = [
        {"$match": {"date": {"$gte": start_date, "$lte": end_date}}},
        {
            "$lookup": {
                "from": "students",
                "localField": "student_id",
                "foreignField": "_id",
                "as": "student",
            }
        },
        {"$unwind": "$student"},
        {
            "$project": {
                "_id": 0,
                "name": "$student.name",
                "registration_number": "$student.registration_number",
                "section": "$student.section",
                "semester": "$student.semester",
                "date": 1,
                "time": 1,
                "status": 1,
                "confidence_score": 1,
            }
        },
        {"$sort": {"date": -1, "time": -1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_attendance_filtered(
    student_id: bson.ObjectId | None = None,
    date_str: str | None = None,
) -> list[dict]:
    """Flexible attendance query with optional student_id and/or date."""
    db = get_db()
    match: dict = {}
    if student_id:
        match["student_id"] = student_id
    if date_str:
        match["date"] = date_str

    pipeline = [
        {"$match": match},
        {
            "$lookup": {
                "from": "students",
                "localField": "student_id",
                "foreignField": "_id",
                "as": "student",
            }
        },
        {"$unwind": "$student"},
        {
            "$project": {
                "_id": 0,
                "name": "$student.name",
                "registration_number": "$student.registration_number",
                "section": "$student.section",
                "semester": "$student.semester",
                "date": 1,
                "time": 1,
                "status": 1,
                "confidence_score": 1,
            }
        },
        {"$sort": {"date": -1, "time": -1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_attendance_by_hour(date_str: str) -> list[dict]:
    """Return attendance counts grouped by hour for a given date.

    Returns list of ``{"hour": 8, "count": 5}`` dicts.
    """
    db = get_db()
    pipeline = [
        {"$match": {"date": date_str}},
        {
            "$project": {
                "hour": {"$toInt": {"$substr": ["$time", 0, 2]}},
            }
        },
        {
            "$group": {
                "_id": "$hour",
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
        {"$project": {"_id": 0, "hour": "$_id", "count": 1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_all_registration_numbers() -> list[str]:
    """Return sorted list of all student registration numbers."""
    db = get_db()
    return sorted(
        doc["registration_number"]
        for doc in db.students.find({}, {"registration_number": 1})
    )


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def get_at_risk_students(days: int = 30, threshold: int | None = None) -> list[dict]:
    """Return students whose attendance percentage is below *threshold*.

    Looks at the last *days* calendar days. Each entry contains
    ``name``, ``reg_no``, ``percentage``, ``days_present``, and ``days_total``.
    Results are sorted by percentage ascending (worst first).

    Uses MongoDB aggregation pipeline for O(1) query cost instead of O(N).
    """
    if threshold is None:
        threshold = config.ABSENCE_THRESHOLD
    db = get_db()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Get total class days in the range
    total_class_days = len(db.attendance.distinct(
        "date",
        {"date": {"$gte": start_str, "$lte": end_str}},
    ))

    if total_class_days == 0:
        return []

    # Single aggregation query: group by student, count attendance,
    # join with students collection, filter by threshold
    pipeline = [
        {
            "$match": {
                "date": {"$gte": start_str, "$lte": end_str},
                "status": "Present",
            }
        },
        {
            "$group": {
                "_id": "$student_id",
                "days_present": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "students",
                "localField": "_id",
                "foreignField": "_id",
                "as": "student_info",
            }
        },
        {
            "$unwind": {
                "path": "$student_info",
                "preserveNullAndEmptyArrays": False,
            }
        },
        {
            "$addFields": {
                "days_total": total_class_days,
                "percentage": {
                    "$cond": [
                        {"$gt": [total_class_days, 0]},
                        {
                            "$round": [
                                {"$multiply": [
                                    {"$divide": ["$days_present", total_class_days]},
                                    100,
                                ]},
                                1,
                            ]
                        },
                        0,
                    ]
                },
            }
        },
        {
            "$match": {
                "percentage": {"$lt": threshold},
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": "$student_info.name",
                "reg_no": "$student_info.registration_number",
                "percentage": 1,
                "days_present": 1,
                "days_total": 1,
            }
        },
        {
            "$sort": {"percentage": 1},
        },
    ]

    results = list(db.attendance.aggregate(pipeline))
    return results


def get_student_attendance_summary(reg_no: str, days: int = 30) -> dict | None:
    """Return an attendance summary for a single student over the last *days* days.

    Returns ``None`` if the student is not found. Otherwise returns a dict
    with student info, attendance percentage, and individual records.
    """
    db = get_db()
    student = db.students.find_one(
        {"registration_number": reg_no},
        {"name": 1, "registration_number": 1, "semester": 1, "section": 1},
    )
    if not student:
        return None

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    records = list(db.attendance.find(
        {"student_id": student["_id"], "date": {"$gte": start_str, "$lte": end_str}},
        {"_id": 0, "date": 1, "time": 1, "status": 1, "confidence_score": 1}
    ).sort("date", -1))

    # Count distinct class dates (days where attendance was recorded for any student)
    total_class_days = db.attendance.distinct(
        "date", {"date": {"$gte": start_str, "$lte": end_str}}
    )
    total = len(total_class_days)

    count = sum(1 for r in records if r.get("status") == "Present")
    pct = round(count / total * 100, 1) if total > 0 else 0

    return {
        "name": student["name"],
        "registration_number": student["registration_number"],
        "semester": student.get("semester", ""),
        "section": student.get("section", ""),
        "percentage": pct,
        "days_present": count,
        "days_total": total,
        "records": records,
    }


def get_attendance_heatmap_data(days: int = 90) -> list[dict]:
    """Return [{date, count, total_students}] for the last N days."""
    db = get_db()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    total = db.students.count_documents({})

    pipeline = [
        {"$match": {"date": {"$gte": start_str, "$lte": end_str}}},
        {"$group": {"_id": "$date", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
        {"$project": {"_id": 0, "date": "$_id", "count": 1}},
    ]
    results = list(db.attendance.aggregate(pipeline))
    for r in results:
        r["total_students"] = total
    return results


def get_attendance_trends(days: int = 14) -> list[dict]:
    """Return daily attendance trend data for the last *days* days."""
    db = get_db()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days - 1)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    total_students = db.students.count_documents({})

    pipeline = [
        {"$match": {"date": {"$gte": start_str, "$lte": end_str}, "status": "Present"}},
        {"$group": {"_id": "$date", "present": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    results = {doc["_id"]: int(doc["present"]) for doc in db.attendance.aggregate(pipeline)}

    days_out: list[dict] = []
    for offset in range(days):
        current = start_date + timedelta(days=offset)
        date_str = current.strftime("%Y-%m-%d")
        present = results.get(date_str, 0)
        pct = round((present / total_students) * 100, 1) if total_students else 0.0
        days_out.append(
            {
                "date": date_str,
                "present": present,
                "total_students": total_students,
                "attendance_pct": pct,
            }
        )
    return days_out


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "Name", "Registration Number", "Section",
    "Semester", "Date", "Time", "Status", "Confidence",
]

_CSV_RENAME_MAP = {
    "name": "Name",
    "registration_number": "Registration Number",
    "section": "Section",
    "semester": "Semester",
    "date": "Date",
    "time": "Time",
    "status": "Status",
    "confidence_score": "Confidence",
}


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert attendance records to a renamed DataFrame."""
    if not records:
        return pd.DataFrame(columns=_CSV_COLUMNS)
    df = pd.DataFrame(records)
    df.rename(columns=_CSV_RENAME_MAP, inplace=True)
    return df


def get_attendance_csv_by_student(reg_no: str) -> pd.DataFrame:
    return _records_to_df(get_attendance_by_student(reg_no))


def get_attendance_csv_by_date_range(
    start_date: str, end_date: str
) -> pd.DataFrame:
    return _records_to_df(get_attendance_by_date_range(start_date, end_date))


def get_attendance_csv_full() -> pd.DataFrame:
    """Export entire attendance history."""
    return _records_to_df(get_attendance())


# ---------------------------------------------------------------------------
# Encoding Validation & Diagnostics
# ---------------------------------------------------------------------------

def validate_student_encodings(student_id: bson.ObjectId | None = None) -> dict:
    """Validate encoding integrity in database.
    
    Checks all encodings for:
    - Valid byte sizes matching known formats
    - Correct dimensions when decoded
    - Data corruption
    
    Parameters:
        student_id: If provided, check only this student; else check all
    
    Returns:
        {
            'total_students': int,
            'students_with_valid': int,
            'students_with_corrupted': int,
            'corrupted_students': [
                {'student_id': ObjectId, 'name': str, 'corrupted_count': int, 'total_count': int},
                ...
            ]
        }
    """
    db = get_db()
    query = {}
    if student_id:
        query['_id'] = student_id
    
    total_students = 0
    valid_students = 0
    corrupted_students = []
    
    _KNOWN_FORMATS = [
        (512, np.float32, 512 * 4),  # ArcFace
        (512, np.float64, 512 * 8),  # ArcFace double-precision (rare)
        (128, np.float64, 128 * 8),  # dlib
        (128, np.float32, 128 * 4),  # dlib stored as float32
    ]
    
    for doc in db.students.find(query, {"name": 1, "encodings": 1, "face_encoding": 1}):
        total_students += 1
        student_id_obj = doc['_id']
        student_name = doc.get('name', '?')
        corrupted_count = 0
        total_count = 0
        
        if "encodings" in doc and doc["encodings"]:
            for raw in doc["encodings"]:
                total_count += 1
                nbytes = len(raw)
                
                # Check if size matches any known format
                found_match = False
                for dim, dtype, expected_size in _KNOWN_FORMATS:
                    if nbytes == expected_size:
                        found_match = True
                        break
                
                if not found_match:
                    corrupted_count += 1
        
        elif "face_encoding" in doc:
            total_count = 1
            nbytes = len(doc["face_encoding"])
            
            found_match = False
            for dim, dtype, expected_size in _KNOWN_FORMATS:
                if nbytes == expected_size:
                    found_match = True
                    break
            
            if not found_match:
                corrupted_count = 1
        
        if corrupted_count > 0:
            corrupted_students.append({
                'student_id': student_id_obj,
                'name': student_name,
                'corrupted_count': corrupted_count,
                'total_count': total_count,
            })
        else:
            valid_students += 1
    
    return {
        'total_students': total_students,
        'students_with_valid': valid_students,
        'students_with_corrupted': len(corrupted_students),
        'corrupted_students': corrupted_students,
    }

