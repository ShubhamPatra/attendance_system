"""
Security event logging and anti-cheat detection.

PHASE 5: Logs suspicious activities including:
- Spoof attempts (detected by liveness check)
- Multi-identity detection (multiple students detected from same person)
- Abnormal attendance patterns (repeated early/late arrivals, skipped classes)
- Failed recognition attempts
- Liveness uncertainty

Stores events in MongoDB 'security_logs' collection with timestamp, camera ID, student ID, etc.
Used for audit trails and anomaly detection dashboards.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import bson

import core.config as config
from core.utils import setup_logging
import core.database as database

logger = setup_logging()


class SecurityEventType(Enum):
    """Types of security events to log."""
    SPOOF_ATTEMPT = "spoof_attempt"
    MULTI_IDENTITY = "multi_identity"
    FAILED_MATCH = "failed_match"
    LIVENESS_UNCERTAIN = "liveness_uncertain"
    REPEATED_SPOOF = "repeated_spoof"
    ABNORMAL_PATTERN = "abnormal_pattern"
    ENROLLMENT_FRAUD = "enrollment_fraud"
    DUPLICATE_ATTENDANCE = "duplicate_attendance"


class SecurityLogger:
    """Logger for security events.
    
    Thread-safe event logging to MongoDB.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._event_buffer: List[Dict[str, Any]] = []
        self._buffer_max_size = 100
        self._last_flush_time = time.monotonic()
        self._flush_interval_seconds = 5.0

    def log_event(
        self,
        event_type: SecurityEventType,
        student_id: Optional[bson.ObjectId] = None,
        camera_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "medium",
    ) -> None:
        """Log a security event.
        
        Args:
            event_type: Type of security event
            student_id: ID of student involved (if applicable)
            camera_id: Camera ID where event occurred
            details: Additional context (dict)
            severity: "low", "medium", "high", "critical"
        """
        now = datetime.utcnow()
        event = {
            "type": event_type.value if isinstance(event_type, SecurityEventType) else str(event_type),
            "timestamp": now,
            "camera_id": camera_id,
            "student_id": student_id,
            "severity": severity,
            "details": details or {},
            "recorded_at": datetime.utcnow(),
        }

        with self._lock:
            self._event_buffer.append(event)
            
            # Auto-flush if buffer full or interval exceeded
            should_flush = (
                len(self._event_buffer) >= self._buffer_max_size or
                (time.monotonic() - self._last_flush_time) > self._flush_interval_seconds
            )

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush buffered events to database."""
        with self._lock:
            if not self._event_buffer:
                return

            events_to_flush = self._event_buffer[:]
            self._event_buffer.clear()
            self._last_flush_time = time.monotonic()

        try:
            db = database.get_db()
            if events_to_flush:
                db.security_logs.insert_many(events_to_flush)
                logger.debug("Flushed %d security events to database", len(events_to_flush))
        except Exception as e:
            logger.error("Failed to flush security events: %s", e)
            # Re-buffer events on failure
            with self._lock:
                self._event_buffer.extend(events_to_flush)

    def log_spoof_attempt(
        self,
        camera_id: int,
        student_id: Optional[bson.ObjectId] = None,
        liveness_score: float = 0.0,
        reason: str = "",
    ) -> None:
        """Log a spoof detection attempt."""
        self.log_event(
            SecurityEventType.SPOOF_ATTEMPT,
            student_id=student_id,
            camera_id=camera_id,
            details={
                "liveness_score": round(liveness_score, 4),
                "reason": reason,
            },
            severity="high",
        )

    def log_multi_identity(
        self,
        camera_id: int,
        candidates: List[Dict[str, Any]],
    ) -> None:
        """Log detection of multiple identities from same face.
        
        Args:
            camera_id: Camera where detected
            candidates: List of {"student_id": ..., "confidence": ..., "name": ...}
        """
        self.log_event(
            SecurityEventType.MULTI_IDENTITY,
            camera_id=camera_id,
            details={
                "candidates": [
                    {
                        "student_id": str(c.get("student_id", "")),
                        "name": c.get("name", ""),
                        "confidence": round(c.get("confidence", 0.0), 4),
                    }
                    for c in candidates
                ]
            },
            severity="medium",
        )

    def log_failed_match(
        self,
        camera_id: int,
        reason: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Log a failed face match attempt."""
        self.log_event(
            SecurityEventType.FAILED_MATCH,
            camera_id=camera_id,
            details={
                "reason": reason,
                "confidence": round(confidence, 4),
            },
            severity="low",
        )

    def log_liveness_uncertain(
        self,
        camera_id: int,
        student_id: Optional[bson.ObjectId] = None,
        liveness_score: float = 0.0,
    ) -> None:
        """Log liveness score uncertainty (borderline cases)."""
        self.log_event(
            SecurityEventType.LIVENESS_UNCERTAIN,
            student_id=student_id,
            camera_id=camera_id,
            details={
                "liveness_score": round(liveness_score, 4),
            },
            severity="low",
        )

    def log_repeated_spoof(
        self,
        camera_id: int,
        student_id: Optional[bson.ObjectId] = None,
        attempt_count: int = 0,
        time_window_minutes: int = 10,
    ) -> None:
        """Log repeated spoof attempts from same source."""
        self.log_event(
            SecurityEventType.REPEATED_SPOOF,
            student_id=student_id,
            camera_id=camera_id,
            details={
                "attempt_count": attempt_count,
                "time_window_minutes": time_window_minutes,
            },
            severity="critical",
        )

    def log_abnormal_pattern(
        self,
        student_id: bson.ObjectId,
        pattern_type: str,
        details_dict: Dict[str, Any],
    ) -> None:
        """Log abnormal attendance patterns."""
        self.log_event(
            SecurityEventType.ABNORMAL_PATTERN,
            student_id=student_id,
            details={
                "pattern": pattern_type,
                **details_dict,
            },
            severity="medium",
        )

    def log_enrollment_fraud(
        self,
        student_id: bson.ObjectId,
        reason: str = "",
    ) -> None:
        """Log suspected enrollment fraud."""
        self.log_event(
            SecurityEventType.ENROLLMENT_FRAUD,
            student_id=student_id,
            details={"reason": reason},
            severity="critical",
        )


class AnomalyDetector:
    """Detects anomalous attendance patterns.
    
    PHASE 5.2: Analyzes:
    - Repeated spoof attempts from same camera/student
    - Abnormal attendance times (very early/very late)
    - Dropout detection (absent > N days in row)
    - Impossible attendance (present at multiple cameras simultaneously)
    """

    def __init__(self):
        self._lock = threading.Lock()

    def detect_repeated_spoofs(
        self,
        camera_id: int,
        student_id: Optional[bson.ObjectId] = None,
        time_window_minutes: int = 10,
        attempt_threshold: int = 3,
    ) -> bool:
        """Check if student has attempted spoof >N times in last M minutes.
        
        Returns True if repeated spoofs detected.
        """
        try:
            db = database.get_db()
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)

            query = {
                "type": "spoof_attempt",
                "timestamp": {"$gte": cutoff_time},
                "camera_id": camera_id,
            }
            if student_id is not None:
                query["student_id"] = student_id

            count = db.security_logs.count_documents(query)
            return count >= attempt_threshold
        except Exception as e:
            logger.warning("Failed to detect repeated spoofs: %s", e)
            return False

    def detect_late_arrival(self, arrival_time: datetime, threshold_hour: int = 12) -> bool:
        """Check if arrival time is suspiciously late."""
        return arrival_time.hour >= threshold_hour

    def detect_early_arrival(self, arrival_time: datetime, threshold_hour: int = 6) -> bool:
        """Check if arrival time is suspiciously early."""
        return arrival_time.hour < threshold_hour

    def detect_dropout(
        self,
        student_id: bson.ObjectId,
        absent_day_threshold: int = 3,
    ) -> bool:
        """Check if student has been absent for >N consecutive days.
        
        Returns True if dropout detected.
        """
        try:
            db = database.get_db()

            # Get last N+1 days of attendance records
            today = datetime.utcnow().date()
            days_to_check = absent_day_threshold + 1

            pipeline = [
                {
                    "$match": {
                        "student_id": student_id,
                        "date": {
                            "$gte": (today - timedelta(days=days_to_check)).isoformat(),
                            "$lte": today.isoformat(),
                        },
                    }
                },
                {"$sort": {"date": -1}},
                {"$limit": days_to_check},
                {"$group": {"_id": None, "dates": {"$push": "$date"}}},
            ]

            result = list(db.attendance.aggregate(pipeline))
            if not result:
                return True  # No recent attendance = potential dropout

            dates_present = set(result[0].get("dates", []))

            # Check for consecutive absences
            consecutive_absences = 0
            for i in range(days_to_check):
                check_date = (today - timedelta(days=i)).isoformat()
                if check_date not in dates_present:
                    consecutive_absences += 1
                    if consecutive_absences >= absent_day_threshold:
                        return True
                else:
                    consecutive_absences = 0

            return False
        except Exception as e:
            logger.warning("Failed to detect dropout for %s: %s", student_id, e)
            return False

    def detect_impossible_attendance(
        self,
        student_id: bson.ObjectId,
        camera_ids: List[int],
        time_window_minutes: int = 5,
    ) -> bool:
        """Check if student was marked present at multiple cameras within time window.
        
        Returns True if impossible attendance detected.
        """
        try:
            db = database.get_db()
            today = datetime.utcnow().date().isoformat()

            # Get all attendance records for this student today
            records = list(db.attendance.find({
                "student_id": student_id,
                "date": today,
            }, {"time": 1, "camera_id": 1}).sort("time", 1))

            if len(records) < 2:
                return False

            # Check for impossible transitions
            for i in range(len(records) - 1):
                time1 = records[i].get("time", "")
                time2 = records[i + 1].get("time", "")

                # Parse times
                try:
                    t1 = datetime.strptime(time1, "%H:%M:%S")
                    t2 = datetime.strptime(time2, "%H:%M:%S")
                    time_diff = (t2 - t1).total_seconds() / 60
                except:
                    continue

                if 0 < time_diff < time_window_minutes:
                    cam1 = records[i].get("camera_id")
                    cam2 = records[i + 1].get("camera_id")
                    if cam1 != cam2:
                        return True  # Same student at different cameras too quickly

            return False
        except Exception as e:
            logger.warning("Failed to detect impossible attendance: %s", e)
            return False


# Global singleton instances
_security_logger = SecurityLogger()
_anomaly_detector = AnomalyDetector()


def get_security_logger() -> SecurityLogger:
    """Get global security logger instance."""
    return _security_logger


def get_anomaly_detector() -> AnomalyDetector:
    """Get global anomaly detector instance."""
    return _anomaly_detector


def log_spoof_attempt(camera_id: int, student_id=None, liveness_score: float = 0.0, reason: str = "") -> None:
    """Log a spoof detection."""
    _security_logger.log_spoof_attempt(camera_id, student_id, liveness_score, reason)


def log_multi_identity(camera_id: int, candidates: List[Dict[str, Any]]) -> None:
    """Log multiple identity detection."""
    _security_logger.log_multi_identity(camera_id, candidates)


def log_failed_match(camera_id: int, reason: str = "", confidence: float = 0.0) -> None:
    """Log failed recognition match."""
    _security_logger.log_failed_match(camera_id, reason, confidence)


def log_liveness_uncertain(camera_id: int, student_id=None, liveness_score: float = 0.0) -> None:
    """Log uncertain liveness score."""
    _security_logger.log_liveness_uncertain(camera_id, student_id, liveness_score)


def log_abnormal_pattern(student_id: bson.ObjectId, pattern_type: str, details: Dict[str, Any]) -> None:
    """Log abnormal attendance pattern."""
    _security_logger.log_abnormal_pattern(student_id, pattern_type, details)


def flush_security_logs() -> None:
    """Flush buffered security events to database."""
    _security_logger.flush()
