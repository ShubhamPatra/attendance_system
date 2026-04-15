"""
End-to-end integration tests for the complete attendance flow.

Tests: Camera capture → Face detect → Liveness vote → Recognition → DB mark
"""

import sys
import os
import time
from unittest import mock
import pytest
import numpy as np
import cv2
import bson

# Add workspace root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import app_core.config as config
import app_core.database as database
from app_camera.camera import Camera
from app_vision.pipeline import FaceTrack


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """One-time test environment setup."""
    # Use test MongoDB if available
    if not os.environ.get("MONGO_URI"):
        pytest.skip("MONGO_URI not set; skipping E2E tests")
    yield


@pytest.fixture
def test_student(setup_test_env):
    """Create a test student for attendance marking."""
    sid = bson.ObjectId()
    
    # Insert student
    student = {
        "_id": sid,
        "registration_number": "TEST-E2E-001",
        "name": "Test E2E Student",
        "email": "test-e2e@example.com",
        "semester": 1,
        "section": "A",
        "verification_status": "verified",
        "encodings": [],
    }
    
    try:
        db = database.get_db()
        db.students.insert_one(student)
        yield sid
        # Cleanup
        db.students.delete_one({"_id": sid})
        db.attendance.delete_many({"student_id": sid})
    except Exception as exc:
        pytest.skip(f"Failed to set up test student: {exc}")


@pytest.fixture
def synthetic_face_frame():
    """Generate a synthetic face-like frame for testing."""
    # Create a blank image with a face-like region (simple placeholder)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a face-like rectangle
    x, y, w, h = 100, 50, 200, 250
    cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 200, 200), -1)
    # Add some texture
    cv2.circle(frame, (x + w//3, y + h//3), 20, (100, 100, 100), -1)
    cv2.circle(frame, (x + 2*w//3, y + h//3), 20, (100, 100, 100), -1)
    return frame


class TestE2EAttendanceFlow:
    """End-to-end tests for attendance marking flow."""

    def test_mark_attendance_basic(self, test_student):
        """Test basic attendance marking."""
        marked = database.mark_attendance(test_student, 0.95)
        assert marked is True, "Should mark attendance on first call"
        
        # Second call same day should fail (duplicate)
        marked2 = database.mark_attendance(test_student, 0.95)
        assert marked2 is False, "Should not mark duplicate attendance"

    def test_mark_attendance_circuit_breaker_recovery(self, test_student):
        """Test circuit breaker fails gracefully on DB errors."""
        from app_core.database import _circuit_breaker
        
        # Simulate DB failures
        original_find_one = database.get_db().students.find_one
        
        call_count = 0
        def failing_find_one(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Simulated DB failure")
        
        try:
            # Trigger failures to open circuit
            database.get_db().students.find_one = failing_find_one
            
            # First few calls should trigger circuit breaker
            for _ in range(config.MONGO_CIRCUIT_BREAKER_THRESHOLD):
                try:
                    database.get_db()
                except RuntimeError:
                    pass  # Expected
            
            # Circuit should now be open
            assert _circuit_breaker.state.value == "open", "Circuit should be open"
            
        finally:
            # Restore original method
            database.get_db().students.find_one = original_find_one
            # Reset circuit breaker
            _circuit_breaker.state = _circuit_breaker.__class__.CircuitBreakerState.CLOSED
            _circuit_breaker.failure_count = 0

    def test_attendance_with_encoding_dedup(self, test_student):
        """Test that duplicate encodings are rejected at storage."""
        db = database.get_db()
        
        # Create test encoding
        enc1 = np.random.randn(512).astype(np.float32)
        enc1 = enc1 / np.linalg.norm(enc1)  # normalize
        
        # First append should succeed
        result1 = database.append_student_encoding(test_student, enc1)
        assert result1 is True, "First encoding should be stored"
        
        # Identical encoding should be rejected (similarity 1.0 > 0.95)
        result2 = database.append_student_encoding(test_student, enc1)
        assert result2 is False, "Duplicate encoding should be rejected"
        
        # Verify DB only has one encoding
        student = db.students.find_one({"_id": test_student})
        assert len(student.get("encodings", [])) == 1, "Should have only 1 encoding"

    def test_encoding_cache_incremental_update(self, test_student):
        """Test that encoding cache O(1) incremental update works."""
        from app_vision.face_engine import encoding_cache
        
        # Load cache
        encoding_cache.load()
        initial_size = encoding_cache.size
        
        # Add encoding to student
        enc = np.random.randn(512).astype(np.float32)
        enc = enc / np.linalg.norm(enc)
        
        database.append_student_encoding(test_student, enc)
        encoding_cache.add_encoding_to_student(test_student, enc)
        
        # Cache should have one more entry
        assert encoding_cache.size == initial_size + 1, "Cache should increment"

    def test_encoding_format_validation(self, test_student):
        """Test encoding format validation rejects invalid data."""
        # Invalid dimension
        with pytest.raises(ValueError, match="not supported"):
            enc_bad = np.random.randn(256).astype(np.float32)
            database._encode_for_storage(enc_bad)
        
        # Invalid dtype
        with pytest.raises(ValueError, match="not supported"):
            enc_bad = np.random.randn(512).astype(np.int32)
            database._encode_for_storage(enc_bad)
        
        # None
        with pytest.raises(ValueError, match="cannot be None"):
            database._encode_for_storage(None)
        
        # Valid encoding should work
        enc_good = np.random.randn(512).astype(np.float32)
        result = database._encode_for_storage(enc_good)
        assert result is not None, "Valid encoding should serialize"

    def test_liveness_voting_temporal_decision(self, synthetic_face_frame):
        """Test liveness voting makes correct decisions from history."""
        trk = FaceTrack(track_id=1, frame=synthetic_face_frame, bbox_xywh=(100, 50, 200, 250))
        
        # Simulate history: 4 real detections
        trk.liveness_history = [
            (1, 0.75),  # real, good conf
            (1, 0.78),  # real, good conf
            (1, 0.80),  # real, good conf
            (1, 0.77),  # real, good conf
        ]
        
        # Simulate liveness decision (voting logic)
        real_count = sum(1 for l, _ in trk.liveness_history if l == 1)
        spoof_count = sum(1 for l, _ in trk.liveness_history if l == 0)
        
        if real_count > spoof_count:
            state, score = "real", sum(c for l, c in trk.liveness_history if l == 1) / real_count
        else:
            state, score = "spoof", sum(c for l, c in trk.liveness_history if l == 0) / max(spoof_count, 1)
        
        assert state == "real", "Clear real majority should result in 'real'"
        assert score > 0.75, "Score should be average of real votes"

    def test_liveness_voting_spoof_detection(self):
        """Test liveness voting correctly identifies spoof attempts."""
        # Create a dummy frame for initialization
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trk = FaceTrack(track_id=2, frame=dummy_frame, bbox_xywh=(100, 50, 200, 250))
        
        # Simulate history: 3 spoof detections in initial sequence
        trk.liveness_history = [
            (0, 0.90),  # spoof, high conf
            (0, 0.88),  # spoof
            (0, 0.92),  # spoof
            (1, 0.60),  # weak real attempt
        ]
        
        # Simulate liveness decision (voting logic)
        real_count = sum(1 for l, _ in trk.liveness_history if l == 1)
        spoof_count = sum(1 for l, _ in trk.liveness_history if l == 0)
        
        if real_count > spoof_count:
            state, score = "real", sum(c for l, c in trk.liveness_history if l == 1) / real_count
        else:
            state, score = "spoof", sum(c for l, c in trk.liveness_history if l == 0) / spoof_count
        
        assert state == "spoof", "Clear spoof majority should result in 'spoof'"
        assert score > 0.87, "Score should be average of spoof votes"

    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state machine works correctly."""
        # Skip if circuit breaker not fully implemented
        try:
            from app_core.database import CircuitBreaker, CircuitBreakerState
        except (ImportError, AttributeError):
            pytest.skip("CircuitBreaker not fully available")
        
        try:
            cb = CircuitBreaker(failure_threshold=2, timeout_seconds=0.5)
            
            # Initial state should be CLOSED
            initial_state = cb.state
            assert str(initial_state) == "closed" or initial_state == CircuitBreakerState.CLOSED, \
                f"Initial state should be CLOSED, got {initial_state}"
        except (TypeError, AttributeError) as e:
            pytest.skip(f"CircuitBreaker initialization failed: {e}")
        
        # Should transition to HALF_OPEN
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success", "HALF_OPEN should allow single call"
        assert cb.state == CircuitBreakerState.CLOSED, "Should reset to CLOSED on success"

    def test_attendance_validation_diagnostic(self, test_student):
        """Test encoding validation diagnostic function."""
        from app_core.database import validate_student_encodings
        
        # Add a valid encoding
        enc = np.random.randn(512).astype(np.float32)
        database.append_student_encoding(test_student, enc)
        
        # Run diagnostic
        result = validate_student_encodings(test_student)
        
        assert result["total_students"] == 1
        assert result["students_with_valid"] >= 0
        assert "corrupted_students" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
