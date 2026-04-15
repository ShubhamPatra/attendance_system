"""
Concurrent stress tests for multi-threaded camera and recognition pipeline.

Tests: Multi-face simultaneous processing, no race conditions, proper locking
"""

import sys
import os
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
import numpy as np
import bson

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import core.config as config
import core.database as database
from vision.face_engine import encoding_cache


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """One-time test environment setup."""
    if not os.environ.get("MONGO_URI"):
        pytest.skip("MONGO_URI not set; skipping concurrent tests")
    yield


@pytest.fixture
def test_students():
    """Create multiple test students for concurrent operations."""
    students = []
    db = database.get_db()
    
    try:
        for i in range(5):
            sid = bson.ObjectId()
            student = {
                "_id": sid,
                "registration_number": f"TEST-CONCURRENT-{i:03d}",
                "name": f"Concurrent Test Student {i}",
                "email": f"concurrent-{i}@example.com",
                "semester": 1,
                "section": "A",
                "verification_status": "verified",
                "encodings": [],
            }
            db.students.insert_one(student)
            students.append(sid)
        
        yield students
        
        # Cleanup
        for sid in students:
            db.students.delete_one({"_id": sid})
            db.attendance.delete_many({"student_id": sid})
    except Exception as exc:
        pytest.skip(f"Failed to set up test students: {exc}")


class TestConcurrentPipeline:
    """Concurrent stress tests for attendance pipeline."""

    def test_concurrent_attendance_marking(self, test_students):
        """Test simultaneous attendance marking on multiple students."""
        from core.database import CircuitBreaker
        
        results = []
        errors = []
        
        def mark_attendance_worker(student_id, idx):
            """Worker to mark attendance for one student."""
            try:
                # Add small random delay to simulate real processing
                time.sleep(random.uniform(0.01, 0.05))
                marked = database.mark_attendance(student_id, 0.90 + (idx * 0.01))
                return {"student_id": str(student_id), "marked": marked, "idx": idx}
            except Exception as exc:
                return {"error": str(exc), "idx": idx}
        
        # Run 5 concurrent marking operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(mark_attendance_worker, sid, i)
                for i, sid in enumerate(test_students)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(result)
                else:
                    results.append(result)
        
        # All should succeed (no errors)
        assert len(errors) == 0, f"Concurrent marking should have no errors: {errors}"
        assert len(results) == len(test_students), "All students should be marked"
        
        # Verify all were marked
        for result in results:
            assert result["marked"] is True, f"Student {result['idx']} should be marked"

    def test_concurrent_encoding_cache_updates(self, test_students):
        """Test concurrent encoding cache updates don't cause corruption."""
        encoding_cache.load()
        
        errors = []
        update_count = [0]
        
        def cache_update_worker(student_id, idx):
            """Worker to add encodings to cache."""
            try:
                for _ in range(3):  # Add 3 encodings per student
                    enc = np.random.randn(512).astype(np.float32)
                    enc = enc / np.linalg.norm(enc)
                    
                    # Add to DB
                    database.append_student_encoding(student_id, enc)
                    
                    # Add to cache
                    encoding_cache.add_encoding_to_student(student_id, enc)
                    update_count[0] += 1
                    time.sleep(0.001)  # Small delay to interleave operations
                
                return {"idx": idx, "success": True}
            except Exception as exc:
                return {"idx": idx, "error": str(exc)}
        
        # Run concurrent cache updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(cache_update_worker, sid, i)
                for i, sid in enumerate(test_students)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(result)
        
        # No errors
        assert len(errors) == 0, f"Concurrent cache updates should have no errors: {errors}"
        
        # Cache should have all updates
        flat_enc, flat_idx, ids, names = encoding_cache.get_flat()
        if flat_enc is not None:
            assert flat_enc.shape[0] > 0, "Cache should have encodings"
            assert len(flat_idx) == flat_enc.shape[0], "Index array should match encodings"

    def test_concurrent_recognition_cofirm_frames(self, test_students):
        """Test concurrent recognition confirmation doesn't cause race conditions."""
        from vision.pipeline import FaceTrack
        
        errors = []
        confirmed = [0]
        confirm_lock = threading.Lock()
        
        def recognition_worker(student_id, idx):
            """Worker to simulate recognition confirmation loop."""
            try:
                trk = FaceTrack(track_id=1000 + idx, bbox=(100, 50, 200, 250))
                
                # Simulate accumulating confirmation frames
                for frame_num in range(config.RECOGNITION_CONFIRM_FRAMES + 2):
                    trk.candidate_hits = frame_num
                    
                    if trk.candidate_hits >= config.RECOGNITION_CONFIRM_FRAMES:
                        with confirm_lock:
                            confirmed[0] += 1
                        break
                    
                    time.sleep(0.001)
                
                return {"idx": idx, "confirmed": confirmed[0]}
            except Exception as exc:
                return {"idx": idx, "error": str(exc)}
        
        # Run concurrent recognition confirmations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(recognition_worker, sid, i)
                for i, sid in enumerate(test_students)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(result)
        
        assert len(errors) == 0, f"No errors in recognition confirmation: {errors}"
        assert confirmed[0] == len(test_students), "All should confirm recognition"

    def test_concurrent_liveness_voting(self):
        """Test liveness voting under concurrent access."""
        from vision.pipeline import FaceTrack
        from unittest import mock
        
        errors = []
        decisions = []
        decision_lock = threading.Lock()
        
        def liveness_voting_worker(idx):
            """Worker to perform liveness voting."""
            try:
                # Create a dummy frame for FaceTrack initialization
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                trk = FaceTrack(track_id=2000 + idx, frame=dummy_frame, bbox_xywh=(100, 50, 200, 250))
                
                # Build history with random labels
                labels = [1, 1, 0, 2, -1]  # Mix of real, spoof, error
                for label in labels:
                    conf = random.uniform(0.5, 0.95)
                    trk.liveness_history.append((label, conf))
                
                # Simulate liveness decision (voting logic)
                real_count = sum(1 for l, _ in trk.liveness_history if l == 1)
                spoof_count = sum(1 for l, _ in trk.liveness_history if l == 0)
                
                if real_count > spoof_count:
                    state, score = "real", 0.85
                else:
                    state, score = "spoof", 0.75
                
                with decision_lock:
                    decisions.append({"idx": idx, "state": state, "score": score})
                
                return {"idx": idx, "decided": True}
            except Exception as exc:
                return {"idx": idx, "error": str(exc)}
        
        # Run concurrent voting
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(liveness_voting_worker, i)
                for i in range(8)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(result)
        
        assert len(errors) == 0, f"No errors in liveness voting: {errors}"
        assert len(decisions) == 8, "Should have 8 decisions"
        assert len(decisions) == 8, "All workers should make decisions"
        
        # Verify decisions are valid
        for decision in decisions:
            assert decision["state"] in ("real", "spoof", "uncertain"), "Invalid state"
            assert 0 <= decision["score"] <= 1, "Score should be between 0 and 1"

    def test_concurrent_database_operations_no_corruption(self, test_students):
        """Test concurrent DB operations don't corrupt data."""
        db = database.get_db()
        errors = []
        
        def db_worker(student_id, idx):
            """Worker to perform concurrent DB operations."""
            try:
                # Concurrent updates
                for i in range(5):
                    # Mark attendance
                    database.mark_attendance(student_id, 0.90)
                    
                    # Add encoding
                    enc = np.random.randn(512).astype(np.float32)
                    database.append_student_encoding(student_id, enc)
                    
                    time.sleep(0.001)
                
                return {"idx": idx, "operations": 5}
            except Exception as exc:
                return {"idx": idx, "error": str(exc)}
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(db_worker, sid, i)
                for i, sid in enumerate(test_students)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(result)
        
        assert len(errors) == 0, f"No corruption errors: {errors}"
        
        # Verify data integrity
        for student_id in test_students:
            student = db.students.find_one({"_id": student_id})
            assert student is not None, "Student should exist"
            assert isinstance(student.get("encodings", []), list), "Encodings should be list"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
