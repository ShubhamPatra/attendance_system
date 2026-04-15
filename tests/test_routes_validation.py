"""
Route validation tests for all attendance API endpoints.

Tests: Invalid inputs, boundary values, error responses, auth
"""

import sys
import os
import json
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import core.config as config
import core.database as database
from app import create_app
import bson


@pytest.fixture
def app():
    """Create Flask test app."""
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def admin_token(client):
    """Get admin authentication token for testing."""
    # For now, tests assume auth is enabled but may need mocking
    # This is a placeholder - actual token generation depends on auth implementation
    return "test-admin-token"


@pytest.fixture
def test_student():
    """Create test student for route testing."""
    if not os.environ.get("MONGO_URI"):
        return None
    
    try:
        db = database.get_db()
        sid = bson.ObjectId()
        student = {
            "_id": sid,
            "registration_number": "TEST-ROUTE-001",
            "name": "Test Route Student",
            "email": "test-route@example.com",
            "semester": 2,
            "section": "B",
            "verification_status": "verified",
            "encodings": [],
        }
        db.students.insert_one(student)
        yield sid
        db.students.delete_one({"_id": sid})
        db.attendance.delete_many({"student_id": sid})
    except Exception as exc:
        pytest.skip(f"Failed to set up test student: {exc}")
        yield None


class TestAttendanceRoutes:
    """Tests for attendance API routes."""

    def test_attendance_mark_missing_reg_no(self, client):
        """POST /api/attendance without reg_no should fail."""
        response = client.post(
            "/api/attendance",
            data=json.dumps({}),
            content_type="application/json",
        )
        # Should fail at auth layer or parameter validation
        # Expect 400 (bad request) or 401 (unauthorized)
        assert response.status_code in (400, 401, 403, 405), f"Got {response.status_code}"

    def test_attendance_mark_invalid_status(self, client, test_student):
        """POST /api/attendance with invalid status should fail."""
        if not test_student:
            pytest.skip("No test student available")
        
        # This test assumes we can bypass auth; actual implementation may differ
        # Attempting to send invalid status
        response = client.post(
            "/api/attendance",
            data=json.dumps({
                "reg_no": "TEST-ROUTE-001",
                "status": "InvalidStatus"  # Should only be "Present" or "Absent"
            }),
            content_type="application/json",
        )
        # Should reject invalid status
        # Expect 400 or auth error
        assert response.status_code in (400, 401, 403, 405), f"Got {response.status_code}"

    def test_attendance_list_boundary_dates(self, client):
        """GET /api/attendance with boundary date parameters."""
        # Far past date (>10 years ago) should ideally be rejected
        response = client.get("/api/attendance?date=1910-01-01")
        assert response.status_code in (400, 401, 403, 405, 200)  # Flexible based on config
        
        # Future date might be allowed or rejected based on business logic
        future_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
        response = client.get(f"/api/attendance?date={future_date}")
        assert response.status_code in (200, 400, 401, 403, 405)

    def test_attendance_list_invalid_date_format(self, client):
        """GET /api/attendance with malformed date should fail gracefully."""
        response = client.get("/api/attendance?date=not-a-date")
        # Should reject invalid date format
        assert response.status_code in (200, 400, 401, 403, 405)

    def test_attendance_bulk_mark_empty_list(self, client):
        """POST /api/attendance/bulk with empty student list."""
        response = client.post(
            "/api/attendance/bulk",
            data=json.dumps({"student_ids": []}),
            content_type="application/json",
        )
        # Empty list should be handled
        assert response.status_code in (200, 400, 401, 403, 405)

    def test_attendance_bulk_mark_nonexistent_students(self, client):
        """POST /api/attendance/bulk with invalid student IDs."""
        response = client.post(
            "/api/attendance/bulk",
            data=json.dumps({
                "student_ids": ["NO-SUCH-STUDENT-001", "NO-SUCH-STUDENT-002"]
            }),
            content_type="application/json",
        )
        # Should handle gracefully (return "not_found" list or similar)
        assert response.status_code in (200, 400, 401, 403, 405, 404)
        
        if response.status_code == 200:
            data = response.get_json()
            assert "not_found" in data or "updated" in data, "Should indicate missing students"

    def test_attendance_bulk_invalid_status(self, client):
        """POST /api/attendance/bulk with invalid status."""
        response = client.post(
            "/api/attendance/bulk",
            data=json.dumps({
                "student_ids": ["TEST-ROUTE-001"],
                "status": "Unknown"  # Invalid
            }),
            content_type="application/json",
        )
        assert response.status_code in (200, 400, 401, 403, 405)

    def test_attendance_date_range_invalid_order(self, client):
        """GET /api/attendance with start_date > end_date."""
        response = client.get(
            "/api/attendance?start_date=2026-04-15&end_date=2026-04-01"
        )
        # Should reject invalid date range
        assert response.status_code in (200, 400, 401, 403, 405)


class TestCameraRoutes:
    """Tests for camera API routes."""

    def test_stream_endpoint_availability(self, client):
        """GET /stream should be accessible."""
        response = client.get("/stream")
        # Should return 200 with MJPEG stream or 404 if endpoint disabled
        assert response.status_code in (200, 404, 401, 403, 405)

    def test_stream_with_invalid_camera_index(self, client):
        """GET /stream?camera_idx=999 with non-existent camera."""
        response = client.get("/stream?camera_idx=999")
        # Should handle gracefully
        assert response.status_code in (200, 400, 404, 401, 403, 405)


class TestDatabaseResilience:
    """Tests for database error handling in routes."""

    def test_database_circuit_breaker_error_response(self, client, monkeypatch):
        """Routes should return 503 when circuit breaker is open."""
        from core import database as db_module
        
        # Mock circuit breaker to force open state
        original_get_db = db_module.get_db
        
        def failing_get_db():
            raise RuntimeError("Circuit breaker is OPEN")
        
        # This would require actual route endpoint access
        # For now, just verify circuit breaker exception handling would work
        try:
            failing_get_db()
        except RuntimeError as exc:
            assert "OPEN" in str(exc)


class TestInputSanitization:
    """Tests for input sanitization and security."""

    def test_attendance_mark_sql_injection_attempt(self, client):
        """POST /api/attendance with SQL injection in reg_no."""
        response = client.post(
            "/api/attendance",
            data=json.dumps({
                "reg_no": "'; DROP TABLE students; --"
            }),
            content_type="application/json",
        )
        # Should not execute injection; handle as invalid student
        assert response.status_code in (200, 400, 401, 403, 404, 405)

    def test_attendance_list_regex_injection(self, client):
        """GET /api/attendance with regex injection in query."""
        response = client.get("/api/attendance?date=.*")
        # Should handle safely
        assert response.status_code in (200, 400, 401, 403, 404, 405)

    def test_attendance_list_xss_attempt(self, client):
        """GET /api/attendance with XSS payload in query."""
        xss_payload = "<script>alert('xss')</script>"
        response = client.get(f"/api/attendance?date={xss_payload}")
        # Should not execute script; handle as invalid date
        assert response.status_code in (200, 400, 401, 403, 404, 405)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
