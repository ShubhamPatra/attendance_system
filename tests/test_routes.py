"""
Tests for Flask routes.
Uses the Flask test client — no live server or webcam needed.
"""

import io
import os
import sys
import base64
import json
import tempfile
import types
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_fr = types.SimpleNamespace(
        face_locations=lambda *args, **kwargs: [],
        face_encodings=lambda *args, **kwargs: [],
        face_landmarks=lambda *args, **kwargs: [],
        load_image_file=lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "face_recognition", fake_fr)
    import importlib
    import app_core.config as config
    importlib.reload(config)


@pytest.fixture
def client():
    """Create a Flask test client with mocked DB calls."""
    with patch("app_core.database.get_client") as mock_client, \
         patch("app_core.database.ensure_indexes"), \
         patch("app_vision.face_engine.encoding_cache") as mock_cache, \
         patch("app_vision.anti_spoofing.init_models"):

        mock_ping = MagicMock()
        mock_client.return_value = MagicMock()
        mock_cache.load = MagicMock()
        mock_cache.size = 0
        mock_cache.get_all.return_value = ([], [], [])

        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


# ── Home page ─────────────────────────────────────────────────────────────

def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Attendance" in resp.data


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


def test_ready_endpoint(client):
    resp = client.get("/healthz")
    assert resp.status_code in (200, 503)
    data = resp.get_json()
    assert "checks" in data


# ── Registration POST ─────────────────────────────────────────────────────

def test_register_missing_fields(client):
    """POST with empty form should return 400 with error flashes."""
    resp = client.post("/register", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_register_success(client):
    """POST with valid multi-image data should insert student and redirect."""
    # Create a tiny 100x100 white JPEG in memory
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, "JPEG")
    buf.seek(0)

    fake_encoding = np.random.rand(128).astype(np.float64)
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("app_web.routes.generate_encoding", return_value=fake_encoding), \
         patch("app_web.registration_routes.cv2.imread", return_value=fake_image), \
         patch("app_web.routes.check_image_quality", return_value=(True, "")), \
         patch("app_web.routes.database") as mock_db, \
         patch("app_web.routes.encoding_cache") as mock_cache:

        import bson
        mock_db.insert_student.return_value = bson.ObjectId()
        mock_cache.refresh = MagicMock()

        resp = client.post(
            "/register",
            data={
                "name": "Test Student",
                "semester": "3",
                "registration_number": "FA21-BCS-099",
                "section": "A",
                "images": (buf, "face.jpg"),
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        # Should redirect on success
        assert resp.status_code in (302, 303)


# ── Dashboard ─────────────────────────────────────────────────────────────

def test_dashboard_loads(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.student_count.return_value = 5
        mock_db.today_attendance_count.return_value = 3
        mock_db.get_attendance.return_value = []
        resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Dashboard" in resp.data


# ── Report page ───────────────────────────────────────────────────────────

def test_report_loads(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_attendance.return_value = []
        resp = client.get("/report?date=2026-02-26")
    assert resp.status_code == 200
    assert b"Report" in resp.data


# ── CSV export ────────────────────────────────────────────────────────────

def test_report_csv(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_attendance_csv.return_value = pd.DataFrame({
            "Name": ["Alice"],
            "Registration Number": ["FA21-BCS-001"],
            "Section": ["A"],
            "Semester": [3],
            "Date": ["2026-02-26"],
            "Time": ["09:00:00"],
            "Status": ["Present"],
            "Confidence": [0.92],
        })
        resp = client.get("/report/csv?date=2026-02-26")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/csv")
    assert b"Alice" in resp.data


# ── Registration validation branches ──────────────────────────────────────

def test_register_invalid_semester(client):
    """POST with non-integer semester should return 400."""
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, "JPEG")
    buf.seek(0)

    resp = client.post(
        "/register",
        data={
            "name": "Test",
            "semester": "abc",
            "registration_number": "FA-001",
            "section": "A",
            "images": (buf, "face.jpg"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_register_invalid_extension(client):
    """POST with .exe image should return 400."""
    buf = io.BytesIO(b"not an image")

    resp = client.post(
        "/register",
        data={
            "name": "Test",
            "semester": "3",
            "registration_number": "FA-002",
            "section": "A",
            "images": (buf, "virus.exe"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_register_duplicate_reg_no(client):
    """POST with duplicate reg number should return 409."""
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, "JPEG")
    buf.seek(0)

    fake_encoding = np.random.rand(128).astype(np.float64)
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("app_web.routes.generate_encoding", return_value=fake_encoding), \
         patch("app_web.registration_routes.cv2.imread", return_value=fake_image), \
         patch("app_web.routes.check_image_quality", return_value=(True, "")), \
         patch("app_web.routes.database") as mock_db, \
         patch("app_web.routes.encoding_cache"):
        mock_db.insert_student.side_effect = ValueError("already exists")

        resp = client.post(
            "/register",
            data={
                "name": "Dup Student",
                "semester": "3",
                "registration_number": "FA-DUP",
                "section": "A",
                "images": (buf, "face.jpg"),
            },
            content_type="multipart/form-data",
        )
    assert resp.status_code == 409


def test_register_encoding_failure(client):
    """POST where face encoding fails for all images should return 400."""
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buf, "JPEG")
    buf.seek(0)

    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("app_web.routes.generate_encoding", side_effect=ValueError("No face detected")), \
         patch("app_web.registration_routes.cv2.imread", return_value=fake_image), \
         patch("app_web.routes.check_image_quality", return_value=(True, "")):
        resp = client.post(
            "/register",
            data={
                "name": "Ghost",
                "semester": "3",
                "registration_number": "FA-GHOST",
                "section": "B",
                "images": (buf, "face.jpg"),
            },
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400


# ── Metrics endpoint ──────────────────────────────────────────────────────

def test_api_metrics(client):
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "accuracy_pct" in data
    assert "fps" in data


# ── New pages load ────────────────────────────────────────────────────────

def test_logs_page_loads(client):
    resp = client.get("/logs")
    assert resp.status_code == 200
    assert b"Recognition Logs" in resp.data


def test_metrics_page_loads(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.student_count.return_value = 0
        mock_db.today_attendance_count.return_value = 0
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"System Metrics" in resp.data


def test_attendance_activity_page_loads(client):
    resp = client.get("/attendance_activity")
    assert resp.status_code == 200
    assert b"Attendance Activity" in resp.data


# ── New API endpoints ─────────────────────────────────────────────────────

def test_api_logs(client):
    with patch("app_camera.camera.get_camera_if_running") as mock_cam:
        mock_cam.return_value.get_log_buffer.return_value = []
        resp = client.get("/api/logs")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_api_events_returns_empty_when_camera_not_running(client):
    with patch("app_camera.camera.get_camera_if_running", return_value=None):
        resp = client.get("/api/events")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_api_attendance_activity(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_attendance_by_hour.return_value = [
            {"hour": 9, "count": 4}
        ]
        resp = client.get("/api/attendance_activity?date=2026-02-26")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["date"] == "2026-02-26"
    assert len(data["hours"]) == 1


def test_api_registration_numbers(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_all_registration_numbers.return_value = [
            "FA21-BCS-001", "FA21-BCS-002"
        ]
        resp = client.get("/api/registration_numbers")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 2


# ── Extended CSV export ───────────────────────────────────────────────────

def test_report_csv_by_student(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_student_by_reg_no.return_value = {"_id": "x"}
        mock_db.get_attendance_csv_by_student.return_value = pd.DataFrame({
            "Name": ["Alice"], "Registration Number": ["FA21-BCS-001"],
            "Section": ["A"], "Semester": [3], "Date": ["2026-02-26"],
            "Time": ["09:00:00"], "Status": ["Present"], "Confidence": [0.92],
        })
        resp = client.get("/report/csv?reg_no=FA21-BCS-001")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/csv")


def test_report_csv_by_student_not_found(client):
    with patch("app_web.routes_helpers.database") as mock_db:
        mock_db.get_student_by_reg_no.return_value = None
        resp = client.get("/report/csv?reg_no=FA21-BCS-404")
    assert resp.status_code == 404


def test_report_csv_requires_both_range_dates(client):
    resp = client.get("/report/csv?start_date=2026-02-01")
    assert resp.status_code == 400
    assert "Both start_date and end_date are required" in resp.get_json()["error"]


def test_report_csv_full(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_attendance_csv_full.return_value = pd.DataFrame(
            columns=["Name", "Registration Number", "Section",
                     "Semester", "Date", "Time", "Status", "Confidence"]
        )
        resp = client.get("/report/csv?full=1")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/csv")


def test_report_xlsx_by_student_not_found(client):
    with patch("app_web.routes_helpers.database") as mock_db:
        mock_db.get_student_by_reg_no.return_value = None
        resp = client.get("/report/xlsx?reg_no=FA21-BCS-404")
    assert resp.status_code == 404


def test_report_xlsx_requires_both_range_dates(client):
    resp = client.get("/report/xlsx?start_date=2026-02-01")
    assert resp.status_code == 400
    assert "Both start_date and end_date are required" in resp.get_json()["error"]


def test_api_report_csv_async_invalid_date(client):
    with patch("celery_app.generate_csv_task") as mock_task:
        resp = client.post(
            "/api/report/csv/async",
            json={"date": "2026-13-99"},
        )
    assert resp.status_code == 400
    mock_task.delay.assert_not_called()


def test_api_report_csv_async_requires_both_range_dates(client):
    with patch("celery_app.generate_csv_task") as mock_task:
        resp = client.post(
            "/api/report/csv/async",
            json={"start_date": "2026-02-01"},
        )
    assert resp.status_code == 400
    mock_task.delay.assert_not_called()


def test_api_report_csv_async_student_not_found(client):
    with patch("app_web.routes_helpers.database") as mock_db, \
         patch("celery_app.generate_csv_task") as mock_task:
        mock_db.get_student_by_reg_no.return_value = None
        resp = client.post(
            "/api/report/csv/async",
            json={"reg_no": "FA21-BCS-404"},
        )
    assert resp.status_code == 404
    mock_task.delay.assert_not_called()


def test_api_debug_diagnostics_disabled_by_default(client):
    resp = client.get("/api/debug/diagnostics")
    assert resp.status_code == 404


def test_api_debug_diagnostics_when_enabled(client):
    with patch("app_web.routes.config.DEBUG_MODE", True), \
         patch("app_camera.camera.get_camera_diagnostics", return_value={"active_cameras": 0, "viewers": {}, "cameras": {}}), \
         patch("app_web.routes._check_mongo_ready", return_value=True), \
         patch("app_web.routes._check_celery_ready", return_value=False), \
         patch("app_web.routes._check_model_artifacts", return_value={"yunet_model": True, "anti_spoof_models": True, "ppe_model": True}):
        resp = client.get("/api/debug/diagnostics")

    assert resp.status_code == 200
    data = resp.get_json()
    assert "cameras" in data
    assert "metrics" in data
    assert "health" in data
    assert data["health"]["ppe_model"] is True


def test_api_register_capture_rejects_non_jpeg(client):
    payload = base64.b64encode(b"not-a-jpeg").decode("ascii")
    resp = client.post(
        "/api/register/capture",
        json={"frame": payload},
    )
    assert resp.status_code == 400


def test_api_register_capture_returns_upload_token(client):
    # Minimal JPEG header + footer bytes for endpoint validation.
    payload = base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app_web.routes.config.UPLOAD_DIR", tmpdir), \
            patch("app_web.registration_routes.cv2.imdecode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
        resp = client.post(
            "/api/register/capture",
            json={"frame": payload},
        )
    assert resp.status_code == 200
    assert "path" in resp.get_json()
    assert os.path.basename(resp.get_json()["path"]) == resp.get_json()["path"]


def test_api_students_create_rejects_outside_upload_dir(client):
    resp = client.post(
        "/api/students",
        json={
            "name": "Alice",
            "semester": 3,
            "registration_number": "FA21-BCS-123",
            "section": "A",
            "image_paths": ["C:/Windows/system32/drivers/etc/hosts"],
        },
    )
    assert resp.status_code == 400


def test_heatmap_page_loads(client):
    resp = client.get("/heatmap")
    assert resp.status_code == 200
    assert b"Heatmap" in resp.data


def test_api_heatmap(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_attendance_heatmap_data.return_value = [
            {"date": "2026-01-01", "count": 10, "total_students": 20}
        ]
        resp = client.get("/api/heatmap")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1


def test_api_at_risk(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_at_risk_students.return_value = [
            {"name": "Alice", "reg_no": "FA21-BCS-001", "percentage": 40.0,
             "days_present": 12, "days_total": 30}
        ]
        resp = client.get("/api/at_risk")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1


def test_student_portal_login(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_student_by_reg_no.return_value = {
            "_id": "test", "name": "Alice", "registration_number": "FA21-BCS-001"
        }
        mock_db.get_student_attendance_summary.return_value = {
            "name": "Alice", "registration_number": "FA21-BCS-001",
            "semester": 3, "section": "A",
            "percentage": 80.0, "days_present": 24, "days_total": 30,
            "records": []
        }
        resp = client.post(
            "/student",
            data={"registration_number": "FA21-BCS-001"},
            follow_redirects=True,
        )
    assert resp.status_code == 200


def test_api_students_list(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_all_students.return_value = [
            {"_id": "id1", "name": "Alice", "registration_number": "FA21-BCS-001",
             "semester": 3, "section": "A"}
        ]
        resp = client.get("/api/students")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) >= 1


def test_api_students_get_single(client):
    with patch("app_web.routes.database") as mock_db:
        import bson

        mock_db.get_student_by_reg_no.return_value = {
            "_id": bson.ObjectId(), "name": "Alice",
            "registration_number": "FA21-BCS-001"
        }
        resp = client.get("/api/students/FA21-BCS-001")
    assert resp.status_code == 200


def test_api_students_delete(client):
    with patch("app_web.routes.database") as mock_db, \
         patch("app_web.routes.encoding_cache") as mock_cache:
        mock_db.delete_student.return_value = True
        mock_cache.refresh = MagicMock()
        resp = client.delete("/api/students/FA21-BCS-001")
    assert resp.status_code == 200


def test_api_attendance_bulk(client):
    with patch("app_web.routes.database") as mock_db:
        import bson

        mock_db.get_student_by_reg_no.return_value = {"_id": bson.ObjectId()}
        mock_db.bulk_upsert_attendance.return_value = 2
        resp = client.post(
            "/api/attendance/bulk",
            data=json.dumps({"student_ids": ["FA21-BCS-001", "FA21-BCS-002"], "status": "Present"}),
            content_type="application/json",
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "updated" in data


def test_report_xlsx(client):
    with patch("app_web.routes.database") as mock_db:
        mock_db.get_attendance_csv.return_value = pd.DataFrame({
            "Name": ["Alice"], "Registration Number": ["FA21-BCS-001"],
            "Section": ["A"], "Semester": [3], "Date": ["2026-02-26"],
            "Time": ["09:00:00"], "Status": ["Present"], "Confidence": [0.92],
        })
        resp = client.get("/report/xlsx?date=2026-02-26")
    assert resp.status_code == 200
    assert "spreadsheetml" in resp.content_type or "xlsx" in resp.content_type


def test_api_report_csv_async_date_mode(client):
    fake_task = MagicMock(id="task-1")
    with patch("celery_app.generate_csv_task") as mock_task:
        mock_task.delay.return_value = fake_task
        resp = client.post(
            "/api/report/csv/async",
            data=json.dumps({"date": "2026-02-26"}),
            content_type="application/json",
        )
    assert resp.status_code == 202
    mock_task.delay.assert_called_once_with("date", date_str="2026-02-26")


def test_api_report_csv_async_student_mode(client):
    fake_task = MagicMock(id="task-2")
    with patch("celery_app.generate_csv_task") as mock_task:
        mock_task.delay.return_value = fake_task
        resp = client.post(
            "/api/report/csv/async",
            data=json.dumps({"reg_no": "FA21-BCS-001"}),
            content_type="application/json",
        )
    assert resp.status_code == 202
    mock_task.delay.assert_called_once_with("student", reg_no="FA21-BCS-001")
