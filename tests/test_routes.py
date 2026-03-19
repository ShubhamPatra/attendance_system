"""
Tests for Flask routes.
Uses the Flask test client — no live server or webcam needed.
"""

import io
import os
import sys
import base64
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
    monkeypatch.setenv("SUBJECTS", "General,Mathematics")
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
    import importlib, config
    importlib.reload(config)


@pytest.fixture
def client():
    """Create a Flask test client with mocked DB calls."""
    with patch("database.get_client") as mock_client, \
         patch("database.ensure_indexes"), \
         patch("face_engine.encoding_cache") as mock_cache, \
         patch("anti_spoofing.init_models"):

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

    with patch("routes.generate_encoding", return_value=fake_encoding), \
         patch("routes.cv2.imread", return_value=fake_image), \
         patch("routes.check_image_quality", return_value=(True, "")), \
         patch("routes.database") as mock_db, \
         patch("routes.encoding_cache") as mock_cache:

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
    with patch("routes.database") as mock_db:
        mock_db.student_count.return_value = 5
        mock_db.today_attendance_count.return_value = 3
        mock_db.get_attendance.return_value = []
        resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Dashboard" in resp.data


# ── Report page ───────────────────────────────────────────────────────────

def test_report_loads(client):
    with patch("routes.database") as mock_db:
        mock_db.get_attendance.return_value = []
        resp = client.get("/report?date=2026-02-26")
    assert resp.status_code == 200
    assert b"Report" in resp.data


# ── CSV export ────────────────────────────────────────────────────────────

def test_report_csv(client):
    with patch("routes.database") as mock_db:
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

    with patch("routes.generate_encoding", return_value=fake_encoding), \
         patch("routes.cv2.imread", return_value=fake_image), \
         patch("routes.check_image_quality", return_value=(True, "")), \
         patch("routes.database") as mock_db, \
         patch("routes.encoding_cache"):
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

    with patch("routes.generate_encoding", side_effect=ValueError("No face detected")), \
         patch("routes.cv2.imread", return_value=fake_image), \
         patch("routes.check_image_quality", return_value=(True, "")):
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
    with patch("routes.database") as mock_db:
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
    with patch("camera.get_camera") as mock_cam:
        mock_cam.return_value.get_log_buffer.return_value = []
        resp = client.get("/api/logs")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_api_attendance_activity(client):
    with patch("routes.database") as mock_db:
        mock_db.get_attendance_by_hour.return_value = [
            {"hour": 9, "count": 4}
        ]
        resp = client.get("/api/attendance_activity?date=2026-02-26")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["date"] == "2026-02-26"
    assert len(data["hours"]) == 1


def test_api_registration_numbers(client):
    with patch("routes.database") as mock_db:
        mock_db.get_all_registration_numbers.return_value = [
            "FA21-BCS-001", "FA21-BCS-002"
        ]
        resp = client.get("/api/registration_numbers")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 2


# ── Extended CSV export ───────────────────────────────────────────────────

def test_report_csv_by_student(client):
    with patch("routes.database") as mock_db:
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
    with patch("routes.database") as mock_db:
        mock_db.get_student_by_reg_no.return_value = None
        resp = client.get("/report/csv?reg_no=FA21-BCS-404")
    assert resp.status_code == 404


def test_report_csv_full(client):
    with patch("routes.database") as mock_db:
        mock_db.get_attendance_csv_full.return_value = pd.DataFrame(
            columns=["Name", "Registration Number", "Section",
                     "Semester", "Date", "Time", "Status", "Confidence"]
        )
        resp = client.get("/report/csv?full=1")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/csv")


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
         patch("routes.config.UPLOAD_DIR", tmpdir), \
         patch("routes.cv2.imdecode", return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
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
