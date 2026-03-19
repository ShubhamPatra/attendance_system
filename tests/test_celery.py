"""
Tests for Celery tasks.
Uses mocks - no live Redis, MongoDB, or network calls.
"""

import base64
import os
import sys
import tempfile
import types
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    """Ensure config module has dummy env vars for all tests."""
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("SENDGRID_API_KEY", "")
    monkeypatch.setenv("NOTIFY_EMAIL", "")
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


# ── 1. Celery app instance ────────────────────────────────────────────────

def test_celery_app_instance_created():
    """The module-level celery_app should be a Celery instance."""
    from celery import Celery
    import celery_app

    assert isinstance(celery_app.celery_app, Celery)


# ── 2. generate_csv_task – date query ────────────────────────────────────

def test_generate_csv_task_date():
    """generate_csv_task with query_type='date' writes a CSV and returns its path."""
    import celery_app

    fake_df = pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Registration Number": ["FA21-BCS-001", "FA21-BCS-002"],
        "Section": ["A", "A"],
        "Semester": [3, 3],
        "Date": ["2026-01-01", "2026-01-01"],
        "Time": ["09:00:00", "09:05:00"],
        "Status": ["Present", "Present"],
        "Confidence": [0.92, 0.88],
    })

    with patch("database.get_attendance_csv", return_value=fake_df):
        result = celery_app.generate_csv_task.run(
            "date", date_str="2026-01-01"
        )

    assert isinstance(result, str)
    assert result.endswith(".csv")
    assert os.path.exists(result)

    # Verify file contents
    loaded = pd.read_csv(result)
    assert len(loaded) == 2
    assert loaded.iloc[0]["Name"] == "Alice"

    # Clean up
    os.unlink(result)


# ── 3. generate_csv_task – full query ────────────────────────────────────

def test_generate_csv_task_full():
    """generate_csv_task with query_type='full' writes a CSV and returns its path."""
    import celery_app

    fake_df = pd.DataFrame({
        "Name": ["Carol"],
        "Registration Number": ["FA21-BCS-003"],
        "Section": ["B"],
        "Semester": [5],
        "Date": ["2026-01-15"],
        "Time": ["10:30:00"],
        "Status": ["Present"],
        "Confidence": [0.95],
    })

    with patch("database.get_attendance_csv_full", return_value=fake_df):
        result = celery_app.generate_csv_task.run("full")

    assert isinstance(result, str)
    assert result.endswith(".csv")
    assert os.path.exists(result)

    loaded = pd.read_csv(result)
    assert len(loaded) == 1
    assert loaded.iloc[0]["Name"] == "Carol"

    # Clean up
    os.unlink(result)


def test_generate_csv_task_student():
    import celery_app

    fake_df = pd.DataFrame({"Name": ["Alice"]})
    with patch("database.get_attendance_csv_by_student", return_value=fake_df):
        result = celery_app.generate_csv_task.run("student", reg_no="FA21-BCS-001")

    assert os.path.exists(result)
    os.unlink(result)


def test_generate_csv_task_range():
    import celery_app

    fake_df = pd.DataFrame({"Name": ["Alice"]})
    with patch("database.get_attendance_csv_by_date_range", return_value=fake_df):
        result = celery_app.generate_csv_task.run(
            "range",
            start_date="2026-01-01",
            end_date="2026-01-10",
        )

    assert os.path.exists(result)
    os.unlink(result)


def test_generate_csv_task_unknown_query_type():
    import celery_app

    with pytest.raises(ValueError, match="Unknown query_type"):
        celery_app.generate_csv_task.run("bad_mode")


# ── 4. compute_encodings_task – success ──────────────────────────────────

def test_compute_encodings_task_success():
    """compute_encodings_task returns base64-encoded encodings on success."""
    import celery_app

    fake_encoding = np.random.rand(128).astype(np.float64)

    with patch("face_engine.generate_encoding", return_value=fake_encoding):
        result = celery_app.compute_encodings_task.run(
            image_paths=["/fake/img1.jpg", "/fake/img2.jpg"]
        )

    assert "encodings" in result
    assert "errors" in result
    assert len(result["encodings"]) == 2
    assert len(result["errors"]) == 0

    # Verify the base64 string decodes back to the correct numpy array
    decoded_bytes = base64.b64decode(result["encodings"][0])
    decoded_array = np.frombuffer(decoded_bytes, dtype=np.float64)
    np.testing.assert_array_almost_equal(decoded_array, fake_encoding)


# ── 5. compute_encodings_task – failure ──────────────────────────────────

def test_compute_encodings_task_failure():
    """compute_encodings_task records errors when generate_encoding raises."""
    import celery_app

    with patch(
        "face_engine.generate_encoding",
        side_effect=ValueError("No face detected in image"),
    ):
        result = celery_app.compute_encodings_task.run(
            image_paths=["/fake/bad1.jpg", "/fake/bad2.jpg"]
        )

    assert "encodings" in result
    assert "errors" in result
    assert len(result["encodings"]) == 0
    assert len(result["errors"]) == 2
    assert result["errors"][0]["path"] == "/fake/bad1.jpg"
    assert "No face detected" in result["errors"][0]["error"]
    assert result["errors"][1]["path"] == "/fake/bad2.jpg"


# ── 6. send_absence_notifications – no API key ──────────────────────────

def test_send_absence_notifications_no_api_key():
    """send_absence_notifications exits gracefully when SendGrid is not configured."""
    import celery_app
    import bson

    mock_db = MagicMock()
    mock_db.students.find.return_value = [
        {
            "_id": bson.ObjectId(),
            "name": "Alice",
            "registration_number": "FA21-BCS-001",
        },
    ]
    # Student has 0 attendance records -> below threshold
    mock_db.attendance.count_documents.return_value = 0

    with patch("database.get_db", return_value=mock_db), \
         patch("config.SENDGRID_API_KEY", ""), \
         patch("config.NOTIFY_EMAIL", ""), \
         patch("config.ABSENCE_THRESHOLD", 75):
        result = celery_app.send_absence_notifications.run()

    # Should return dict indicating email was NOT sent
    assert isinstance(result, dict)
    assert result["flagged"] == 1
    assert result["email_sent"] is False


# ── 7. backup_mongodb ────────────────────────────────────────────────────

def test_backup_mongodb():
    """backup_mongodb creates a .tar.gz archive in the configured backup dir."""
    import celery_app

    mock_db = MagicMock()
    mock_db.students.find.return_value = []
    mock_db.attendance.find.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("database.get_db", return_value=mock_db), \
             patch("config.BACKUP_DIR", tmpdir), \
             patch("config.BACKUP_RETENTION_DAYS", 30):
            result = celery_app.backup_mongodb.run()

        assert isinstance(result, dict)
        assert "archive" in result
        assert result["students"] == 0
        assert result["attendance"] == 0

        # Verify the archive file was actually created
        archive_path = result["archive"]
        assert os.path.exists(archive_path)
        assert archive_path.endswith(".tar.gz")
