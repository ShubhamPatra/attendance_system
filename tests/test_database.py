"""
Tests for the database layer.
Uses unittest.mock to avoid requiring a live MongoDB Atlas connection.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import bson
import numpy as np
import pytest


# We need to mock config before importing database, because config.py
# raises EnvironmentError if MONGO_URI is not set.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    """Ensure config module has a dummy MONGO_URI for all tests."""
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    # Reload config so it picks up the env var
    import importlib
    import app_core.config as config
    importlib.reload(config)


@pytest.fixture
def mock_db():
    """Return a mock database object and patch database.get_db."""
    db = MagicMock()
    with patch("app_core.database.get_db", return_value=db):
        yield db


# ── Insert student ────────────────────────────────────────────────────────

def test_insert_student_success(mock_db):
    import app_core.database as database

    fake_id = bson.ObjectId()
    mock_db.students.insert_one.return_value = MagicMock(inserted_id=fake_id)

    encodings = [np.random.rand(128).astype(np.float64) for _ in range(3)]
    result = database.insert_student("Alice", 3, "FA21-BCS-001", "A", encodings)

    assert result == fake_id
    mock_db.students.insert_one.assert_called_once()
    doc = mock_db.students.insert_one.call_args[0][0]
    assert doc["name"] == "Alice"
    assert doc["semester"] == 3
    assert doc["registration_number"] == "FA21-BCS-001"
    assert doc["section"] == "A"
    assert isinstance(doc["encodings"], list)
    assert len(doc["encodings"]) == 3
    for b in doc["encodings"]:
        assert isinstance(b, bson.Binary)


# ── Reject duplicate registration_number ──────────────────────────────────

def test_insert_student_duplicate_raises(mock_db):
    import app_core.database as database
    from pymongo.errors import DuplicateKeyError

    mock_db.students.insert_one.side_effect = DuplicateKeyError("dup")

    encodings = [np.random.rand(128).astype(np.float64)]
    with pytest.raises(ValueError, match="already exists"):
        database.insert_student("Bob", 2, "FA21-BCS-001", "B", encodings)


# ── Mark attendance ───────────────────────────────────────────────────────

def test_mark_attendance_success(mock_db):
    import app_core.database as database

    mock_db.attendance.insert_one.return_value = MagicMock()

    sid = bson.ObjectId()
    result = database.mark_attendance(sid, 0.87)

    assert result is True
    mock_db.attendance.insert_one.assert_called_once()
    doc = mock_db.attendance.insert_one.call_args[0][0]
    assert doc["student_id"] == sid
    assert doc["status"] == "Present"
    assert doc["confidence_score"] == 0.87


# ── Prevent duplicate attendance per day ──────────────────────────────────

def test_mark_attendance_duplicate_returns_false(mock_db):
    import app_core.database as database
    from pymongo.errors import DuplicateKeyError

    mock_db.attendance.insert_one.side_effect = DuplicateKeyError("dup")

    sid = bson.ObjectId()
    result = database.mark_attendance(sid, 0.87)

    assert result is False


# ── get_all_students ──────────────────────────────────────────────────────

def test_get_all_students(mock_db):
    import app_core.database as database

    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [
        {"_id": bson.ObjectId(), "name": "Alice", "semester": 3,
         "registration_number": "FA21-BCS-001", "section": "A"},
    ]
    mock_db.students.find.return_value = mock_cursor

    result = database.get_all_students()
    assert len(result) == 1
    assert result[0]["name"] == "Alice"
    # Verify both encoding fields are excluded from projection
    call_args = mock_db.students.find.call_args
    assert call_args[0][1] == {"face_encoding": 0, "encodings": 0}


def test_get_student_by_id_excludes_encodings(mock_db):
    import app_core.database as database
    sid = bson.ObjectId()

    database.get_student_by_id(sid)
    args, _ = mock_db.students.find_one.call_args
    assert args[1] == {"face_encoding": 0, "encodings": 0, "created_at": 0}


def test_ensure_indexes(mock_db):
    import app_core.database as database

    database.ensure_indexes()

    mock_db.attendance.create_index.assert_any_call(
        [("student_id", 1), ("date", 1)],
        unique=True,
        name="uq_student_date",
    )
    mock_db.attendance.create_index.assert_any_call(
        [("date", 1)],
        name="idx_date",
    )


# ── get_student_encodings ────────────────────────────────────────────────

def test_get_student_encodings_new_format(mock_db):
    """New multi-encoding format: 'encodings' field is a list of Binary."""
    import app_core.database as database

    enc1 = np.random.rand(128).astype(np.float64)
    enc2 = np.random.rand(128).astype(np.float64)
    sid = bson.ObjectId()
    mock_db.students.find.return_value = [
        {"_id": sid, "name": "Bob", "encodings": [
            bson.Binary(enc1.tobytes()),
            bson.Binary(enc2.tobytes()),
        ]},
    ]

    result = database.get_student_encodings()
    assert len(result) == 1
    assert result[0][0] == sid
    assert result[0][1] == "Bob"
    assert len(result[0][2]) == 2
    np.testing.assert_array_almost_equal(result[0][2][0], enc1)
    np.testing.assert_array_almost_equal(result[0][2][1], enc2)


def test_get_student_encodings_legacy_format(mock_db):
    """Backward-compat: 'face_encoding' single Binary field."""
    import app_core.database as database

    enc = np.random.rand(128).astype(np.float64)
    sid = bson.ObjectId()
    mock_db.students.find.return_value = [
        {"_id": sid, "name": "Bob", "face_encoding": bson.Binary(enc.tobytes())},
    ]

    result = database.get_student_encodings()
    assert len(result) == 1
    assert result[0][0] == sid
    assert result[0][1] == "Bob"
    assert len(result[0][2]) == 1
    np.testing.assert_array_almost_equal(result[0][2][0], enc)


# ── student_count ─────────────────────────────────────────────────────────

def test_student_count(mock_db):
    import app_core.database as database

    mock_db.students.count_documents.return_value = 7
    assert database.student_count() == 7


# ── today_attendance_count ────────────────────────────────────────────────

def test_today_attendance_count(mock_db):
    import app_core.database as database

    mock_db.attendance.count_documents.return_value = 3
    assert database.today_attendance_count() == 3


# ── get_attendance ────────────────────────────────────────────────────────

def test_get_attendance(mock_db):
    import app_core.database as database

    mock_db.attendance.aggregate.return_value = [
        {"name": "Alice", "registration_number": "FA21-BCS-001",
         "section": "A", "semester": 3, "date": "2026-02-26",
         "time": "09:00:00", "status": "Present", "confidence_score": 0.92},
    ]

    result = database.get_attendance("2026-02-26")
    assert len(result) == 1
    assert result[0]["name"] == "Alice"


# ── get_attendance_csv ────────────────────────────────────────────────────

def test_get_attendance_csv_empty(mock_db):
    import app_core.database as database

    mock_db.attendance.aggregate.return_value = []

    df = database.get_attendance_csv("2026-02-26")
    assert len(df) == 0
    assert "Name" in df.columns


def test_get_attendance_csv_with_data(mock_db):
    import app_core.database as database

    mock_db.attendance.aggregate.return_value = [
        {"name": "Alice", "registration_number": "FA21-BCS-001",
         "section": "A", "semester": 3, "date": "2026-02-26",
         "time": "09:00:00", "status": "Present", "confidence_score": 0.92},
    ]

    df = database.get_attendance_csv("2026-02-26")
    assert len(df) == 1
    assert df.iloc[0]["Name"] == "Alice"


# ── get_attendance_by_hour ────────────────────────────────────────────────

def test_get_attendance_by_hour(mock_db):
    import app_core.database as database

    mock_db.attendance.aggregate.return_value = [
        {"hour": 8, "count": 3},
        {"hour": 9, "count": 5},
    ]

    result = database.get_attendance_by_hour("2026-02-26")
    assert len(result) == 2
    assert result[0]["hour"] == 8
    assert result[1]["count"] == 5


# ── get_all_registration_numbers ──────────────────────────────────────────

def test_get_all_registration_numbers(mock_db):
    import app_core.database as database

    mock_db.students.find.return_value = [
        {"registration_number": "FA21-BCS-002"},
        {"registration_number": "FA21-BCS-001"},
    ]

    result = database.get_all_registration_numbers()
    assert result == ["FA21-BCS-001", "FA21-BCS-002"]


# ── get_attendance_csv_by_student ─────────────────────────────────────────

def test_get_attendance_csv_by_student(mock_db):
    import app_core.database as database

    sid = bson.ObjectId()
    mock_db.students.find_one.return_value = {"_id": sid}
    mock_db.attendance.aggregate.return_value = [
        {"name": "Alice", "registration_number": "FA21-BCS-001",
         "section": "A", "semester": 3, "date": "2026-02-26",
         "time": "09:00:00", "status": "Present", "confidence_score": 0.92},
    ]

    df = database.get_attendance_csv_by_student("FA21-BCS-001")
    assert len(df) == 1
    assert df.iloc[0]["Name"] == "Alice"


# ── get_attendance_csv_full ───────────────────────────────────────────────

def test_get_attendance_csv_full(mock_db):
    import app_core.database as database

    mock_db.attendance.aggregate.return_value = []
    df = database.get_attendance_csv_full()
    assert len(df) == 0
    assert "Name" in df.columns


def test_append_student_encoding(mock_db):
    import app_core.database as database

    sid = bson.ObjectId()
    enc = np.random.rand(128).astype(np.float64)
    mock_db.students.update_one.return_value = MagicMock(modified_count=1)

    result = database.append_student_encoding(sid, enc)
    assert result is True
    mock_db.students.update_one.assert_called_once()


def test_mark_attendance_with_confidence_field(mock_db):
    import app_core.database as database

    sid = bson.ObjectId()
    mock_db.attendance.insert_one.return_value = MagicMock()

    result = database.mark_attendance(sid, 0.92)
    assert result is True
    doc = mock_db.attendance.insert_one.call_args[0][0]
    assert doc["confidence_score"] == 0.92


def test_bulk_upsert_attendance(mock_db):
    import app_core.database as database

    sid1 = bson.ObjectId()
    sid2 = bson.ObjectId()
    entries = [
        {"student_id": sid1, "status": "Present"},
        {"student_id": sid2, "status": "Absent"},
    ]
    mock_db.attendance.bulk_write.return_value = MagicMock(
        upserted_count=1,
        modified_count=1,
    )

    result = database.bulk_upsert_attendance(entries)
    assert result == 2


def test_get_student_by_reg_no(mock_db):
    import app_core.database as database

    sid = bson.ObjectId()
    mock_db.students.find_one.return_value = {
        "_id": sid,
        "name": "Alice",
        "registration_number": "FA21-BCS-001",
    }

    result = database.get_student_by_reg_no("FA21-BCS-001")
    assert result["name"] == "Alice"


def test_delete_student(mock_db):
    import app_core.database as database

    sid = bson.ObjectId()
    mock_db.students.find_one.return_value = {"_id": sid}
    mock_db.students.delete_one.return_value = MagicMock(deleted_count=1)

    result = database.delete_student("FA21-BCS-001")
    assert result is True
    mock_db.attendance.delete_many.assert_called_once()


def test_delete_student_not_found(mock_db):
    import app_core.database as database

    mock_db.students.find_one.return_value = None
    result = database.delete_student("NONEXISTENT")
    assert result is False


def test_get_at_risk_students(mock_db):
    import app_core.database as database

    today = datetime.now(timezone.utc).date()
    mock_db.attendance.distinct.return_value = [
        (today - timedelta(days=delta)).strftime("%Y-%m-%d")
        for delta in range(10)
    ]
    mock_db.attendance.aggregate.return_value = [
        {
            "name": "Alice",
            "reg_no": "FA21-BCS-001",
            "percentage": 30.0,
            "days_present": 3,
            "days_total": 10,
        }
    ]

    result = database.get_at_risk_students(days=30, threshold=75)
    assert len(result) == 1
    assert result[0]["percentage"] < 75


def test_get_attendance_heatmap_data(mock_db):
    import app_core.database as database

    mock_db.students.count_documents.return_value = 20
    mock_db.attendance.aggregate.return_value = [
        {"date": "2026-01-01", "count": 15},
    ]

    result = database.get_attendance_heatmap_data(days=90)
    assert len(result) == 1
    assert result[0]["total_students"] == 20


def test_get_student_attendance_summary(mock_db):
    import app_core.database as database

    sid = bson.ObjectId()
    mock_db.students.find_one.return_value = {
        "_id": sid,
        "name": "Alice",
        "registration_number": "FA21-BCS-001",
        "semester": 3,
        "section": "A",
    }
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [
        {"date": "2026-01-01", "time": "09:00:00", "status": "Present", "confidence_score": 0.92}
    ]
    mock_db.attendance.find.return_value = mock_cursor

    result = database.get_student_attendance_summary("FA21-BCS-001")
    assert result is not None
    assert result["name"] == "Alice"
    assert result["days_present"] == 1


def test_get_student_attendance_summary_not_found(mock_db):
    import app_core.database as database

    mock_db.students.find_one.return_value = None
    result = database.get_student_attendance_summary("NONEXISTENT")
    assert result is None
