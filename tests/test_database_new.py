"""
Tests for new database functions (append_encoding, bulk_upsert, at_risk, etc.).
"""

import os
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone

import bson
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib, config
    importlib.reload(config)


@pytest.fixture
def mock_db():
    db = MagicMock()
    with patch("database.get_db", return_value=db):
        yield db


def test_append_student_encoding(mock_db):
    import database
    sid = bson.ObjectId()
    enc = np.random.rand(128).astype(np.float64)
    mock_db.students.update_one.return_value = MagicMock(modified_count=1)

    result = database.append_student_encoding(sid, enc)
    assert result is True
    mock_db.students.update_one.assert_called_once()


def test_mark_attendance(mock_db):
    import database
    sid = bson.ObjectId()
    mock_db.attendance.insert_one.return_value = MagicMock()

    result = database.mark_attendance(sid, 0.92)
    assert result is True
    doc = mock_db.attendance.insert_one.call_args[0][0]
    assert doc["confidence_score"] == 0.92


def test_bulk_upsert_attendance(mock_db):
    import database
    sid1 = bson.ObjectId()
    sid2 = bson.ObjectId()

    entries = [
        {"student_id": sid1, "status": "Present"},
        {"student_id": sid2, "status": "Absent"},
    ]
    mock_db.attendance.bulk_write.return_value = MagicMock(
        upserted_count=1, modified_count=1
    )

    result = database.bulk_upsert_attendance(entries)
    assert result == 2


def test_get_student_by_reg_no(mock_db):
    import database
    sid = bson.ObjectId()
    mock_db.students.find_one.return_value = {
        "_id": sid, "name": "Alice", "registration_number": "FA21-BCS-001"
    }

    result = database.get_student_by_reg_no("FA21-BCS-001")
    assert result["name"] == "Alice"


def test_delete_student(mock_db):
    import database
    sid = bson.ObjectId()
    mock_db.students.find_one.return_value = {"_id": sid}
    mock_db.students.delete_one.return_value = MagicMock(deleted_count=1)

    result = database.delete_student("FA21-BCS-001")
    assert result is True
    mock_db.attendance.delete_many.assert_called_once()


def test_delete_student_not_found(mock_db):
    import database
    mock_db.students.find_one.return_value = None

    result = database.delete_student("NONEXISTENT")
    assert result is False


def test_get_at_risk_students(mock_db):
    import database
    sid = bson.ObjectId()
    mock_db.students.find.return_value = [
        {"_id": sid, "name": "Alice", "registration_number": "FA21-BCS-001"}
    ]
    mock_db.attendance.distinct.return_value = [
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05",
        "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10",
    ]
    mock_db.attendance.count_documents.return_value = 3  # 3/10 = 30%

    result = database.get_at_risk_students(days=30, threshold=75)
    assert len(result) == 1
    assert result[0]["percentage"] < 75


def test_get_attendance_heatmap_data(mock_db):
    import database
    mock_db.students.count_documents.return_value = 20
    mock_db.attendance.aggregate.return_value = [
        {"date": "2026-01-01", "count": 15},
    ]

    result = database.get_attendance_heatmap_data(days=90)
    assert len(result) == 1
    assert result[0]["total_students"] == 20


def test_get_student_attendance_summary(mock_db):
    import database
    sid = bson.ObjectId()
    mock_db.students.find_one.return_value = {
        "_id": sid, "name": "Alice", "registration_number": "FA21-BCS-001",
        "semester": 3, "section": "A"
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
    import database
    mock_db.students.find_one.return_value = None

    result = database.get_student_attendance_summary("NONEXISTENT")
    assert result is None
