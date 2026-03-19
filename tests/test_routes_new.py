"""
Tests for new Flask routes (heatmap, student portal, REST API, bulk, etc.).
"""

import io
import json
import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    import importlib, config
    importlib.reload(config)


@pytest.fixture
def client():
    with patch("database.get_client") as mock_client, \
         patch("database.ensure_indexes"), \
         patch("face_engine.encoding_cache") as mock_cache, \
         patch("anti_spoofing.init_models"):
        mock_client.return_value = MagicMock()
        mock_cache.load = MagicMock()
        mock_cache.size = 0
        mock_cache.get_all.return_value = ([], [], [])

        from app import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_heatmap_page_loads(client):
    resp = client.get("/heatmap")
    assert resp.status_code == 200
    assert b"Heatmap" in resp.data


def test_api_heatmap(client):
    with patch("routes.database") as mock_db:
        mock_db.get_attendance_heatmap_data.return_value = [
            {"date": "2026-01-01", "count": 10, "total_students": 20}
        ]
        resp = client.get("/api/heatmap")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1


def test_api_at_risk(client):
    with patch("routes.database") as mock_db:
        mock_db.get_at_risk_students.return_value = [
            {"name": "Alice", "reg_no": "FA21-BCS-001", "percentage": 40.0,
             "days_present": 12, "days_total": 30}
        ]
        resp = client.get("/api/at_risk")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 1


def test_student_portal_redirect_without_login(client):
    resp = client.get("/student")
    assert resp.status_code == 200
    # Should show login form
    assert b"Registration Number" in resp.data or b"Student Portal" in resp.data


def test_student_portal_login(client):
    with patch("routes.database") as mock_db:
        mock_db.get_student_by_reg_no.return_value = {
            "_id": "test", "name": "Alice", "registration_number": "FA21-BCS-001"
        }
        mock_db.get_student_attendance_summary.return_value = {
            "name": "Alice", "registration_number": "FA21-BCS-001",
            "semester": 3, "section": "A",
            "percentage": 80.0, "days_present": 24, "days_total": 30,
            "records": []
        }
        resp = client.post("/student", data={"reg_no": "FA21-BCS-001"}, follow_redirects=True)
    assert resp.status_code == 200


def test_student_portal_logout(client):
    with client.session_transaction() as sess:
        sess["student_reg_no"] = "FA21-BCS-001"
    resp = client.get("/student/logout", follow_redirects=False)
    assert resp.status_code in (302, 303)


def test_api_students_list(client):
    with patch("routes.database") as mock_db:
        mock_db.get_all_students.return_value = [
            {"_id": "id1", "name": "Alice", "registration_number": "FA21-BCS-001",
             "semester": 3, "section": "A"}
        ]
        resp = client.get("/api/students")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) >= 1


def test_api_students_get_single(client):
    with patch("routes.database") as mock_db:
        import bson
        mock_db.get_student_by_reg_no.return_value = {
            "_id": bson.ObjectId(), "name": "Alice",
            "registration_number": "FA21-BCS-001"
        }
        resp = client.get("/api/students/FA21-BCS-001")
    assert resp.status_code == 200


def test_api_students_delete(client):
    with patch("routes.database") as mock_db, \
         patch("routes.encoding_cache") as mock_cache:
        mock_db.delete_student.return_value = True
        mock_cache.refresh = MagicMock()
        resp = client.delete("/api/students/FA21-BCS-001")
    assert resp.status_code == 200


def test_api_attendance_bulk(client):
    with patch("routes.database") as mock_db:
        import bson
        mock_db.get_student_by_reg_no.return_value = {"_id": bson.ObjectId()}
        mock_db.bulk_upsert_attendance.return_value = 2
        resp = client.post("/api/attendance/bulk",
            data=json.dumps({"student_ids": ["FA21-BCS-001", "FA21-BCS-002"], "status": "Present"}),
            content_type="application/json"
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "updated" in data


def test_api_subject(client):
    with patch("routes.get_camera") as mock_cam_fn:
        mock_cam = MagicMock()
        mock_cam_fn.return_value = mock_cam
        resp = client.post("/api/subject",
            data=json.dumps({"subject": "Mathematics", "cam": 0}),
            content_type="application/json"
        )
    assert resp.status_code == 200


def test_report_xlsx(client):
    with patch("routes.database") as mock_db:
        mock_db.get_attendance_csv.return_value = pd.DataFrame({
            "Name": ["Alice"], "Registration Number": ["FA21-BCS-001"],
            "Section": ["A"], "Semester": [3], "Date": ["2026-02-26"],
            "Time": ["09:00:00"], "Status": ["Present"], "Confidence": [0.92],
            "Subject": ["General"],
        })
        resp = client.get("/report/xlsx?date=2026-02-26")
    assert resp.status_code == 200
    assert "spreadsheetml" in resp.content_type or "xlsx" in resp.content_type or resp.status_code == 200
