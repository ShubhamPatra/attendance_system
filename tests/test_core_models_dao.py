"""DAO-layer tests for migrated core.models adapters."""

from unittest.mock import MagicMock

from core.models import AttendanceDAO, AttendanceSessionDAO, StudentDAO


def test_student_dao_uses_injected_backend():
    backend = MagicMock()
    backend.get_student_by_reg_no.return_value = {"registration_number": "FA21-BCS-001"}

    dao = StudentDAO(backend)
    out = dao.get_by_registration_number("FA21-BCS-001")

    assert out["registration_number"] == "FA21-BCS-001"
    backend.get_student_by_reg_no.assert_called_once_with("FA21-BCS-001")


def test_attendance_session_dao_uses_injected_backend():
    backend = MagicMock()
    backend.create_attendance_session.return_value = "sid-1"
    backend.end_attendance_session.return_value = True

    dao = AttendanceSessionDAO(backend)

    sid = dao.create("CS101", "cam-1")
    ended = dao.end("sid-1")

    assert sid == "sid-1"
    assert ended is True
    backend.create_attendance_session.assert_called_once_with(course_id="CS101", camera_id="cam-1")
    backend.end_attendance_session.assert_called_once_with("sid-1")


def test_attendance_dao_bulk_upsert_uses_backend():
    backend = MagicMock()
    backend.bulk_upsert_attendance.return_value = 2

    dao = AttendanceDAO(backend)
    count = dao.bulk_upsert([
        {"student_id": "a", "status": "Present", "confidence_score": 0.9},
        {"student_id": "b", "status": "Present", "confidence_score": 0.8},
    ])

    assert count == 2
    backend.bulk_upsert_attendance.assert_called_once()
