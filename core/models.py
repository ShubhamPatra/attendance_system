"""DAO layer aligned with the migrated package structure.

These DAOs are thin adapters over the stable database functions. This keeps
behavior unchanged while giving the app a consistent object-based data layer.
"""

from __future__ import annotations

from typing import Any

import core.database as database


class StudentDAO:
	"""Data access for student records."""

	def __init__(self, db_module=database):
		self.db = db_module

	def get_by_registration_number(self, reg_no: str) -> dict | None:
		return self.db.get_student_by_reg_no(reg_no)

	def get_by_id(self, student_id: Any, include_sensitive: bool = False) -> dict | None:
		return self.db.get_student_by_id(student_id, include_sensitive=include_sensitive)


class AttendanceSessionDAO:
	"""Data access for attendance sessions."""

	def __init__(self, db_module=database):
		self.db = db_module

	def create(self, course_id: str, camera_id: str):
		return self.db.create_attendance_session(course_id=course_id, camera_id=camera_id)

	def end(self, session_id: Any) -> bool:
		return self.db.end_attendance_session(session_id)

	def get_active(self, camera_id: str) -> dict | None:
		return self.db.get_active_attendance_session(camera_id)

	def get_by_id(self, session_id: Any) -> dict | None:
		return self.db.get_attendance_session_by_id(session_id)

	def auto_close_idle(self, idle_seconds: int | None = None) -> int:
		if idle_seconds is None:
			return self.db.auto_close_idle_attendance_sessions()
		return self.db.auto_close_idle_attendance_sessions(idle_seconds=idle_seconds)


class AttendanceDAO:
	"""Data access for attendance marks and queries."""

	def __init__(self, db_module=database):
		self.db = db_module

	def bulk_upsert(self, entries: list[dict], session_id: Any | None = None) -> int:
		return self.db.bulk_upsert_attendance(entries, session_id=session_id)

	def list(self, date: str | None = None) -> list[dict]:
		return self.db.get_attendance(date)

	def list_by_student(self, reg_no: str) -> list[dict]:
		return self.db.get_attendance_by_student(reg_no)

	def list_by_range(self, start_date: str, end_date: str) -> list[dict]:
		return self.db.get_attendance_by_date_range(start_date, end_date)


__all__ = [
	"StudentDAO",
	"AttendanceSessionDAO",
	"AttendanceDAO",
]
