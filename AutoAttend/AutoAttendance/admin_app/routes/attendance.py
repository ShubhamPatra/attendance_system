from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId
from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from admin_app.routes.auth import role_required
from core.models import AttendanceDAO, LogDAO, SessionDAO


attendance_bp = Blueprint("admin_attendance", __name__)


def _db():
	from admin_app import get_course_dao

	return get_course_dao().db


def _attendance_dao() -> AttendanceDAO:
	return AttendanceDAO(_db())


def _session_dao() -> SessionDAO:
	return SessionDAO(_db())


def _log_dao() -> LogDAO:
	return LogDAO(_db())


@attendance_bp.get("/attendance")
@login_required
@role_required("super_admin", "admin")
def attendance_overview():
	today = datetime.now(timezone.utc).date().isoformat()
	rows = []
	for course in _db().courses.find({"is_active": True}):
		stats = _attendance_dao().get_stats(str(course["_id"]))
		rows.append({"course": course, "stats": stats})
	return render_template("attendance/attendance_overview.html", today=today, rows=rows)


@attendance_bp.get("/attendance/session/<session_id>")
@login_required
@role_required("super_admin", "admin")
def session_live(session_id: str):
	session = _db().attendance_sessions.find_one({"_id": ObjectId(session_id)})
	if not session:
		return jsonify({"error": "session not found"}), 404
	records = list(_db().attendance_records.find({"session_id": session_id}).sort("created_at", -1))
	return render_template("attendance/session_live.html", session=session, records=records)


@attendance_bp.route("/attendance/manual", methods=["GET", "POST"])
@login_required
@role_required("super_admin", "admin")
def manual_mark():
	if request.method == "GET":
		return render_template("attendance/attendance_overview.html", today=datetime.now(timezone.utc).date().isoformat(), rows=[])

	student_id = request.form.get("student_id", "")
	course_id = request.form.get("course_id", "")
	status = request.form.get("status", "present")
	result = _attendance_dao().record_attendance(
		student_id=student_id,
		course_id=course_id,
		status=status,
		confidence=1.0,
		liveness_score=1.0,
		verification_method="manual",
		ip_address=request.remote_addr,
		user_agent=request.user_agent.string if request.user_agent else None,
		session_id=request.form.get("session_id"),
	)
	_log_dao().log_event("manual_attendance", current_user.id, "admin", {"student_id": student_id, "course_id": course_id, "status": status, "result": result}, request.remote_addr)
	if not result.get("inserted"):
		return jsonify(result), 400
	flash("Attendance marked", "success")
	return redirect(url_for("admin_attendance.attendance_overview"))


@attendance_bp.post("/attendance/<record_id>/override")
@login_required
@role_required("super_admin", "admin")
def override(record_id: str):
	status = request.form.get("status", "present")
	reason = request.form.get("reason", "")
	updated = _attendance_dao().override_status(record_id, status)
	_log_dao().log_event("attendance_override", current_user.id, "admin", {"record_id": record_id, "status": status, "reason": reason, "updated": updated}, request.remote_addr)
	if not updated:
		return jsonify({"error": "record not found"}), 404
	return redirect(url_for("admin_attendance.attendance_overview"))
