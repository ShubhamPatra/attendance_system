from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, render_template
from flask_login import current_user, login_required

from admin_app.routes.auth import role_required
from core.extensions import socketio
from core.models import AttendanceDAO


dashboard_bp = Blueprint("admin_dashboard", __name__)


def _db():
	from admin_app import get_course_dao

	return get_course_dao().db


def get_dashboard_stats() -> dict:
	db = _db()
	attendance_dao = AttendanceDAO(db)
	today = datetime.now(timezone.utc).date().isoformat()
	active_students = db.students.count_documents({"is_active": True})
	active_sessions = db.attendance_sessions.count_documents({"status": "open"})
	today_records = list(db.attendance_records.find({"date": today}))
	day_total = len(today_records)
	day_present = sum(1 for record in today_records if record.get("status") in {"present", "late"})
	total_rate = round((day_present / active_students) * 100, 2) if active_students else 0.0
	recent = []
	for record in db.attendance_records.find().sort("created_at", -1).limit(20):
		student = db.students.find_one({"_id": record.get("student_id")})
		course = db.courses.find_one({"_id": record.get("course_id")})
		recent.append(
			{
				"id": str(record.get("_id")),
				"date": record.get("date"),
				"status": record.get("status", "unknown"),
				"confidence": float(record.get("confidence_score", 0.0)),
				"student_name": student.get("name") if student else "Unknown student",
				"student_id": student.get("student_id") if student else None,
				"course_code": course.get("course_code") if course else None,
				"course_name": course.get("course_name") if course else None,
				"time": record.get("check_in_time").astimezone(timezone.utc).strftime("%H:%M") if record.get("check_in_time") else "--:--",
			}
		)
	trend_labels: list[str] = []
	trend_values: list[int] = []
	for offset in range(6, -1, -1):
		day = (datetime.now(timezone.utc) - timedelta(days=offset)).date().isoformat()
		trend_labels.append(day[-5:])
		trend_values.append(db.attendance_records.count_documents({"date": day, "status": {"$in": ["present", "late"]}}))
	course_stats = []
	for course in db.courses.find({"is_active": True}):
		stats = attendance_dao.get_stats(str(course["_id"]))
		total = stats["present"] + stats["late"] + stats["absent"]
		rate = round(((stats["present"] + stats["late"]) / total) * 100, 2) if total else 0.0
		course_stats.append({"course_code": course["course_code"], "rate": rate, "stats": stats})
	dept_counts = Counter(student.get("department") for student in db.students.find({"is_active": True}))
	return {
		"today": today,
		"present_today": day_present,
		"attendance_rate": total_rate,
		"active_sessions": active_sessions,
		"total_students": active_students,
		"today_total": day_total,
		"course_stats": course_stats,
		"recent_checkins": recent,
		"department_breakdown": dict(dept_counts),
		"attendance_trend": {"labels": trend_labels, "values": trend_values},
	}


def emit_attendance_update(payload: dict) -> None:
	socketio.emit("attendance_update", payload)


@dashboard_bp.get("/dashboard")
@login_required
@role_required("super_admin", "admin", "viewer")
def dashboard():
	stats = get_dashboard_stats()
	return render_template("dashboard/dashboard.html", stats=stats)


@socketio.on("connect")
def handle_connect():
	if not current_user.is_authenticated:
		return False
	try:
		stats = get_dashboard_stats()
	except RuntimeError:
		return True
	socketio.emit("stats_refresh", stats)
	return True


@socketio.on("stats_refresh")
def handle_stats_refresh():
	return get_dashboard_stats()
