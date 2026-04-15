from __future__ import annotations

import json

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from admin_app.forms import CourseForm
from admin_app.routes.auth import role_required


courses_bp = Blueprint("admin_courses", __name__)


def _course_dao():
	from admin_app import get_course_dao

	return get_course_dao()


@courses_bp.get("/courses")
@login_required
@role_required("super_admin", "admin")
def list_courses():
	department = request.args.get("department")
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 20))
	result = _course_dao().list_courses(department=department, page=page, per_page=per_page)
	return render_template("courses/course_list.html", courses=result)


@courses_bp.get("/courses/<course_id>")
@login_required
@role_required("super_admin", "admin")
def course_detail(course_id: str):
	course = _course_dao().get_by_id(course_id)
	if not course:
		return jsonify({"error": "course not found"}), 404
	students = _course_dao().get_enrolled_students(course_id, page=1, per_page=100)
	return render_template("courses/course_detail.html", course=course, students=students)


@courses_bp.route("/courses/add", methods=["GET", "POST"])
@login_required
@role_required("super_admin", "admin")
def add_course():
	form = CourseForm()
	if form.validate_on_submit():
		schedule = json.loads(form.schedule_json.data or "[]")
		course_id = _course_dao().create_course(
			course_code=form.course_code.data,
			course_name=form.course_name.data,
			department=form.department.data,
			instructor=form.instructor.data,
			schedule=schedule,
		)
		flash("Course created", "success")
		return redirect(url_for("admin_courses.course_detail", course_id=course_id))
	return render_template("courses/course_form.html", form=form, action="add")


@courses_bp.post("/courses/<course_id>/edit")
@login_required
@role_required("super_admin", "admin")
def edit_course(course_id: str):
	fields = {
		"course_code": request.form.get("course_code"),
		"course_name": request.form.get("course_name"),
		"department": request.form.get("department"),
		"instructor": request.form.get("instructor"),
	}
	schedule_json = request.form.get("schedule_json")
	if schedule_json:
		fields["schedule"] = json.loads(schedule_json)
	updated = _course_dao().update_course(course_id, {k: v for k, v in fields.items() if v not in (None, "")})
	if not updated:
		return jsonify({"error": "course not found"}), 404
	flash("Course updated", "success")
	return redirect(url_for("admin_courses.course_detail", course_id=course_id))


@courses_bp.post("/courses/<course_id>/enroll")
@login_required
@role_required("super_admin", "admin")
def enroll_students(course_id: str):
	student_ids = request.form.getlist("student_ids") or ([] if not request.form.get("student_ids") else [request.form.get("student_ids")])
	count = 0
	for student_id in student_ids:
		if _course_dao().enroll_student(course_id, student_id):
			count += 1
	flash(f"Enrolled {count} students", "success")
	return redirect(url_for("admin_courses.course_detail", course_id=course_id))


@courses_bp.post("/courses/<course_id>/session/open")
@login_required
@role_required("super_admin", "admin")
def open_session(course_id: str):
	settings = {
		"recognition_mode": request.form.get("recognition_mode", "BALANCED"),
		"anti_spoofing_enabled": request.form.get("anti_spoofing_enabled", "1") not in {"0", "false", "False"},
		"late_threshold_minutes": int(request.form.get("late_threshold_minutes", 15)),
	}
	from core.models import SessionDAO

	session_dao = SessionDAO(_course_dao().db)
	opened = session_dao.open_session(course_id, current_user.id, settings)
	if not opened.get("opened") and opened.get("reason") == "invalid_id":
		return jsonify({"error": "invalid ids"}), 400
	flash("Session opened", "success")
	return redirect(url_for("admin_courses.course_detail", course_id=course_id))


@courses_bp.post("/courses/<course_id>/session/close")
@login_required
@role_required("super_admin", "admin")
def close_session(course_id: str):
	from core.models import AttendanceDAO, SessionDAO

	db = _course_dao().db
	session_dao = SessionDAO(db)
	attendance_dao = AttendanceDAO(db)
	session = session_dao.get_open_session(course_id)
	if not session:
		return jsonify({"error": "no open session"}), 404
	stats = attendance_dao.get_stats(course_id)
	session_dao.close_session(str(session["_id"]))
	flash(f"Session closed: {stats}", "success")
	return redirect(url_for("admin_courses.course_detail", course_id=course_id))
