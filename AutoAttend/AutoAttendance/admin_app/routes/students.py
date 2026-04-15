from __future__ import annotations

import csv
import io
import os
from pathlib import Path

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import login_required

from admin_app.forms import StudentForm
from admin_app.routes.auth import role_required
from core.models import AttendanceDAO
from core.utils import validate_file_upload
from tasks.embedding_tasks import generate_student_embedding


students_bp = Blueprint("admin_students", __name__)


def _student_dao():
	from admin_app import get_student_dao

	return get_student_dao()


def _project_root() -> Path:
	return Path(current_app.root_path).parent


def _student_upload_dir() -> Path:
	upload_dir = _project_root() / "uploads" / "students"
	upload_dir.mkdir(parents=True, exist_ok=True)
	return upload_dir


@students_bp.get("/students")
@login_required
@role_required("super_admin", "admin")
def list_students():
	query = request.args.get("q")
	department = request.args.get("department")
	year = request.args.get("year")
	page = int(request.args.get("page", 1))
	per_page = int(request.args.get("per_page", 20))

	search = _student_dao().search_students(query=query, department=department, year=int(year) if year else None, page=page, per_page=per_page)
	return render_template("students/student_list.html", students=search)


@students_bp.get("/students/<student_id>")
@login_required
@role_required("super_admin", "admin")
def student_detail(student_id: str):
	dao = _student_dao()
	student = dao.get_by_student_id(student_id)
	if not student:
		return jsonify({"error": "student not found"}), 404

	attendance_dao = AttendanceDAO(dao.db)
	attendance_records = list(attendance_dao.collection.find({"student_id": student["_id"]}).sort("created_at", -1))
	attendance_summary = {"present": 0, "late": 0, "absent": 0}
	for record in attendance_records:
		status = record.get("status")
		if status in attendance_summary:
			attendance_summary[status] += 1
	return render_template(
		"students/student_detail.html",
		student=student,
		attendance_summary=attendance_summary,
		attendance_records=attendance_records,
		embedding_count=len(student.get("face_embeddings", [])),
	)


@students_bp.route("/students/add", methods=["GET", "POST"])
@login_required
@role_required("super_admin", "admin")
def add_student():
	form = StudentForm()
	if form.validate_on_submit():
		_student_dao().create_student(
			student_id=form.student_id.data,
			email=form.email.data,
			password="Student123!",
			name=form.name.data,
			department=form.department.data,
			year=form.year.data or 1,
		)
		flash("Student created", "success")
		return redirect(url_for("admin_students.list_students"))

	return render_template("students/student_form.html", form=form, action="add")


@students_bp.post("/students/<student_id>/edit")
@login_required
@role_required("super_admin", "admin")
def edit_student(student_id: str):
	fields = {
		"student_id": request.form.get("student_id", student_id),
		"name": request.form.get("name"),
		"email": request.form.get("email"),
		"department": request.form.get("department"),
		"year": int(request.form.get("year", 1)),
		"face_photo_path": request.form.get("face_photo_path"),
	}
	updated = _student_dao().update_student(student_id, {k: v for k, v in fields.items() if v not in (None, "")})
	if not updated:
		return jsonify({"error": "student not found"}), 404
	flash("Student updated", "success")
	return redirect(url_for("admin_students.student_detail", student_id=student_id))


@students_bp.post("/students/<student_id>/delete")
@login_required
@role_required("super_admin", "admin")
def delete_student(student_id: str):
	if not _student_dao().soft_delete_student(student_id):
		return jsonify({"error": "student not found"}), 404
	flash("Student deactivated", "success")
	return redirect(url_for("admin_students.list_students"))


@students_bp.post("/students/<student_id>/upload-face")
@login_required
@role_required("super_admin", "admin")
def upload_face(student_id: str):
	file = request.files.get("face_photo")
	if not file or not file.filename:
		return jsonify({"error": "missing file"}), 400
	if not validate_file_upload(file.filename):
		return jsonify({"error": "invalid file type"}), 400
	file.stream.seek(0, os.SEEK_END)
	file_size = file.stream.tell()
	file.stream.seek(0)
	if file_size > 5 * 1024 * 1024:
		return jsonify({"error": "file too large"}), 400

	upload_dir = _student_upload_dir()
	file_path = upload_dir / f"{student_id}_{file.filename}"
	file.save(file_path)
	task = generate_student_embedding.delay(student_id, str(file_path))
	_student_dao().update_student(student_id, {"face_photo_path": str(file_path)})
	return jsonify({"status": "queued", "task": task, "file_path": str(file_path)})


@students_bp.post("/students/bulk-upload")
@login_required
@role_required("super_admin", "admin")
def bulk_upload():
	file = request.files.get("csv_file")
	if not file or not file.filename.lower().endswith(".csv"):
		return jsonify({"error": "csv file required"}), 400

	rows: list[dict[str, str]] = []
	content = io.StringIO(file.stream.read().decode("utf-8-sig"))
	reader = csv.DictReader(content)
	for row in reader:
		rows.append(row)

	created = _student_dao().bulk_create_students(rows)
	return jsonify({"created": created, "rows": len(rows)})
