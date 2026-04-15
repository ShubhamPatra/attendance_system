from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required


profile_bp = Blueprint("student_profile", __name__)


@profile_bp.get("/profile")
@login_required
def profile_home():
	from student_app import get_student_dao

	student = get_student_dao().get_by_student_id(current_user.student_id)
	return render_template("profile/profile.html", student=student)


@profile_bp.post("/profile/update-face")
@login_required
def update_face_profile():
	from student_app import get_student_dao

	face_photo_path = (request.form.get("face_photo_path") or "").strip()
	if not face_photo_path:
		flash("Face photo path is required", "error")
		return redirect(url_for("student_profile.profile_home"))

	get_student_dao().update_student(current_user.student_id, {"face_photo_path": face_photo_path})
	flash("Face profile updated", "success")
	return redirect(url_for("student_profile.profile_home"))


@profile_bp.get("/profile/courses")
@login_required
def profile_courses():
	from student_app import get_student_dao

	student_dao = get_student_dao()
	courses = student_dao.db.courses.find(
		{
			"_id": {
				"$in": (student_dao.get_by_student_id(current_user.student_id) or {}).get("enrolled_courses", [])
			},
			"is_active": True,
		}
	)
	return render_template("profile/courses.html", courses=list(courses))
