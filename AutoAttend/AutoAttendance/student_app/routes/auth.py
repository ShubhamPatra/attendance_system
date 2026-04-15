from __future__ import annotations

from bson import ObjectId
from flask import Blueprint, flash, redirect, render_template, url_for
from flask_login import UserMixin, current_user, login_required, login_user, logout_user

from core import limiter, login_manager
from core.models import StudentDAO
from core.utils import check_password
from student_app.forms import RegistrationForm, StudentLoginForm


auth_bp = Blueprint("student_auth", __name__)


class StudentUser(UserMixin):
	def __init__(self, doc):
		self.id = str(doc["_id"])
		self.student_id = doc["student_id"]
		self.email = doc["email"]
		self.name = doc.get("name", "Student")


@login_manager.user_loader
def load_student(user_id: str):
	from student_app import get_student_dao

	try:
		oid = ObjectId(user_id)
	except Exception:
		return None

	student = get_student_dao().collection.find_one({"_id": oid, "is_active": True})
	return StudentUser(student) if student else None


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per 15 minutes")
def login():
	from student_app import get_student_dao

	if current_user.is_authenticated:
		return redirect(url_for("student_profile.profile_home"))

	form = StudentLoginForm()
	if form.validate_on_submit():
		dao: StudentDAO = get_student_dao()
		student = dao.get_by_student_id(form.student_id.data or "")
		if student and check_password(form.password.data or "", student["password_hash"]):
			login_user(StudentUser(student))
			return redirect(url_for("student_profile.profile_home"))
		flash("Invalid student ID or password", "error")

	return render_template("auth/login.html", form=form)


@auth_bp.get("/logout")
@login_required
def logout():
	logout_user()
	return redirect(url_for("student_auth.login"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
	from student_app import get_student_dao

	if current_user.is_authenticated:
		return redirect(url_for("student_profile.profile_home"))

	form = RegistrationForm()
	if form.validate_on_submit():
		dao: StudentDAO = get_student_dao()
		if dao.get_by_student_id(form.student_id.data or ""):
			flash("Student ID already exists", "error")
			return render_template("auth/register.html", form=form)
		if dao.get_by_email(form.email.data or ""):
			flash("Email already exists", "error")
			return render_template("auth/register.html", form=form)

		dao.create_student(
			student_id=form.student_id.data or "",
			email=form.email.data or "",
			password=form.password.data or "",
			name=form.name.data or "",
			department=form.department.data or "",
			year=form.year.data or 1,
		)
		if form.face_photo_path.data:
			dao.update_student(form.student_id.data or "", {"face_photo_path": form.face_photo_path.data})

		student = dao.get_by_student_id(form.student_id.data or "")
		if student:
			login_user(StudentUser(student))
			flash("Registration complete. You can now mark attendance.", "success")
			return redirect(url_for("student_profile.profile_home"))

	return render_template("auth/register.html", form=form)
