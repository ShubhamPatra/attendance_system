"""Routes for the standalone student portal."""

from __future__ import annotations

import base64
import os
import uuid
from functools import wraps

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user

import core.config as config
from core.utils import sanitize_string, setup_logging

from . import database
from .auth import StudentUser, authenticate_student
from .verification import evaluate_student_samples

logger = setup_logging()


bp = Blueprint("student", __name__)


def login_required(view_func):
    """No-op decorator: student authentication checks are disabled."""

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        return view_func(*args, **kwargs)

    return wrapped


def _student_dir(reg_no: str) -> str:
    path = os.path.join(config.STUDENT_SAMPLE_DIR, reg_no)
    os.makedirs(path, exist_ok=True)
    return path


def _decode_capture(value: str) -> bytes:
    if "," in value:
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def _save_capture_frame(reg_no: str, frame_bytes: bytes, index: int) -> str:
    directory = _student_dir(reg_no)
    filename = f"capture_{index}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(directory, filename)
    with open(path, "wb") as handle:
        handle.write(frame_bytes)
    return path


def _resolve_registration_number() -> str:
    """Resolve active registration number from session or request payload."""
    if getattr(current_user, "is_authenticated", False):
        value = sanitize_string(getattr(current_user, "registration_number", "")).strip()
        if value:
            return value

    for source in (request.args, request.form):
        value = sanitize_string(source.get("registration_number", "")).strip()
        if value:
            return value

    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        value = sanitize_string(str(payload.get("registration_number", ""))).strip()
        if value:
            return value

    return ""


@bp.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("student.status"))
    return redirect(url_for("student.login"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("student.capture"))

    if request.method == "GET":
        return render_template("register.html")

    name = sanitize_string(request.form.get("name", "")).strip()
    email = sanitize_string(request.form.get("email", "")).strip().lower()
    registration_number = sanitize_string(request.form.get("registration_number", "")).strip()
    semester_str = sanitize_string(request.form.get("semester", "")).strip()
    section = sanitize_string(request.form.get("section", "")).strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    errors: list[str] = []
    if not name:
        errors.append("Name is required.")
    if not email:
        errors.append("Email is required.")
    if not registration_number:
        errors.append("Registration number is required.")
    if not semester_str:
        errors.append("Semester is required.")
    if not section:
        errors.append("Section is required.")
    if not password or len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    if password != confirm_password:
        errors.append("Passwords do not match.")
    if database.get_student_by_reg_no(registration_number) is not None:
        errors.append("Registration number already exists.")
    if email and database.get_student_by_email(email) is not None:
        errors.append("Email already exists.")

    # Validate semester as integer
    try:
        semester = int(semester_str)
        if semester < 1 or semester > 8:
            errors.append("Semester must be between 1 and 8.")
    except ValueError:
        errors.append("Semester must be a valid number.")

    if errors:
        for error in errors:
            flash(error, "danger")
        return render_template("register.html"), 400

    try:
        student_id = database.create_student_account(
            name,
            email,
            registration_number,
            password,
            semester=semester,
            section=section,
        )
    except ValueError as exc:
        flash(str(exc), "danger")
        return render_template("register.html"), 409

    student_doc = database.get_student_by_id(student_id, include_sensitive=False)
    if student_doc:
        login_user(
            StudentUser(
                student_id=str(student_doc["_id"]),
                registration_number=student_doc.get("registration_number", ""),
                name=student_doc.get("name", ""),
                verification_status=student_doc.get("verification_status", "pending"),
            )
        )

    flash("Account created. Capture your face samples to complete verification.", "success")
    return redirect(url_for("student.capture"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("student.status"))

    if request.method == "GET":
        return render_template("login.html")

    credential = sanitize_string(request.form.get("credential", "")).strip()
    password = request.form.get("password", "")

    if not credential or not password:
        flash("Registration number/email and password are required.", "danger")
        return render_template("login.html"), 400

    student = authenticate_student(credential, password)
    if student is None:
        flash("Invalid credentials.", "danger")
        return render_template("login.html"), 401

    login_user(student, remember=True)
    return redirect(url_for("student.status"))


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("student.login"))


@bp.route("/status")
@login_required
def status():
    reg_no = _resolve_registration_number()
    if not reg_no:
        return render_template(
            "status.html",
            student=None,
            error="registration_number is required in query or form data when not logged in.",
        ), 400

    student = database.get_student_status(reg_no)
    return render_template("status.html", student=student)


@bp.route("/capture")
@login_required
def capture():
    reg_no = _resolve_registration_number()
    if not reg_no:
        return render_template(
            "capture.html",
            student=None,
            min_images=config.STUDENT_MIN_CAPTURE_IMAGES,
            max_images=config.STUDENT_MAX_CAPTURE_IMAGES,
            error="registration_number is required in query or form data when not logged in.",
        ), 400

    student = database.get_student_status(reg_no)
    return render_template(
        "capture.html",
        student=student,
        min_images=config.STUDENT_MIN_CAPTURE_IMAGES,
        max_images=config.STUDENT_MAX_CAPTURE_IMAGES,
    )


@bp.route("/api/capture", methods=["POST"])
@login_required
def api_capture():
    payload = request.get_json(silent=True) or {}
    reg_no = _resolve_registration_number()
    if not reg_no:
        return jsonify({"error": "registration_number is required when not logged in."}), 400

    frames = payload.get("frames") or []

    if not isinstance(frames, list):
        return jsonify({"error": "frames must be a list of webcam captures."}), 400
    if len(frames) < config.STUDENT_MIN_CAPTURE_IMAGES:
        return jsonify({"error": f"At least {config.STUDENT_MIN_CAPTURE_IMAGES} frames are required."}), 400
    if len(frames) > config.STUDENT_MAX_CAPTURE_IMAGES:
        return jsonify({"error": f"No more than {config.STUDENT_MAX_CAPTURE_IMAGES} frames are allowed."}), 400

    sample_paths: list[str] = []
    for index, frame in enumerate(frames, 1):
        try:
            raw_bytes = _decode_capture(str(frame))
        except Exception:
            return jsonify({"error": f"Frame {index} is not valid base64 image data."}), 400

        if len(raw_bytes) > config.UPLOAD_MAX_SIZE:
            return jsonify({"error": f"Frame {index} exceeds the maximum image size."}), 400

        path = _save_capture_frame(reg_no, raw_bytes, index)
        sample_paths.append(path)

    result = evaluate_student_samples(sample_paths, registration_number=reg_no)
    database.save_verification_result(reg_no, result, sample_paths)

    return jsonify(result.to_dict())


@bp.route("/attendance")
@login_required
def attendance():
    reg_no = _resolve_registration_number()
    if not reg_no:
        return render_template(
            "attendance.html",
            student=None,
            error="registration_number is required in query or form data when not logged in.",
        ), 400

    try:
        student = database.get_student_status(reg_no)
    except Exception as e:
        logger.error(f"Error fetching student status for {reg_no}: {e}", exc_info=True)
        flash("Error retrieving student information. Please try again.", "danger")
        return redirect(url_for("student.status"))

    if not student:
        flash("Student record not found.", "danger")
        return redirect(url_for("student.status"))
    if student.get("verification_status") != "approved":
        flash("Attendance is available after verification is approved.", "warning")
        return redirect(url_for("student.status"))

    date = sanitize_string(request.args.get("date", "")).strip() or None
    month = sanitize_string(request.args.get("month", "")).strip() or None
    
    try:
        overview = database.get_attendance_overview(reg_no, date=date, month=month)
    except Exception as e:
        logger.error(f"Error fetching attendance overview for {reg_no}: {e}", exc_info=True)
        flash("Error retrieving attendance records. Please try again.", "danger")
        return redirect(url_for("student.status"))
    
    if overview is None:
        flash("Attendance history not found.", "danger")
        return redirect(url_for("student.status"))
    return render_template("attendance.html", student=overview)


def register_routes(app, url_prefix: str = "/student") -> None:
    app.register_blueprint(bp, url_prefix=url_prefix)
