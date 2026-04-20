"""Routes for the standalone student portal."""

from __future__ import annotations

import base64
import os
import uuid
import cv2
import numpy as np
from functools import wraps

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user

import core.config as config
from core.utils import sanitize_string, setup_logging
from core.auth import validate_password
import core.database as core_db

from . import database
from .auth import StudentUser, authenticate_student
from .verification import evaluate_student_samples
from .enrollment_validator import EnrollmentValidator  # PHASE 4

logger = setup_logging()


bp = Blueprint("student", __name__)


def login_required(view_func):
    """
    Decorator to require student login for protected routes.
    
    Redirects unauthenticated users to login page.
    For JSON requests (API), returns 401 Unauthorized.
    """
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            if request.headers.get('Accept') == 'application/json' or request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('student.login'))
        
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


def _decode_capture_to_image(value: str) -> np.ndarray:
    """Decode base64 frame to BGR image for validation.
    
    PHASE 4: Used for enrollment image validation before saving.
    
    Args:
        value: Base64-encoded image data (with or without data URL prefix)
    
    Returns:
        BGR numpy array, or None if decoding fails
    """
    try:
        raw_bytes = _decode_capture(value)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image_bgr
    except Exception as e:
        logger.debug("Failed to decode capture to image: %s", e)
        return None


def _save_capture_frame(reg_no: str, frame_bytes: bytes, index: int) -> str:
    directory = _student_dir(reg_no)
    filename = f"capture_{index}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(directory, filename)
    with open(path, "wb") as handle:
        handle.write(frame_bytes)
    return path


def _resolve_registration_number() -> str:
    """
    Resolve active registration number from authenticated session.
    
    SECURITY: Only return registration number from current_user if authenticated.
    Do NOT accept registration_number from query/form params for security reasons.
    This prevents attackers from accessing other students' data.
    
    Returns:
        Registration number if authenticated, empty string otherwise
    """
    if getattr(current_user, "is_authenticated", False):
        value = sanitize_string(getattr(current_user, "registration_number", "")).strip()
        if value:
            return value
    
    # For unauthenticated requests, return empty string
    # The calling route (protected by @login_required) will handle the error
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
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')

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
    
    # Validate password strength
    if not password:
        errors.append("Password is required.")
    else:
        is_valid, error_msg = validate_password(password, registration_number, email)
        if not is_valid:
            errors.append(error_msg)
    
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
        # Log registration
        core_db.log_auth_event(
            event_type='REGISTRATION',
            user_id=student_id,
            status='success',
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Auto-login user
        login_user(
            StudentUser(
                student_id=str(student_id),
                registration_number=student_doc.get("registration_number", ""),
                name=student_doc.get("name", ""),
                verification_status=student_doc.get("verification_status", "pending"),
            )
        )
        
        flash("Account created successfully! Please upload your face samples to complete enrollment.", "success")
        return redirect(url_for("student.portal"))

    flash("Failed to create account. Please try again.", "danger")
    return render_template("register.html"), 500





@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("student.status"))

    if request.method == "GET":
        return render_template("login.html")

    credential = sanitize_string(request.form.get("credential", "")).strip()
    password = request.form.get("password", "")
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')

    if not credential or not password:
        flash("Registration number/email and password are required.", "danger")
        return render_template("login.html"), 400

    student = authenticate_student(credential, password)
    if student is None:
        # Check if account is locked (for better error message)
        from . import database as student_db
        
        # Try to get student for lockout check
        test_student = student_db.get_student_by_reg_no(credential)
        if not test_student:
            test_student = student_db.get_student_by_email(credential)
        
        if test_student:
            lockout_status = core_db.get_account_lockout_status(test_student.get("_id"), threshold=5, lockout_minutes=30)
            
            # Log failed login attempt
            core_db.log_auth_event(
                event_type='LOGIN_FAILED',
                user_id=test_student.get("_id"),
                status='locked' if lockout_status['is_locked'] else 'invalid_password',
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            if lockout_status['is_locked']:
                flash(
                    f"Account locked due to too many failed login attempts. "
                    f"Please try again in {lockout_status['minutes_until_unlock']} minutes.",
                    "danger"
                )
                return render_template("login.html"), 403
        
        flash("Invalid credentials.", "danger")
        return render_template("login.html"), 401

    # Log successful login
    core_db.log_auth_event(
        event_type='LOGIN',
        user_id=student.id,
        status='success',
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    login_user(student, remember=True)
    
    return redirect(url_for("student.status"))


@bp.route("/logout")
@login_required
def logout():
    # Log logout event
    ip_address = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    if current_user.is_authenticated:
        core_db.log_auth_event(
            event_type='LOGOUT',
            user_id=current_user.id,
            status='success',
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    logout_user()
    return redirect(url_for("student.login"))


@bp.route("/portal")
@login_required
def portal():
    """Unified student portal: shows photo upload if pending, status if verified."""
    reg_no = _resolve_registration_number()
    if not reg_no:
        return render_template(
            "student_portal_unified.html",
            student=None,
            error="registration_number is required in query or form data when not logged in.",
            min_images=config.STUDENT_MIN_CAPTURE_IMAGES,
            max_images=config.STUDENT_MAX_CAPTURE_IMAGES,
        ), 400

    try:
        student = database.get_student_status(reg_no)
        if not student:
            return render_template(
                "student_portal_unified.html",
                student=None,
                error="Student record not found.",
                min_images=config.STUDENT_MIN_CAPTURE_IMAGES,
                max_images=config.STUDENT_MAX_CAPTURE_IMAGES,
            ), 404

        # For verified students, get detailed attendance data
        if student.get("verification_status") in ["approved", "verified"]:
            try:
                student = database.get_attendance_overview(reg_no)
            except Exception as e:
                logger.error(f"Error fetching attendance overview for {reg_no}: {e}", exc_info=True)
                # Fall back to status data if attendance fetch fails
                pass
        
        return render_template(
            "student_portal_unified.html",
            student=student,
            min_images=config.STUDENT_MIN_CAPTURE_IMAGES,
            max_images=config.STUDENT_MAX_CAPTURE_IMAGES,
        )
    except Exception as e:
        logger.error(f"Error loading portal for {reg_no}: {e}", exc_info=True)
        return render_template(
            "student_portal_unified.html",
            student=None,
            error="Error loading portal data.",
            min_images=config.STUDENT_MIN_CAPTURE_IMAGES,
            max_images=config.STUDENT_MAX_CAPTURE_IMAGES,
        ), 500


@bp.route("/status")
@login_required
def status():
    """Legacy status route - redirects to unified portal."""
    return redirect(url_for("student.portal"))


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
    """Process student face enrollment with multi-angle validation (PHASE 4)."""
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

    # PHASE 4: Decode frames and validate before saving
    images_bgr: list[np.ndarray] = []
    decode_errors: list[str] = []
    
    for index, frame in enumerate(frames, 1):
        try:
            image_bgr = _decode_capture_to_image(str(frame))
            if image_bgr is None:
                decode_errors.append(f"Frame {index}: Could not decode image data")
                continue
            images_bgr.append(image_bgr)
        except Exception as e:
            logger.debug("Failed to decode frame %d: %s", index, e)
            decode_errors.append(f"Frame {index}: Invalid base64 encoding")
    
    if decode_errors:
        return jsonify({
            "success": False,
            "error": "Some frames could not be decoded",
            "details": decode_errors,
        }), 400
    
    # PHASE 4: Validate enrollment images with EnrollmentValidator
    student_id = core_db.get_student_by_reg_no(reg_no)
    if student_id and "_id" in student_id:
        student_id = student_id["_id"]
    
    validator_result = EnrollmentValidator.validate_multi_angle_enrollment(
        images_bgr,
        student_id=student_id,
    )
    
    if not validator_result.get("valid"):
        # Return detailed validation feedback for UI guidance
        error_msg = validator_result.get("error", "Enrollment validation failed")
        image_feedback = []
        for img_result in validator_result.get("image_results", []):
            if img_result.get("valid"):
                image_feedback.append({
                    "index": img_result.get("index"),
                    "status": "✓ Good",
                    "message": "Image quality acceptable",
                })
            else:
                image_feedback.append({
                    "index": img_result.get("index"),
                    "status": "✗ Failed",
                    "message": img_result.get("error", "Image validation failed"),
                })
        
        angle_info = validator_result.get("angle_diversity", {})
        
        return jsonify({
            "success": False,
            "error": error_msg,
            "image_results": image_feedback,
            "angle_diversity": {
                "valid": angle_info.get("valid"),
                "message": angle_info.get("error") or "Angle diversity is sufficient",
            },
            "duplicate_detected": validator_result.get("duplicate_detected"),
        }), 400
    
    # Validation passed - save frames and proceed with verification
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

    return jsonify({
        **result.to_dict(),
        "enrollment_validated": True,  # PHASE 4: Indicate quality validation passed
    })


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
