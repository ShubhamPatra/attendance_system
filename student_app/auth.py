"""Student portal authentication."""

from flask import request
from flask_login import UserMixin, LoginManager
from core.auth import check_password
import core.database as core_db
from . import database


class StudentUser(UserMixin):
    """Flask-Login user object for student portal."""

    def __init__(self, student_id: str, registration_number: str, name: str, verification_status: str):
        self.id = student_id
        self.registration_number = registration_number
        self.name = name
        self.verification_status = verification_status

    def get_id(self):
        return self.id


def authenticate_student(credential: str, password: str) -> StudentUser | None:
    """Authenticate a student by registration number or email.
    
    Implements account lockout after 5 failed attempts in 15 minutes.
    
    Args:
        credential: Registration number or email
        password: Password to verify
        
    Returns:
        StudentUser if authentication succeeds, None otherwise
    """
    # Try to get student by registration number first
    student = database.get_student_by_reg_no(credential, include_sensitive=True)
    
    # If not found, try by email
    if not student:
        student = database.get_student_by_email(credential, include_sensitive=True)
    
    if not student:
        return None
    
    student_id = student.get("_id")
    ip_address = request.remote_addr if request else ""
    user_agent = request.headers.get('User-Agent', '') if request else ""
    
    # Check account lockout status
    lockout_status = core_db.get_account_lockout_status(student_id, threshold=5, lockout_minutes=30)
    if lockout_status['is_locked']:
        # Account is locked; record the attempt and return None
        core_db.record_login_attempt(student_id, success=False, ip_address=ip_address, user_agent=user_agent)
        return None
    
    # Verify password
    password_hash = student.get("password_hash")
    if not password_hash or not check_password(password_hash, password):
        # Record failed attempt
        core_db.record_login_attempt(student_id, success=False, ip_address=ip_address, user_agent=user_agent)
        return None
    
    # Record successful login
    core_db.record_login_attempt(student_id, success=True, ip_address=ip_address, user_agent=user_agent)
    
    # Return authenticated user
    return StudentUser(
        student_id=str(student_id),
        registration_number=student.get("registration_number", ""),
        name=student.get("name", ""),
        verification_status=student.get("verification_status", "pending"),
    )


def init_auth(app):
    """Initialize Flask-Login for the student portal."""
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "student.login"
    
    @login_manager.user_loader
    def load_student_user(student_id: str) -> StudentUser | None:
        """Load a student user by ID."""
        student = database.get_student_by_id(student_id, include_sensitive=False)
        if not student:
            return None
        return StudentUser(
            student_id=str(student.get("_id")),
            registration_number=student.get("registration_number", ""),
            name=student.get("name", ""),
            verification_status=student.get("verification_status", "pending"),
        )
