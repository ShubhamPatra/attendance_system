"""Student portal authentication stub (removed for ML-only focus)."""

from flask_login import UserMixin, LoginManager
from core.auth import check_password
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
    
    # Verify password
    password_hash = student.get("password_hash")
    if not password_hash or not check_password(password_hash, password):
        return None
    
    # Return authenticated user
    return StudentUser(
        student_id=str(student.get("_id")),
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
