from .attendance import attendance_bp
from .auth import auth_bp
from .profile import profile_bp

__all__ = ["auth_bp", "attendance_bp", "profile_bp"]
