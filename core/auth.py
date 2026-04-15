"""Authentication utilities for the attendance system."""

from werkzeug.security import generate_password_hash, check_password_hash


def hash_password(password: str) -> str:
    """Hash a password using werkzeug."""
    return generate_password_hash(password, method='pbkdf2:sha256')


def check_password(password_hash: str, password: str) -> bool:
    """Verify a password against its hash."""
    return check_password_hash(password_hash, password)
