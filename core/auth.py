"""Authentication utilities for the attendance system."""

import os
import re
import secrets
from datetime import datetime, timedelta
from functools import wraps

from werkzeug.security import generate_password_hash, check_password_hash
from flask import current_app, jsonify, request


def hash_password(password: str) -> str:
    """Hash a password using werkzeug."""
    return generate_password_hash(password, method='pbkdf2:sha256')


def check_password(password_hash: str, password: str) -> bool:
    """Verify a password against its hash."""
    return check_password_hash(password_hash, password)


def validate_password(password: str, registration_number: str = "", email: str = "") -> tuple[bool, str]:
    """
    Validate password strength requirements.
    
    Requirements:
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character (!@#$%^&*)
    - Cannot contain registration number or email
    
    Args:
        password: Password to validate
        registration_number: Student's registration number (to prevent in password)
        email: Student's email (to prevent in password)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password is required"
    
    if len(password) < 12:
        return False, "Password must be at least 12 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*]', password):
        return False, "Password must contain at least one special character (!@#$%^&*)"
    
    # Check if password contains registration number or email
    if registration_number and registration_number.lower() in password.lower():
        return False, "Password cannot contain your registration number"
    
    if email:
        email_local = email.split('@')[0]
        if email_local.lower() in password.lower():
            return False, "Password cannot contain part of your email"
    
    return True, ""


def authenticate_user(username: str, password: str):
    """
    Authenticate an admin/teacher user by username and password.
    
    Args:
        username: Username to authenticate
        password: Password to verify
        
    Returns:
        User document if authentication succeeds, None otherwise
    """
    from core.database import get_user_by_username
    
    user = get_user_by_username(username, include_sensitive=True)
    if not user:
        return None
    
    # Verify password
    password_hash = user.get("password_hash")
    if not password_hash or not check_password(password_hash, password):
        return None
    
    # Check if user is active
    if not user.get("is_active", True):
        return None
    
    return user


def generate_jwt_token(user_id: str, role: str, expires_in_hours: int = 1) -> str:
    """
    Generate a JWT token for API authentication.
    
    Args:
        user_id: User/Student ID
        role: User role (admin, teacher, student)
        expires_in_hours: Token lifetime in hours
        
    Returns:
        JWT token string
    """
    try:
        import jwt
    except ImportError:
        raise ImportError("PyJWT is required for JWT token generation. Install with: pip install PyJWT")
    
    payload = {
        'user_id': str(user_id),
        'role': role,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'jti': secrets.token_urlsafe(32),  # JWT ID for token blacklist
    }
    
    secret_key = current_app.config.get('SECRET_KEY')
    if not secret_key:
        raise ValueError("SECRET_KEY not set in Flask config")
    
    token = jwt.encode(payload, secret_key, algorithm='HS256')
    return token


def verify_jwt_token(token: str) -> dict | None:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        import jwt
    except ImportError:
        return None
    
    try:
        secret_key = current_app.config.get('SECRET_KEY')
        if not secret_key:
            return None
        
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        
        # Check if token is blacklisted
        from core.database import is_token_blacklisted
        if payload.get('jti') and is_token_blacklisted(payload.get('jti')):
            return None
        
        return payload
    except Exception:
        return None


def generate_verification_token(expires_in_hours: int = 24) -> tuple[str, datetime]:
    """
    Generate a verification token for email verification or password reset.
    
    Args:
        expires_in_hours: Token lifetime in hours
        
    Returns:
        Tuple of (token, expiration_datetime)
    """
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
    return token, expires_at


def require_api_auth(f):
    """
    Decorator to require valid JWT token for API endpoints.
    
    Returns 401 Unauthorized if token is missing or invalid.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        payload = verify_jwt_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Store user info in request context for use in route handler
        request.user_id = payload.get('user_id')
        request.user_role = payload.get('user_role')
        
        return f(*args, **kwargs)
    
    return decorated_function
