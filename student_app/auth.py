"""Student portal authentication helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from bson import ObjectId
from flask import Flask
from flask_login import LoginManager, UserMixin

import app_core.config as config
from app_core.auth import verify_password
from app_core.utils import setup_logging

from . import database


login_manager = LoginManager()
logger = setup_logging()


@dataclass
class StudentUser(UserMixin):
    """Adapter around student documents for Flask-Login."""

    student_id: str
    registration_number: str
    name: str
    verification_status: str

    @property
    def id(self) -> str:
        return self.student_id


@login_manager.user_loader
def load_student(user_id: str) -> StudentUser | None:
    try:
        oid = ObjectId(user_id)
    except Exception as exc:
        logger.debug("Invalid student user id: %s", exc)
        return None

    student = database.get_student_by_id(oid, include_sensitive=True)
    if not student:
        return None

    return StudentUser(
        student_id=str(student["_id"]),
        registration_number=student.get("registration_number", ""),
        name=student.get("name", ""),
        verification_status=student.get("verification_status", "pending"),
    )


def init_auth(app: Flask) -> None:
    login_manager.login_view = "student.login"
    login_manager.session_protection = "strong"
    login_manager.init_app(app)
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(
        hours=max(1, config.AUTH_SESSION_DURATION_HOURS)
    )


def authenticate_student(credential: str, password: str) -> StudentUser | None:
    """Authenticate student by registration number or email."""
    # Try to fetch by registration number first
    student = database.get_student_by_reg_no(credential, include_sensitive=True)
    
    # If not found, try by email
    if not student:
        student = database.get_student_by_email(credential, include_sensitive=True)
    
    if not student:
        return None
    if not verify_password(student.get("password_hash", ""), password):
        return None

    return StudentUser(
        student_id=str(student["_id"]),
        registration_number=student.get("registration_number", ""),
        name=student.get("name", ""),
        verification_status=student.get("verification_status", "pending"),
    )
