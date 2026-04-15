"""Authentication helpers and Flask-Login integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from bson import ObjectId
from flask import Flask
from flask_login import LoginManager, UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

import app_core.config as config
import app_core.database as database
from app_core.utils import setup_logging


login_manager = LoginManager()
logger = setup_logging()


@dataclass
class AppUser(UserMixin):
    """Small adapter around user documents for Flask-Login."""

    user_id: str
    username: str
    role: str
    is_active_flag: bool = True

    @property
    def id(self) -> str:
        return self.user_id

    @property
    def is_active(self) -> bool:
        return self.is_active_flag

    def has_role(self, *roles: str) -> bool:
        return self.role in roles


@login_manager.user_loader
def load_user(user_id: str) -> AppUser | None:
    """Load authenticated user from session id."""
    try:
        oid = ObjectId(user_id)
    except Exception as exc:
        logger.debug("Invalid user_id format: %s", exc)
        return None

    doc = database.get_user_by_id(oid)
    if not doc:
        return None

    return AppUser(
        user_id=str(doc["_id"]),
        username=doc.get("username", ""),
        role=doc.get("role", "teacher"),
        is_active_flag=bool(doc.get("is_active", True)),
    )


def init_auth(app: Flask) -> None:
    """Configure Flask-Login and secure session behavior."""
    login_manager.login_view = "main.login"
    login_manager.session_protection = "strong"
    login_manager.init_app(app)

    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(
        hours=max(1, config.AUTH_SESSION_DURATION_HOURS)
    )


def hash_password(password: str) -> str:
    """Return a salted password hash suitable for storage."""
    return generate_password_hash(password)


def verify_password(password_hash: str, raw_password: str) -> bool:
    """Verify a raw password against a stored hash."""
    return check_password_hash(password_hash, raw_password)


def authenticate_user(username: str, password: str) -> AppUser | None:
    """Authenticate user credentials and return AppUser on success."""
    doc = database.get_user_by_username(username)
    if not doc:
        return None
    if not doc.get("is_active", True):
        return None
    if not verify_password(doc.get("password_hash", ""), password):
        return None

    database.update_user_last_login(doc["_id"])
    return AppUser(
        user_id=str(doc["_id"]),
        username=doc.get("username", ""),
        role=doc.get("role", "teacher"),
        is_active_flag=True,
    )
