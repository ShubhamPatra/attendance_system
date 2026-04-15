"""Reusable utility functions used across the project."""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime

import bcrypt


STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,30}$")


def hash_password(plain: str) -> str:
	encoded = plain.encode("utf-8")
	return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def check_password(plain: str, hashed: str) -> bool:
	return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def generate_id() -> str:
	return uuid.uuid4().hex


def format_datetime(dt: datetime) -> str:
	return dt.isoformat()


def parse_date(value: str) -> date:
	return datetime.strptime(value, "%Y-%m-%d").date()


def get_today() -> date:
	return date.today()


def validate_email(email: str) -> bool:
	return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))


def validate_student_id(sid: str) -> bool:
	return bool(STUDENT_ID_PATTERN.match(sid or ""))


def validate_file_upload(filename: str, allowed_extensions: set[str] | None = None) -> bool:
	if not filename or "." not in filename:
		return False
	if allowed_extensions is None:
		allowed_extensions = {"png", "jpg", "jpeg"}
	ext = filename.rsplit(".", 1)[1].lower()
	return ext in allowed_extensions
