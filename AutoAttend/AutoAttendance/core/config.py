"""Application configuration classes and environment loading helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


def _to_bool(value: str | None, default: bool = False) -> bool:
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
	if value is None or value == "":
		return default
	return int(value)


def _to_float(value: str | None, default: float) -> float:
	if value is None or value == "":
		return default
	return float(value)


@dataclass
class Config:
	SECRET_KEY: str
	ENV: str
	DEBUG: bool

	MONGODB_URI: str
	MONGODB_DB_NAME: str

	REDIS_URL: str
	CELERY_BROKER_URL: str
	CELERY_RESULT_BACKEND: str

	RECOGNITION_MODE: str
	FACE_DETECTION_CONFIDENCE: float
	RECOGNITION_THRESHOLD: float
	ANTI_SPOOFING_THRESHOLD: float

	ADMIN_EMAIL: str
	ADMIN_DEFAULT_PASSWORD: str

	SOCKETIO_ASYNC_MODE: str

	SESSION_COOKIE_SECURE: bool
	SESSION_COOKIE_HTTPONLY: bool
	SESSION_COOKIE_SAMESITE: str
	PERMANENT_SESSION_LIFETIME: int

	@classmethod
	def from_env(cls) -> "Config":
		load_dotenv()

		secret_key = os.getenv("FLASK_SECRET_KEY")
		mongodb_uri = os.getenv("MONGODB_URI")

		if not secret_key:
			raise ValueError("Missing required environment variable: FLASK_SECRET_KEY")
		if not mongodb_uri:
			raise ValueError("Missing required environment variable: MONGODB_URI")

		return cls(
			SECRET_KEY=secret_key,
			ENV=os.getenv("FLASK_ENV", "development"),
			DEBUG=_to_bool(os.getenv("FLASK_DEBUG"), True),
			MONGODB_URI=mongodb_uri,
			MONGODB_DB_NAME=os.getenv("MONGODB_DB_NAME", "autoattendance"),
			REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
			CELERY_BROKER_URL=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
			CELERY_RESULT_BACKEND=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
			RECOGNITION_MODE=os.getenv("RECOGNITION_MODE", "BALANCED"),
			FACE_DETECTION_CONFIDENCE=_to_float(os.getenv("FACE_DETECTION_CONFIDENCE"), 0.7),
			RECOGNITION_THRESHOLD=_to_float(os.getenv("RECOGNITION_THRESHOLD"), 0.45),
			ANTI_SPOOFING_THRESHOLD=_to_float(os.getenv("ANTI_SPOOFING_THRESHOLD"), 0.5),
			ADMIN_EMAIL=os.getenv("ADMIN_EMAIL", "admin@autoattendance.com"),
			ADMIN_DEFAULT_PASSWORD=os.getenv("ADMIN_DEFAULT_PASSWORD", "changeme123"),
			SOCKETIO_ASYNC_MODE=os.getenv("SOCKETIO_ASYNC_MODE", "eventlet"),
			SESSION_COOKIE_SECURE=_to_bool(os.getenv("SESSION_COOKIE_SECURE"), False),
			SESSION_COOKIE_HTTPONLY=_to_bool(os.getenv("SESSION_COOKIE_HTTPONLY"), True),
			SESSION_COOKIE_SAMESITE=os.getenv("SESSION_COOKIE_SAMESITE", "Lax"),
			PERMANENT_SESSION_LIFETIME=_to_int(os.getenv("PERMANENT_SESSION_LIFETIME"), 1800),
		)


class DevelopmentConfig(Config):
	@classmethod
	def from_env(cls) -> "DevelopmentConfig":
		cfg = Config.from_env()
		cfg.ENV = "development"
		cfg.DEBUG = True
		return cls(**cfg.__dict__)


class TestingConfig(Config):
	@classmethod
	def from_env(cls) -> "TestingConfig":
		cfg = Config.from_env()
		cfg.ENV = "testing"
		cfg.DEBUG = False
		return cls(**cfg.__dict__)


class ProductionConfig(Config):
	@classmethod
	def from_env(cls) -> "ProductionConfig":
		cfg = Config.from_env()
		cfg.ENV = "production"
		cfg.DEBUG = False
		return cls(**cfg.__dict__)


def load_config() -> Config:
	env = os.getenv("FLASK_ENV", "development").strip().lower()
	if env == "production":
		return ProductionConfig.from_env()
	if env == "testing":
		return TestingConfig.from_env()
	return DevelopmentConfig.from_env()
