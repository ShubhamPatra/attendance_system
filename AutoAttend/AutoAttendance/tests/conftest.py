from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import fakeredis
import mongomock
import numpy as np
import pytest

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")
os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("FLASK_DEBUG", "0")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/autoattendance_test")
os.environ.setdefault("MONGODB_DB_NAME", "autoattendance_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
os.environ.setdefault("SOCKETIO_ASYNC_MODE", "eventlet")

from admin_app import create_app as create_admin_app
import admin_app as admin_app_module
from core.database import ensure_indexes
from core.models import AdminDAO, CourseDAO, StudentDAO
from student_app import create_app as create_student_app
import student_app as student_app_module


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _seed_data(db) -> None:
	ensure_indexes(db)
	admin_dao = AdminDAO(db)
	student_dao = StudentDAO(db)
	course_dao = CourseDAO(db)

	admin_dao.create_admin("admin@example.com", "Secret123!", "Admin", "admin")
	admin_dao.create_admin("viewer@example.com", "Secret123!", "Viewer", "viewer")
	student_dao.create_student("CS2026001", "student@example.com", "Secret123!", "Student One", "CS", 2)
	course_id = course_dao.create_course(
		"CS101",
		"Intro to Computing",
		"CS",
		"Prof. Ada",
		[{"day": "Monday", "start_time": "09:00", "end_time": "10:00", "room": "A1"}],
	)
	course_dao.enroll_student(course_id, "CS2026001")


def _make_mongo_client():
	return mongomock.MongoClient()


def _build_app(app_factory, app_module, db, redis_client):
	app_module.get_mongo_client = lambda _cfg: SimpleNamespace(get_db=lambda: db)
	app_module.redis.Redis.from_url = lambda _url, decode_responses=True: redis_client
	app = app_factory()
	app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
	return app


@pytest.fixture
def mongo_client():
	return _make_mongo_client()


@pytest.fixture
def test_db(mongo_client):
	db = mongo_client["autoattendance_test"]
	_seed_data(db)
	return db


@pytest.fixture
def redis_client():
	return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def fixtures_dir():
	return FIXTURES_DIR


@pytest.fixture
def sample_face_real_path(fixtures_dir):
	return fixtures_dir / "sample_face_real.png"


@pytest.fixture
def sample_face_spoof_path(fixtures_dir):
	return fixtures_dir / "sample_face_spoof.png"


@pytest.fixture
def sample_embedding(fixtures_dir):
	return np.load(fixtures_dir / "sample_embedding.npy")


@pytest.fixture
def admin_app(test_db, redis_client):
	return _build_app(create_admin_app, admin_app_module, test_db, redis_client)


@pytest.fixture
def student_app(test_db, redis_client):
	return _build_app(create_student_app, student_app_module, test_db, redis_client)


@pytest.fixture
def admin_client(admin_app):
	return admin_app.test_client()


@pytest.fixture
def student_client(student_app):
	return student_app.test_client()


@pytest.fixture
def logged_in_admin(admin_client, test_db):
	def _login(role: str = "admin"):
		email = f"{role}@example.com" if role != "admin" else "admin@example.com"
		if test_db.admins.find_one({"email": email}) is None:
			AdminDAO(test_db).create_admin(email, "Secret123!", role.title(), role)
		response = admin_client.post(
			"/admin/login",
			data={"email": email, "password": "Secret123!", "remember_me": "y"},
			follow_redirects=True,
		)
		assert response.status_code == 200
		return admin_client

	return _login


@pytest.fixture
def logged_in_student(student_client, test_db):
	def _login(student_id: str = "CS2026001"):
		student = test_db.students.find_one({"student_id": student_id})
		if student is None:
			StudentDAO(test_db).create_student(student_id, "student@example.com", "Secret123!", "Student One", "CS", 2)
		response = student_client.post(
			"/student/login",
			data={"student_id": student_id, "password": "Secret123!"},
			follow_redirects=True,
		)
		assert response.status_code == 200
		return student_client

	return _login
