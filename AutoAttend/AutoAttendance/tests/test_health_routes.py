from __future__ import annotations

from types import SimpleNamespace

import admin_app as admin_app_module
import student_app as student_app_module

from admin_app import create_app as create_admin_app
from student_app import create_app as create_student_app


class _FakeMongoDB:
	def command(self, name: str):
		assert name == "ping"
		return {"ok": 1}


class _FakeMongoClient:
	def get_db(self):
		return _FakeMongoDB()


class _FakeRedis:
	def ping(self):
		return True


def test_admin_health_reports_connected(monkeypatch):
	monkeypatch.setattr(admin_app_module, "get_mongo_client", lambda _cfg: _FakeMongoClient())
	monkeypatch.setattr(admin_app_module.redis.Redis, "from_url", lambda _url: _FakeRedis())

	app = create_admin_app()
	client = app.test_client()

	response = client.get("/admin/health")
	assert response.status_code == 200
	assert response.get_json() == {"status": "ok", "service": "admin", "db": "connected", "redis": "connected"}


def test_student_health_reports_degraded_when_redis_down(monkeypatch):
	monkeypatch.setattr(student_app_module, "get_mongo_client", lambda _cfg: _FakeMongoClient())

	class _BrokenRedis:
		def ping(self):
			raise RuntimeError("down")

	monkeypatch.setattr(student_app_module.redis.Redis, "from_url", lambda _url: _BrokenRedis())

	app = create_student_app()
	client = app.test_client()

	response = client.get("/student/health")
	assert response.status_code == 503
	assert response.get_json() == {"status": "degraded", "service": "student", "db": "connected", "redis": "disconnected"}