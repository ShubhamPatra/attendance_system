from __future__ import annotations

from datetime import datetime, timedelta, timezone

import mongomock

import tasks.cleanup_tasks as cleanup_tasks


class _FakeRedis:
	def __init__(self) -> None:
		self._store: dict[str, str] = {}

	def set(self, key: str, value: str) -> None:
		self._store[key] = value

	def get(self, key: str):
		return self._store.get(key)

	def delete(self, key: str) -> int:
		return 1 if self._store.pop(key, None) is not None else 0

	def scan_iter(self, match: str):
		prefix = match.replace("*", "")
		for key in list(self._store.keys()):
			if key.startswith(prefix):
				yield key


def test_cleanup_expired_sessions_removes_invalid_and_expired(monkeypatch):
	now = datetime.now(timezone.utc)
	client = _FakeRedis()

	client.set(
		"verification:session:valid",
		'{"expires_at": "%s"}' % (now + timedelta(minutes=2)).isoformat(),
	)
	client.set(
		"verification:session:expired",
		'{"expires_at": "%s"}' % (now - timedelta(minutes=1)).isoformat(),
	)
	client.set("verification:session:broken", "not-json")

	monkeypatch.setattr("tasks.cleanup_tasks._get_redis_client", lambda: client)

	result = cleanup_tasks.cleanup_expired_sessions.run()

	assert result["status"] == "success"
	assert result["scanned"] == 3
	assert result["removed"] == 2
	assert client.get("verification:session:valid") is not None
	assert client.get("verification:session:expired") is None


def test_archive_old_logs_reports_ttl_index(monkeypatch):
	db = mongomock.MongoClient()["autoattendance_test"]
	db.system_logs.create_index("timestamp", expireAfterSeconds=7776000)
	db.system_logs.insert_one(
		{"timestamp": datetime.now(timezone.utc) - timedelta(days=120), "event_type": "x"}
	)

	monkeypatch.setattr("tasks.cleanup_tasks._get_db", lambda: db)

	result = cleanup_tasks.archive_old_logs.run()

	assert result["status"] == "success"
	assert result["ttl_index_exists"] is True
	assert isinstance(result["old_logs_pending"], int)
	assert result["old_logs_pending"] >= 0