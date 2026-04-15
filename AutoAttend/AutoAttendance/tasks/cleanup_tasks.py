"""Periodic maintenance tasks for Redis sessions and log retention checks."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import redis

from core.config import load_config
from core.database import get_mongo_client
from tasks.celery_app import celery_app


def _utcnow() -> datetime:
	return datetime.now(timezone.utc)


def _get_redis_client():
	try:
		cfg = load_config()
		redis_url = cfg.REDIS_URL
	except Exception:
		redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
	return redis.Redis.from_url(redis_url, decode_responses=True)


def _get_db():
	config = load_config()
	return get_mongo_client(config).get_db()


def _parse_iso_datetime(value: str | None) -> datetime | None:
	if not value:
		return None
	try:
		normalized = value.replace("Z", "+00:00")
		parsed = datetime.fromisoformat(normalized)
		if parsed.tzinfo is None:
			return parsed.replace(tzinfo=timezone.utc)
		return parsed.astimezone(timezone.utc)
	except Exception:
		return None


@celery_app.task(name="tasks.cleanup_tasks.cleanup_expired_sessions")
def cleanup_expired_sessions() -> dict[str, Any]:
	client = _get_redis_client()
	now = _utcnow()
	scanned = 0
	removed = 0

	for key in client.scan_iter(match="verification:session:*"):
		scanned += 1
		payload = client.get(key)
		if not payload:
			continue

		try:
			data = json.loads(payload)
		except Exception:
			removed += int(client.delete(key))
			continue

		expires_at = _parse_iso_datetime(data.get("expires_at"))
		if expires_at is None or expires_at <= now:
			removed += int(client.delete(key))

	return {
		"status": "success",
		"scanned": scanned,
		"removed": removed,
		"timestamp": now.isoformat(),
	}


@celery_app.task(name="tasks.cleanup_tasks.archive_old_logs")
def archive_old_logs() -> dict[str, Any]:
	db = _get_db()
	indexes = db.system_logs.index_information()
	# Store/query logs in naive UTC to stay compatible with mongomock and existing records.
	cutoff = _utcnow().replace(tzinfo=None) - timedelta(days=90)

	ttl_index_exists = any(
		index.get("expireAfterSeconds") == 7776000
		for index in indexes.values()
	)

	old_logs_pending = db.system_logs.count_documents({"timestamp": {"$lt": cutoff}})

	return {
		"status": "success",
		"ttl_index_exists": ttl_index_exists,
		"old_logs_pending": int(old_logs_pending),
		"message": "TTL index handles archival/deletion asynchronously.",
	}
