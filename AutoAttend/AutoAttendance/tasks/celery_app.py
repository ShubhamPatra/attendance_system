"""Celery application setup for background tasks."""

from __future__ import annotations

import os

from celery import Celery
from celery.schedules import crontab

from core.config import load_config


def _resolve_celery_urls() -> tuple[str, str]:
	try:
		cfg = load_config()
		return cfg.CELERY_BROKER_URL, cfg.CELERY_RESULT_BACKEND
	except Exception:
		broker = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
		backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
		return broker, backend


def create_celery() -> Celery:
	broker_url, result_backend = _resolve_celery_urls()
	app = Celery("autoattendance", broker=broker_url, backend=result_backend)

	app.conf.update(
		task_serializer="json",
		result_serializer="json",
		accept_content=["json"],
		task_acks_late=True,
		worker_prefetch_multiplier=1,
		task_default_retry_delay=1,
		task_annotations={"*": {"max_retries": 3, "retry_backoff": True}},
		beat_schedule={
			"cleanup-expired-verification-sessions": {
				"task": "tasks.cleanup_tasks.cleanup_expired_sessions",
				"schedule": 300.0,
			},
			"verify-log-ttl-index": {
				"task": "tasks.cleanup_tasks.archive_old_logs",
				"schedule": crontab(hour=3, minute=0),
			},
		},
	)

	app.autodiscover_tasks(["tasks"])
	return app


celery_app = create_celery()
