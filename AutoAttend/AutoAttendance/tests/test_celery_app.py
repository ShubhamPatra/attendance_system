from __future__ import annotations

from tasks.celery_app import create_celery


def test_create_celery_uses_env_urls_when_set(monkeypatch):
	monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/9")
	monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/10")

	app = create_celery()

	assert app.main == "autoattendance"
	assert app.conf.broker_url == "redis://localhost:6379/9"
	assert app.conf.result_backend == "redis://localhost:6379/10"


def test_create_celery_has_required_worker_and_serializer_settings():
	app = create_celery()

	assert app.conf.task_serializer == "json"
	assert app.conf.result_serializer == "json"
	assert app.conf.accept_content == ["json"]
	assert app.conf.task_acks_late is True
	assert app.conf.worker_prefetch_multiplier == 1
	assert app.conf.task_annotations["*"]["max_retries"] == 3
	assert app.conf.task_annotations["*"]["retry_backoff"] is True
	assert "cleanup-expired-verification-sessions" in app.conf.beat_schedule
	assert app.conf.beat_schedule["cleanup-expired-verification-sessions"]["task"] == "tasks.cleanup_tasks.cleanup_expired_sessions"
	assert "verify-log-ttl-index" in app.conf.beat_schedule
	assert app.conf.beat_schedule["verify-log-ttl-index"]["task"] == "tasks.cleanup_tasks.archive_old_logs"