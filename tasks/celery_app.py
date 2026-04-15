"""Backward-compatible Celery entrypoint for grouped tasks module."""

from tasks.celery_app import *  # noqa: F401,F403
