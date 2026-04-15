"""Cleanup task compatibility module.

Current project exposes backup/retention cleanup via ``backup_mongodb``.
"""

from celery_app import backup_mongodb

__all__ = ["backup_mongodb"]
