"""Core package compatibility layer for migrated project structure."""

from . import auth, config, database, models, profiling, utils

__all__ = ["auth", "config", "database", "models", "profiling", "utils"]
