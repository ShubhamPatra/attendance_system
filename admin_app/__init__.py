"""Admin app compatibility entrypoint for migrated package layout.

Import lazily to avoid circular imports when ``app`` imports admin routes.
"""


def create_app(*args, **kwargs):
	from admin_app.app import create_app as _create_app

	return _create_app(*args, **kwargs)


def get_socketio():
	from admin_app.app import socketio as _socketio

	return _socketio


__all__ = ["create_app", "get_socketio"]
