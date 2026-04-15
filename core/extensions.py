"""Runtime extension access helpers for migrated package layout."""

from __future__ import annotations

from flask_socketio import SocketIO


def get_socketio() -> SocketIO:
    """Return the singleton SocketIO instance from the admin app."""
    from app import socketio

    return socketio
