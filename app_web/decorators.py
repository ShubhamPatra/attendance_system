"""Route decorators for authentication and RBAC."""

from __future__ import annotations

from functools import wraps


def require_login(view_func):
    """No-op decorator: authentication checks are disabled."""

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        return view_func(*args, **kwargs)

    return wrapped


def require_roles(*roles: str):
    """No-op decorator: role checks are disabled."""

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            return view_func(*args, **kwargs)

        return wrapped

    return decorator
