"""Route decorators for authentication and RBAC."""

from __future__ import annotations

import logging
from functools import wraps

from flask import redirect, url_for, jsonify, request
from flask_login import current_user

from core.auth_config import is_auth_enabled

logger = logging.getLogger(__name__)


def require_login(view_func):
    """
    Decorator to require user login for admin routes.
    
    Redirects unauthenticated users to login page.
    
    Can be disabled by setting AUTH_REQUIRED=false (development only).
    """
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        # Skip authentication check if auth is disabled (development mode)
        if not is_auth_enabled():
            logger.debug(f"Skipping auth check for {request.path} (AUTH_REQUIRED=false)")
            return view_func(*args, **kwargs)
        
        # Check if user is authenticated
        try:
            is_authenticated = current_user.is_authenticated
        except (AttributeError, RuntimeError) as e:
            # Flask-Login not properly initialized; log warning and allow access
            logger.warning(f"Flask-Login not initialized; allowing access to {request.path}: {e}")
            return view_func(*args, **kwargs)
        
        if not is_authenticated:
            # Return JSON for API requests, redirect for HTML
            if request.headers.get('Accept') == 'application/json' or request.path.startswith('/api/'):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('admin.login'))
        
        return view_func(*args, **kwargs)
    
    return wrapped


def require_roles(*roles: str):
    """
    Decorator to require specific user roles.
    
    Can be disabled by setting AUTH_REQUIRED=false (development only).
    
    Args:
        *roles: One or more role strings (e.g., 'admin', 'teacher')
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            # Skip authentication check if auth is disabled (development mode)
            if not is_auth_enabled():
                logger.debug(f"Skipping auth check for {request.path} (AUTH_REQUIRED=false)")
                return view_func(*args, **kwargs)
            
            # Check if user is authenticated
            try:
                is_authenticated = current_user.is_authenticated
            except (AttributeError, RuntimeError) as e:
                # Flask-Login not properly initialized; log warning and allow access
                logger.warning(f"Flask-Login not initialized; allowing access to {request.path}: {e}")
                return view_func(*args, **kwargs)
            
            if not is_authenticated:
                if request.headers.get('Accept') == 'application/json' or request.path.startswith('/api/'):
                    return jsonify({'error': 'Authentication required'}), 401
                return redirect(url_for('admin.login'))
            
            # Check if user has required role
            user_role = getattr(current_user, 'role', None)
            if user_role not in roles:
                if request.headers.get('Accept') == 'application/json' or request.path.startswith('/api/'):
                    return jsonify({'error': f'Insufficient permissions. Required roles: {", ".join(roles)}'}), 403
                return redirect(url_for('admin.dashboard'))
            
            return view_func(*args, **kwargs)
        
        return wrapped
    
    return decorator
