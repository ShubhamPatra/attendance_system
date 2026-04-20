"""
Authentication configuration module.

Provides centralized control for enabling/disabling authentication via
the AUTH_REQUIRED environment variable.

Usage:
    from core.auth_config import is_auth_enabled
    
    if is_auth_enabled():
        # Authentication is enforced
    else:
        # Authentication is disabled (development mode)

Environment Variables:
    AUTH_REQUIRED (default: "true")
        - "true" or "yes" or "1": Authentication is enabled (production-safe)
        - "false" or "no" or "0": Authentication is disabled (development only)
        
    APP_ENV (default: None)
        - "production": Some operations will refuse to run (e.g., seed scripts)
        - Any other value: Development mode
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_auth_enabled() -> bool:
    """
    Check if authentication is enabled via the AUTH_REQUIRED environment variable.
    
    Returns:
        True if authentication is enabled (default), False if disabled for local development
        
    Raises:
        None - Returns False on any parsing error (safe default)
    """
    auth_required = os.environ.get("AUTH_REQUIRED", "true").lower().strip()
    
    # Support common boolean representations
    return auth_required not in ("false", "0", "no", "off")


def get_auth_status_message() -> str:
    """
    Get a human-readable message about current authentication status.
    
    Useful for logging on application startup.
    """
    if is_auth_enabled():
        return "✅ Authentication is ENABLED (production mode)"
    else:
        return "⚠️  Authentication is DISABLED (development mode only - set AUTH_REQUIRED=true for production)"


def is_production() -> bool:
    """
    Check if running in production environment.
    
    Returns:
        True if APP_ENV is set to "production", False otherwise
    """
    return os.environ.get("APP_ENV", "").lower() == "production"
