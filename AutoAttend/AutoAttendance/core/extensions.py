"""Shared Flask extension instances used by admin and student apps."""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_socketio import SocketIO


login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to continue."

socketio = SocketIO(cors_allowed_origins="*", async_mode="eventlet")

limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
