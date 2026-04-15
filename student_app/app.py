"""Flask application factory for the student portal."""

from __future__ import annotations

import atexit
import os
import sys

# Ensure project root is importable when running `python app.py` inside student_app/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, redirect, url_for

import core.config as config
import core.database as database
import vision.anti_spoofing as anti_spoofing
import vision.pipeline as pipeline
from core.utils import setup_logging

try:
    from .auth import init_auth
    from .routes import register_routes
except ImportError:
    # Support `python app.py` execution inside student_app/.
    from student_app.auth import init_auth
    from student_app.routes import register_routes


logger = setup_logging()


def create_app() -> Flask:
    """Create the standalone student portal application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/student/static",
    )
    app.secret_key = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.UPLOAD_MAX_SIZE
    app.config["SESSION_COOKIE_SECURE"] = config.SESSION_COOKIE_SECURE
    app.config["SESSION_COOKIE_HTTPONLY"] = config.SESSION_COOKIE_HTTPONLY
    app.config["SESSION_COOKIE_SAMESITE"] = config.SESSION_COOKIE_SAMESITE

    init_auth(app)

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.STUDENT_SAMPLE_DIR, exist_ok=True)

    database.ensure_indexes()
    
    try:
        anti_spoofing.init_models()
    except Exception as exc:
        logger.error(
            "Anti-spoofing model initialization failed at startup: %s. "
            "App will continue with degraded anti-spoofing (all faces marked real).",
            exc,
        )
    
    try:
        pipeline.init_yunet(config.YUNET_MODEL_PATH, config.FRAME_PROCESS_WIDTH)
    except Exception as exc:
        logger.error("YuNet face detector initialization failed: %s", exc)
        raise  # This is critical; app cannot continue

    register_routes(app)

    @app.route("/")
    def _student_root_redirect():
        return redirect("/student/login")

    @app.route("/login")
    def _student_login_redirect():
        return redirect("/student/login")

    @app.route("/register")
    def _student_register_redirect():
        return redirect("/student/register")

    @app.route("/capture")
    def _student_capture_redirect():
        return redirect("/student/capture")

    @app.route("/status")
    def _student_status_redirect():
        return redirect("/student/status")

    @app.route("/attendance")
    def _student_attendance_redirect():
        return redirect("/student/attendance")

    @app.route("/favicon.ico")
    def _student_favicon_redirect():
        return redirect(url_for("static", filename="favicon.svg"))

    @atexit.register
    def _cleanup() -> None:
        return None

    logger.info("Student portal application initialised.")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.STUDENT_APP_HOST, port=config.STUDENT_APP_PORT, debug=config.APP_DEBUG)
