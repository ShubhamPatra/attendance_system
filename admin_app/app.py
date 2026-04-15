"""
Flask application factory and entry point.
"""

import atexit
import os
import sys
import time
from urllib.parse import urlparse

# Ensure the package directory is on sys.path so local imports work
# when running `python app.py` from inside attendance_system/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_socketio import SocketIO
from flasgger import Swagger

import core.config as config
import core.database as database
from core.auth import init_auth
from anti_spoofing.model import init_models
import recognition.pipeline as pipeline
import app_vision.ppe_detection as ppe_detection
from recognition.embedder import encoding_cache, get_arcface_backend
from app_web.routes import bp as main_bp
from core.utils import setup_logging

socketio = SocketIO()


def _cleanup_uploads(logger):
    """Delete stale uploaded capture files older than retention window."""
    cutoff = time.time() - config.UPLOAD_RETENTION_SECONDS
    try:
        for name in os.listdir(config.UPLOAD_DIR):
            path = os.path.join(config.UPLOAD_DIR, name)
            if not os.path.isfile(path):
                continue
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning("Upload cleanup failed: %s", exc)


def _startup_diagnostics(logger):
    """Run startup checks so deployments fail fast when misconfigured."""
    errors: list[str] = []

    if not os.path.isfile(config.YUNET_MODEL_PATH):
        errors.append(f"YuNet model missing: {config.YUNET_MODEL_PATH}")

    if not os.path.isdir(config.ANTI_SPOOF_MODEL_DIR):
        errors.append(f"Anti-spoof model directory missing: {config.ANTI_SPOOF_MODEL_DIR}")
    else:
        pth_files = [
            name for name in os.listdir(config.ANTI_SPOOF_MODEL_DIR)
            if name.lower().endswith(".pth")
        ]
        if not pth_files:
            errors.append(
                f"No anti-spoof .pth model files found in: {config.ANTI_SPOOF_MODEL_DIR}"
            )

    if config.PPE_DETECTION_ENABLED and not os.path.isfile(config.PPE_MODEL_PATH):
        errors.append(f"PPE model missing: {config.PPE_MODEL_PATH}")

    # MongoDB is required for this app.
    try:
        database.get_client()
    except Exception as exc:
        mongo_uri = (config.MONGO_URI or "").strip()
        host_hint = "configured MongoDB host"
        try:
            parsed = urlparse(mongo_uri)
            host_hint = parsed.netloc or parsed.path or host_hint
        except Exception:
            pass
        errors.append(
            "MongoDB ping failed: "
            f"{exc} | target={host_hint}. "
            "Check MONGO_URI in .env and clear conflicting shell/system MONGO_URI values."
        )

    # Celery broker check (filesystem by default; can be overridden by env).
    try:
        broker = (config.CELERY_BROKER_URL or "").strip().lower()
        if broker.startswith("filesystem://"):
            os.makedirs(config.CELERY_DATA_DIR, exist_ok=True)
            test_file = os.path.join(config.CELERY_DATA_DIR, ".healthcheck")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_file)
    except Exception as exc:
        logger.warning("Celery broker check failed at startup: %s", exc)

    if config.STARTUP_CAMERA_PROBE:
        try:
            import cv2

            cap = cv2.VideoCapture(config.CAMERA_HEALTHCHECK_INDEX)
            try:
                if not cap.isOpened():
                    errors.append(
                        "Camera probe failed: could not open camera index "
                        f"{config.CAMERA_HEALTHCHECK_INDEX}"
                    )
            finally:
                cap.release()
        except Exception as exc:
            errors.append(f"Camera probe failed: {exc}")

    if errors:
        for item in errors:
            logger.error("Startup check failed: %s", item)
        if config.STRICT_STARTUP_CHECKS:
            raise RuntimeError("Startup diagnostics failed. See logs for details.")
    else:
        logger.info("Startup diagnostics passed.")


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.UPLOAD_MAX_SIZE
    app.config["SESSION_COOKIE_SECURE"] = config.SESSION_COOKIE_SECURE
    app.config["SESSION_COOKIE_HTTPONLY"] = config.SESSION_COOKIE_HTTPONLY
    app.config["SESSION_COOKIE_SAMESITE"] = config.SESSION_COOKIE_SAMESITE
    app.config["REMEMBER_COOKIE_SECURE"] = config.REMEMBER_COOKIE_SECURE
    app.config["REMEMBER_COOKIE_HTTPONLY"] = config.REMEMBER_COOKIE_HTTPONLY
    app.config["REMEMBER_COOKIE_SAMESITE"] = config.REMEMBER_COOKIE_SAMESITE

    init_auth(app)

    socketio.init_app(
        app,
        cors_allowed_origins=config.SOCKETIO_CORS_ORIGINS,
        async_mode="threading",
    )

    if config.ENABLE_RESTX_API:
        from app_web.api_restx import init_restx_api

        init_restx_api(app)
    else:
        swagger_config = {
            "headers": [],
            "specs": [
                {
                    "endpoint": "apispec",
                    "route": "/apispec.json",
                    "rule_filter": lambda rule: True,
                    "model_filter": lambda tag: True,
                }
            ],
            "static_url_path": "/flasgger_static",
            "swagger_ui": True,
            "specs_route": "/api/docs",
        }
        swagger_template = {
            "info": {
                "title": "AutoAttendance API",
                "description": "REST API for the AutoAttendance face recognition system",
                "version": "2.0.0",
            }
        }
        Swagger(app, config=swagger_config, template=swagger_template)

    logger = setup_logging()

    # Ensure required directories exist
    os.makedirs(config.UNKNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    _cleanup_uploads(logger)

    _startup_diagnostics(logger)

    # Database indexes
    database.ensure_indexes()

    # Warm encoding cache
    encoding_cache.load()

    # Preload ArcFace model at startup (avoids cold-start lag on first recognition)
    if config.EMBEDDING_BACKEND == "arcface":
        try:
            get_arcface_backend().preload()
        except Exception as exc:
            logger.warning("ArcFace preload failed (will lazy-load later): %s", exc)

    # Load anti-spoofing models once at startup
    try:
        init_models()
    except Exception as exc:
        logger.error(
            "Anti-spoofing model initialization failed at startup: %s. "
            "App will continue with degraded anti-spoofing (all faces marked real). "
            "Check model files and Silent-Face-Anti-Spoofing library installation.",
            exc,
        )

    # Load PPE model (optional; controlled by config flag)
    try:
        ppe_detection.init_model(config.PPE_MODEL_PATH)
    except Exception as exc:
        logger.warning("PPE model initialization failed (will be disabled): %s", exc)

    # Load YuNet face detector
    try:
        pipeline.init_yunet(
            config.YUNET_MODEL_PATH,
            config.FRAME_PROCESS_WIDTH,
        )
    except Exception as exc:
        logger.error("YuNet face detector initialization failed: %s", exc)
        raise  # This is critical; app cannot continue

    # Register blueprint
    app.register_blueprint(main_bp)

    # Wire SocketIO to camera module
    from app_camera.camera import set_socketio
    set_socketio(socketio)

    # Clean up camera on shutdown
    @atexit.register
    def _cleanup():
        from app_camera.camera import release_camera
        release_camera()
        _cleanup_uploads(logger)

    logger.info("Application initialised.")
    return app


if __name__ == "__main__":
    app = create_app()
    socketio.run(
        app,
        host=config.APP_HOST,
        port=config.APP_PORT,
        debug=config.APP_DEBUG,
        use_reloader=False,
    )
