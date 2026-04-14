"""
Flask application factory and entry point.
"""

import atexit
import os
import sys
import time

# Ensure the package directory is on sys.path so local imports work
# when running `python app.py` from inside attendance_system/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_socketio import SocketIO
from flasgger import Swagger

import app_core.config as config
import app_core.database as database
import app_vision.anti_spoofing as anti_spoofing
import app_vision.pipeline as pipeline
import app_vision.ppe_detection as ppe_detection
from app_vision.face_engine import encoding_cache
from app_web.routes import bp as main_bp
from app_core.utils import setup_logging

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
        errors.append(f"MongoDB ping failed: {exc}")

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

    socketio.init_app(
        app,
        cors_allowed_origins=config.SOCKETIO_CORS_ORIGINS,
        async_mode="threading",
    )

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

    # Load anti-spoofing models once at startup
    anti_spoofing.init_models()

    # Load PPE model (optional; controlled by config flag)
    ppe_detection.init_model(config.PPE_MODEL_PATH)

    # Load YuNet face detector
    pipeline.init_yunet(
        config.YUNET_MODEL_PATH,
        config.FRAME_PROCESS_WIDTH,
    )

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
