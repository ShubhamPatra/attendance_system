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

import anti_spoofing
import config
import database
import pipeline
from face_engine import encoding_cache
from routes import bp as main_bp
from utils import setup_logging

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

    # Database indexes
    database.ensure_indexes()

    # Warm encoding cache
    encoding_cache.load()

    # Load anti-spoofing models once at startup
    anti_spoofing.init_models()

    # Load YuNet face detector
    if os.path.isfile(config.YUNET_MODEL_PATH):
        pipeline.init_yunet(
            config.YUNET_MODEL_PATH,
            config.FRAME_PROCESS_WIDTH,
        )
    else:
        logger.error(
            "YuNet model not found at %s.  Download it from "
            "https://github.com/opencv/opencv_zoo/tree/main/models/"
            "face_detection_yunet and place the .onnx file in "
            "the models/ directory.",
            config.YUNET_MODEL_PATH,
        )

    # Register blueprint
    app.register_blueprint(main_bp)

    # Wire SocketIO to camera module
    from camera import set_socketio
    set_socketio(socketio)

    # Clean up camera on shutdown
    @atexit.register
    def _cleanup():
        from camera import release_camera
        release_camera()
        _cleanup_uploads(logger)

    logger.info("Application initialised.")
    return app


if __name__ == "__main__":
    app = create_app()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=False)
