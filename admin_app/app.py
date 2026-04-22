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
from anti_spoofing.model import init_models
import recognition.pipeline as pipeline
import vision.ppe_detection as ppe_detection
from recognition.embedder import encoding_cache, get_arcface_backend
from web.routes import bp as main_bp
from core.utils import setup_logging

socketio = SocketIO()


def _build_swagger_paths(app: Flask) -> dict[str, dict[str, dict]]:
    """Build a minimal Swagger paths object from registered /api routes."""
    paths: dict[str, dict[str, dict]] = {}
    ignored_prefixes = ("/api/docs", "/apispec", "/flasgger_static")

    for rule in app.url_map.iter_rules():
        route = rule.rule
        if not route.startswith("/api/"):
            continue
        if route.startswith(ignored_prefixes):
            continue

        methods = sorted(m for m in rule.methods if m not in {"HEAD", "OPTIONS"})
        if not methods:
            continue

        # Convert Flask path params (<name>, <type:name>) to Swagger style {name}.
        swagger_path = route
        while "<" in swagger_path and ">" in swagger_path:
            start = swagger_path.index("<")
            end = swagger_path.index(">", start)
            segment = swagger_path[start + 1:end]
            param_name = segment.split(":", 1)[-1]
            swagger_path = f"{swagger_path[:start]}{{{param_name}}}{swagger_path[end + 1:]}"

        operation_group = route.split("/")[2] if len(route.split("/")) > 2 else "api"
        path_item = paths.setdefault(swagger_path, {})
        for method in methods:
            path_item[method.lower()] = {
                "summary": rule.endpoint,
                "tags": [operation_group],
                "responses": {
                    "200": {
                        "description": "Success",
                    }
                },
            }

    return paths


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
    warnings: list[str] = []

    if not os.path.isfile(config.YUNET_MODEL_PATH):
        errors.append(f"YuNet model missing: {config.YUNET_MODEL_PATH}")

    # Anti-spoofing is optional; warn but don't fail startup
    if not os.path.isdir(config.ANTI_SPOOF_MODEL_DIR):
        warnings.append(
            f"Anti-spoof model directory not found: {config.ANTI_SPOOF_MODEL_DIR}. "
            f"Anti-spoofing will be disabled (degraded mode)."
        )
    else:
        model_files = [
            name for name in os.listdir(config.ANTI_SPOOF_MODEL_DIR)
            if name.lower().endswith((".pth", ".onnx"))
        ]
        if model_files:
            logger.info(
                f"✓ Anti-spoofing models found: {len(model_files)} model files in {config.ANTI_SPOOF_MODEL_DIR}"
            )
        else:
            warnings.append(
                f"No anti-spoof model files (.pth/.onnx) found in: {config.ANTI_SPOOF_MODEL_DIR}. "
                f"Anti-spoofing will be disabled."
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

    # Log warnings
    for item in warnings:
        logger.warning("Startup warning: %s", item)

    if errors:
        for item in errors:
            logger.error("Startup check failed: %s", item)
        if config.STRICT_STARTUP_CHECKS:
            raise RuntimeError("Startup diagnostics failed. See logs for details.")
    else:
        logger.info("Startup diagnostics passed.")


def create_app() -> Flask:
    """Create and configure the Flask application."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_folder = os.path.join(project_root, "templates")
    static_folder = os.path.join(project_root, "static")
    
    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
        static_url_path="/static"
    )
    app.secret_key = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.UPLOAD_MAX_SIZE
    
    # Session security configuration
    app.config["SESSION_COOKIE_SECURE"] = config.SESSION_COOKIE_SECURE
    app.config["SESSION_COOKIE_HTTPONLY"] = config.SESSION_COOKIE_HTTPONLY
    app.config["SESSION_COOKIE_SAMESITE"] = config.SESSION_COOKIE_SAMESITE
    app.config["PERMANENT_SESSION_LIFETIME"] = config.ADMIN_SESSION_LIFETIME

    socketio.init_app(
        app,
        cors_allowed_origins=config.SOCKETIO_CORS_ORIGINS,
        async_mode="threading",
    )

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
            "Check anti-spoofing model file installation.",
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
    
    # PHASE 2 & 5: Register metrics and anomaly routes
    try:
        from web.metrics_routes import register_metrics_routes
        register_metrics_routes(app)
        logger.info("Metrics routes registered")
    except Exception as e:
        logger.warning("Failed to register metrics routes: %s", e)
    
    try:
        from web.anomaly_routes import register_analytics_routes
        register_analytics_routes(app)
        logger.info("Analytics routes registered")
    except Exception as e:
        logger.warning("Failed to register analytics routes: %s", e)

    # Initialize API documentation after route registration so all
    # endpoints are visible in generated specs.
    if config.ENABLE_RESTX_API:
        from web.api_restx import init_restx_api

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
            },
            "paths": _build_swagger_paths(app),
        }
        Swagger(app, config=swagger_config, template=swagger_template)

    # Favicon redirect
    from flask import redirect, url_for
    @app.route("/favicon.ico")
    def _favicon_redirect():
        return redirect(url_for("static", filename="favicon.svg"))

    # Wire SocketIO to camera module
    from camera.camera import set_socketio
    set_socketio(socketio)

    # Clean up camera on shutdown
    @atexit.register
    def _cleanup():
        from camera.camera import release_camera
        release_camera()
        _cleanup_uploads(logger)

    # Log authentication status
    from core.auth_config import get_auth_status_message
    auth_status = get_auth_status_message()
    logger.warning(auth_status) if "DISABLED" in auth_status else logger.info(auth_status)

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
