"""Health and readiness checks for the main web blueprint."""

import os

from flask import jsonify

import app_core.config as config
import app_core.database as database
from app_core.utils import setup_logging


logger = setup_logging()


def _check_model_artifacts() -> dict[str, bool]:
    anti_spoof_ok = False
    if os.path.isdir(config.ANTI_SPOOF_MODEL_DIR):
        anti_spoof_ok = any(
            name.lower().endswith(".pth")
            for name in os.listdir(config.ANTI_SPOOF_MODEL_DIR)
        )

    ppe_ok = True
    if config.PPE_DETECTION_ENABLED:
        ppe_ok = os.path.isfile(config.PPE_MODEL_PATH)

    return {
        "yunet_model": os.path.isfile(config.YUNET_MODEL_PATH),
        "anti_spoof_models": anti_spoof_ok,
        "ppe_model": ppe_ok,
    }


def _check_mongo_ready() -> bool:
    try:
        database.get_client().admin.command("ping")
        return True
    except Exception as exc:
        logger.debug("MongoDB health check failed: %s", exc)
        return False


def _check_celery_ready() -> bool:
    try:
        broker = (config.CELERY_BROKER_URL or "").strip().lower()
        if broker.startswith("filesystem://"):
            os.makedirs(config.CELERY_DATA_DIR, exist_ok=True)
            probe = os.path.join(config.CELERY_DATA_DIR, ".healthcheck")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return True

        # Non-filesystem brokers are considered configured when set.
        return bool(broker)
    except (OSError, IOError) as exc:
        logger.debug("Celery health check failed: %s", exc)
        return False


def register_health_routes(bp):
    @bp.route("/health")
    def health():
        """Simple liveness endpoint for process-level checks."""
        return jsonify({"status": "ok", "service": "autoattendance"})

    @bp.route("/ready")
    @bp.route("/healthz")
    def ready():
        """Readiness endpoint for orchestration and probes."""
        # Resolve through the public routes shim so unit tests can patch
        # routes._check_* without changing endpoint behavior.
        from app_web import routes as routes_module

        checks = {
            "mongo": routes_module._check_mongo_ready(),
            "celery": routes_module._check_celery_ready(),
            **routes_module._check_model_artifacts(),
        }

        critical_ok = (
            checks["mongo"]
            and checks["yunet_model"]
            and checks["anti_spoof_models"]
        )
        status_code = 200 if critical_ok else 503
        payload = {
            "status": "ready" if critical_ok else "degraded",
            "checks": checks,
        }
        return jsonify(payload), status_code