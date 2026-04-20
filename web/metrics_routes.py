"""
Metrics API endpoints for real-time performance monitoring.

PHASE 2: Provides REST API endpoints for:
- GET /api/metrics — aggregated metrics across all cameras
- GET /api/metrics/camera/<camera_id> — per-camera metrics
- GET /health — basic health check
- GET /api/health/detailed — detailed health including dependencies
"""

from flask import Blueprint, jsonify
import core.metrics as metrics
import core.database as database
from core.utils import setup_logging

logger = setup_logging()

# Create blueprint
metrics_bp = Blueprint("metrics", __name__, url_prefix="/api")


@metrics_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """GET /api/metrics — Return aggregated metrics across all cameras.
    
    Returns JSON with overall FPS, latencies, frame drops, and per-camera breakdowns.
    """
    try:
        aggregated = metrics.get_aggregated_metrics()
        return jsonify({
            "status": "success",
            "timestamp": None,  # Will be added by client if needed
            "data": aggregated,
        }), 200
    except Exception as e:
        logger.error("Failed to get aggregated metrics: %s", e)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@metrics_bp.route("/metrics/camera/<int:camera_id>", methods=["GET"])
def get_camera_metrics(camera_id):
    """GET /api/metrics/camera/<camera_id> — Return metrics for a specific camera.
    
    Parameters:
    - camera_id: integer camera ID
    
    Returns JSON with FPS, latencies, queue depth, recognition stats for that camera.
    """
    try:
        all_snapshots = metrics.get_all_snapshots()
        if camera_id not in all_snapshots:
            return jsonify({
                "status": "error",
                "message": f"Camera {camera_id} not found or no metrics available",
            }), 404

        return jsonify({
            "status": "success",
            "data": all_snapshots[camera_id],
        }), 200
    except Exception as e:
        logger.error("Failed to get metrics for camera %d: %s", camera_id, e)
        return jsonify({
            "status": "error",
            "message": str(e),
        }), 500


@metrics_bp.route("/health", methods=["GET"])
def health_check():
    """GET /api/health — Basic health check (liveness probe).
    
    Returns 200 if system is running, regardless of errors.
    Useful for Kubernetes/Docker health checks.
    """
    return jsonify({
        "status": "healthy",
        "service": "attendance_system",
    }), 200


@metrics_bp.route("/health/detailed", methods=["GET"])
def detailed_health_check():
    """GET /api/health/detailed — Detailed health check including dependencies.
    
    Returns health status of:
    - Main service
    - MongoDB connectivity
    - Camera pipeline
    - Model loading state
    """
    health_status = {
        "status": "healthy",
        "service": "attendance_system",
        "components": {
            "database": {
                "status": "unknown",
                "error": None,
            },
            "camera_pipeline": {
                "status": "unknown",
                "active_cameras": 0,
                "error": None,
            },
        },
    }

    # Check database connectivity
    try:
        # Try a simple ping operation
        client = database.get_db_client()
        if client is not None:
            try:
                # Ping the server
                client.admin.command('ping')
                health_status["components"]["database"]["status"] = "healthy"
            except Exception as e:
                health_status["components"]["database"]["status"] = "unhealthy"
                health_status["components"]["database"]["error"] = str(e)
                health_status["status"] = "degraded"
        else:
            health_status["components"]["database"]["status"] = "unavailable"
            health_status["status"] = "degraded"
    except Exception as e:
        logger.warning("Failed to check database health: %s", e)
        health_status["components"]["database"]["status"] = "unavailable"
        health_status["components"]["database"]["error"] = str(e)
        health_status["status"] = "degraded"

    # Check camera pipeline
    try:
        from camera.camera import get_all_cameras
        active_cameras = len(get_all_cameras())
        health_status["components"]["camera_pipeline"]["active_cameras"] = active_cameras
        health_status["components"]["camera_pipeline"]["status"] = "healthy" if active_cameras >= 0 else "degraded"
    except Exception as e:
        logger.warning("Failed to check camera pipeline health: %s", e)
        health_status["components"]["camera_pipeline"]["status"] = "unavailable"
        health_status["components"]["camera_pipeline"]["error"] = str(e)

    # Determine HTTP status code based on overall health
    http_status = 200 if health_status["status"] == "healthy" else 503

    return jsonify(health_status), http_status


# Optional: Readiness probe (for Kubernetes)
@metrics_bp.route("/readiness", methods=["GET"])
def readiness_check():
    """GET /api/readiness — Readiness probe for deployment orchestration.
    
    Returns 200 only when all critical dependencies are ready.
    Returns 503 if any critical service is unavailable.
    """
    # Check database readiness
    try:
        client = database.get_db_client()
        if client is None:
            return jsonify({"status": "not_ready", "reason": "Database not initialized"}), 503
        client.admin.command('ping')
    except Exception as e:
        logger.warning("Readiness check: database not ready: %s", e)
        return jsonify({"status": "not_ready", "reason": f"Database error: {str(e)}"}), 503

    return jsonify({"status": "ready"}), 200


def register_metrics_routes(app):
    """Register metrics routes with Flask app.
    
    Usage in app.py:
        from web.metrics_routes import register_metrics_routes
        register_metrics_routes(app)
    """
    app.register_blueprint(metrics_bp)
    logger.info("Metrics routes registered: /api/metrics, /api/metrics/camera/<id>, /health, /api/health/detailed")
