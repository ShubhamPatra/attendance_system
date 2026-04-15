"""Operational metrics, diagnostics, and event polling routes."""

import re

from flask import jsonify

from app_web.decorators import require_roles


def register_ops_routes(bp):
    @bp.route("/api/metrics")
    @require_roles("admin", "teacher")
    def api_metrics():
        from app_web import routes as routes_module
        from app_vision.face_engine import get_embedding_backend_name
        from app_camera.camera import get_camera_diagnostics

        payload = routes_module.tracker.metrics()
        payload["embedding_backend"] = get_embedding_backend_name()
        payload["camera_diagnostics"] = get_camera_diagnostics()
        return jsonify(payload)

    @bp.route("/api/events")
    @require_roles("admin", "teacher")
    def api_events():
        from app_camera.camera import get_camera_if_running

        cam = get_camera_if_running()
        if cam is None:
            return jsonify([])
        return jsonify(cam.pop_events())

    @bp.route("/api/logs")
    @require_roles("admin")
    def api_logs():
        from app_camera.camera import get_camera_if_running

        cam = get_camera_if_running()
        if cam is None:
            return jsonify([])
        return jsonify(cam.get_log_buffer())

    @bp.route("/api/debug/diagnostics")
    @require_roles("admin")
    def api_debug_diagnostics():
        from app_web import routes as routes_module

        if not routes_module.config.DEBUG_MODE:
            return jsonify({"error": "Diagnostics endpoint is disabled."}), 404

        from app_camera.camera import get_camera_diagnostics

        payload = {
            "cameras": get_camera_diagnostics(),
            "metrics": routes_module.tracker.metrics(),
            "health": {
                "mongo": routes_module._check_mongo_ready(),
                "celery": routes_module._check_celery_ready(),
                **routes_module._check_model_artifacts(),
            },
        }
        return jsonify(payload)

    @bp.route("/api/task/<task_id>")
    @require_roles("admin", "teacher")
    def api_task_status(task_id):
        if not re.fullmatch(r"[0-9a-fA-F-]{8,64}", task_id):
            return jsonify({"error": "Invalid task id format."}), 400

        try:
            from celery_app import celery_app
        except ImportError:
            return jsonify({"error": "Celery is not configured."}), 503

        try:
            task = celery_app.AsyncResult(task_id)
            if task.state == "PENDING":
                return jsonify({"state": "PENDING"})
            if task.state == "FAILURE":
                return jsonify({"state": "FAILURE", "error": str(task.result)})
            return jsonify({"state": task.state, "result": task.result})
        except Exception:
            return jsonify({"error": "Task backend is unavailable."}), 503

    @bp.route("/api/cameras")
    @require_roles("admin", "teacher")
    def api_cameras():
        from app_camera.camera import get_camera_diagnostics

        return jsonify(get_camera_diagnostics())