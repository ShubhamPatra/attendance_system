"""Operational metrics, diagnostics, and event polling routes."""

from flask import jsonify


def register_ops_routes(bp):
    @bp.route("/api/metrics")
    def api_metrics():
        from app_web import routes as routes_module

        return jsonify(routes_module.tracker.metrics())

    @bp.route("/api/events")
    def api_events():
        from app_camera.camera import get_camera_if_running

        cam = get_camera_if_running()
        if cam is None:
            return jsonify([])
        return jsonify(cam.pop_events())

    @bp.route("/api/logs")
    def api_logs():
        from app_camera.camera import get_camera_if_running

        cam = get_camera_if_running()
        if cam is None:
            return jsonify([])
        return jsonify(cam.get_log_buffer())

    @bp.route("/api/debug/diagnostics")
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
    def api_task_status(task_id):
        try:
            from celery_app import celery_app
        except ImportError:
            return jsonify({"error": "Celery is not configured."}), 503

        task = celery_app.AsyncResult(task_id)
        if task.state == "PENDING":
            return jsonify({"state": "PENDING"})
        if task.state == "FAILURE":
            return jsonify({"state": "FAILURE", "error": str(task.result)})
        return jsonify({"state": task.state, "result": task.result})