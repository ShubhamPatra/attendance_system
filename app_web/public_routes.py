"""Public page and read-only API route registrations."""

from flask import jsonify, redirect, render_template, request

import app_core.config as config
from app_web.decorators import require_roles


def register_public_routes(bp):
    @bp.route("/")
    def index():
        return render_template("index.html")

    @bp.route("/logs")
    @require_roles("admin", "teacher")
    def logs():
        return render_template("logs.html")

    @bp.route("/metrics")
    @require_roles("admin", "teacher")
    def metrics():
        from app_web import routes as routes_module

        m = routes_module.tracker.metrics()
        m["total_students"] = routes_module.database.student_count()
        m["today_count"] = routes_module.database.today_attendance_count()
        return render_template("metrics.html", metrics=m)

    @bp.route("/attendance_activity")
    @require_roles("admin", "teacher")
    def attendance_activity():
        return render_template("attendance_activity.html")

    @bp.route("/api/attendance_activity")
    @require_roles("admin", "teacher")
    def api_attendance_activity():
        from app_web import routes as routes_module

        date = request.args.get("date", routes_module.today_str())
        _, err = routes_module._validate_date_param(date, "date")
        if err:
            return routes_module._api_error(err, 400)
        data = routes_module.database.get_attendance_by_hour(date)
        return jsonify({"date": date, "hours": data})

    @bp.route("/heatmap")
    @require_roles("admin", "teacher")
    def heatmap():
        return render_template("heatmap.html")

    @bp.route("/student")
    @bp.route("/student/")
    def student_portal_redirect():
        return redirect(f"{config.STUDENT_PORTAL_BASE_URL}/login")

    @bp.route("/api/heatmap")
    @require_roles("admin", "teacher")
    def api_heatmap():
        from app_web import routes as routes_module

        data = routes_module.database.get_attendance_heatmap_data()
        return jsonify(data)