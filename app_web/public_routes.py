"""Public page and read-only API route registrations."""

from flask import jsonify, render_template, request


def register_public_routes(bp):
    @bp.route("/")
    def index():
        return render_template("index.html")

    @bp.route("/logs")
    def logs():
        return render_template("logs.html")

    @bp.route("/metrics")
    def metrics():
        from app_web import routes as routes_module

        m = routes_module.tracker.metrics()
        m["total_students"] = routes_module.database.student_count()
        m["today_count"] = routes_module.database.today_attendance_count()
        return render_template("metrics.html", metrics=m)

    @bp.route("/attendance_activity")
    def attendance_activity():
        return render_template("attendance_activity.html")

    @bp.route("/api/attendance_activity")
    def api_attendance_activity():
        from app_web import routes as routes_module

        date = request.args.get("date", routes_module.today_str())
        data = routes_module.database.get_attendance_by_hour(date)
        return jsonify({"date": date, "hours": data})

    @bp.route("/heatmap")
    def heatmap():
        return render_template("heatmap.html")

    @bp.route("/api/heatmap")
    def api_heatmap():
        from app_web import routes as routes_module

        data = routes_module.database.get_attendance_heatmap_data()
        return jsonify(data)