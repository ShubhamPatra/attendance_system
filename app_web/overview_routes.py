"""Overview/dashboard route registrations."""

from flask import jsonify, render_template, request

from app_web.decorators import require_roles


def register_overview_routes(bp):
    @bp.route("/dashboard")
    @require_roles("admin", "teacher")
    def dashboard():
        from app_web import routes as routes_module

        total_students = routes_module.database.student_count()
        today_count = routes_module.database.today_attendance_count()
        pct = (
            round(today_count / total_students * 100, 1)
            if total_students
            else 0.0
        )
        recent = routes_module.database.get_attendance()[:20]
        at_risk = routes_module.database.get_at_risk_students()
        return render_template(
            "dashboard.html",
            total_students=total_students,
            today_count=today_count,
            attendance_pct=pct,
            recent=recent,
            at_risk=at_risk,
        )

    @bp.route("/api/registration_numbers")
    @require_roles("admin", "teacher")
    def api_registration_numbers():
        from app_web import routes as routes_module

        if not routes_module._check_rate_limit("registration_numbers"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        return jsonify(routes_module.database.get_all_registration_numbers())

    @bp.route("/api/analytics/trends")
    @require_roles("admin", "teacher")
    def api_analytics_trends():
        from app_web import routes as routes_module

        if not routes_module._check_rate_limit("analytics_trends"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        days = request.args.get("days", 14, type=int)
        return jsonify(routes_module.database.get_attendance_trends(days=days))

    @bp.route("/api/analytics/at_risk")
    @require_roles("admin", "teacher")
    def api_analytics_at_risk():
        from app_web import routes as routes_module

        if not routes_module._check_rate_limit("analytics_at_risk"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        days = request.args.get("days", 30, type=int)
        threshold = request.args.get("threshold", None, type=int)
        data = routes_module.database.get_at_risk_students(days=days, threshold=threshold)
        for row in data:
            row["attendance_pct"] = row.get("percentage", row.get("attendance_pct", 0))
        return jsonify(data)