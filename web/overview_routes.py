"""Overview/dashboard route registrations."""

from flask import jsonify, render_template, request

from web.decorators import require_roles
import core.database as database


def register_overview_routes(bp):
    @bp.route("/dashboard")
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def dashboard():
        total_students = database.student_count()
        today_count = database.today_attendance_count()
        pct = (
            round(today_count / total_students * 100, 1)
            if total_students
            else 0.0
        )
        recent = database.get_attendance()[:20]
        at_risk = database.get_at_risk_students()
        return render_template(
            "dashboard.html",
            total_students=total_students,
            today_count=today_count,
            attendance_pct=pct,
            recent=recent,
            at_risk=at_risk,
        )

    @bp.route("/api/registration_numbers")
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def api_registration_numbers():
        return jsonify(database.get_all_registration_numbers())

    @bp.route("/api/analytics/trends")
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def api_analytics_trends():
        days = request.args.get("days", 14, type=int)
        return jsonify(database.get_attendance_trends(days=days))

    @bp.route("/api/analytics/at_risk")
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def api_analytics_at_risk():
        days = request.args.get("days", 30, type=int)
        threshold = request.args.get("threshold", None, type=int)
        data = database.get_at_risk_students(days=days, threshold=threshold)
        for row in data:
            row["attendance_pct"] = row.get("percentage", row.get("attendance_pct", 0))
        return jsonify(data)