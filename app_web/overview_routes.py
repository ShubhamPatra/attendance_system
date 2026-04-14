"""Overview/dashboard route registrations."""

from flask import jsonify, render_template


def register_overview_routes(bp):
    @bp.route("/dashboard")
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
    def api_registration_numbers():
        from app_web import routes as routes_module

        return jsonify(routes_module.database.get_all_registration_numbers())