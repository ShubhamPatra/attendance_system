"""Notification dry-run route registrations."""

from flask import jsonify, request

from app_core.notifications import record_notification_event
from app_web.decorators import require_roles


def register_notification_routes(bp):
    @bp.route("/api/notifications/dry-run", methods=["GET", "POST"])
    @require_roles("admin", "teacher")
    def api_notifications_dry_run():
        from app_web import routes as routes_module

        threshold = request.args.get("threshold", None, type=int)
        at_risk = routes_module.database.get_at_risk_students(threshold=threshold)
        today_records = routes_module.database.get_attendance(routes_module.today_str())
        present_regs = {row["registration_number"] for row in today_records if row.get("status") == "Present"}
        all_students = routes_module.database.get_all_students()

        events = []
        for student in all_students:
            reg_no = student.get("registration_number")
            if reg_no not in present_regs:
                payload = {
                    "registration_number": reg_no,
                    "name": student.get("name"),
                    "reason": "absent_today",
                    "mode": "dry_run",
                }
                record_notification_event(
                    "absence_alert",
                    student.get("email") or reg_no,
                    f"Absence alert for {student.get('name')}",
                    payload,
                )
                events.append(payload)

        for student in at_risk:
            payload = {
                "registration_number": student.get("reg_no"),
                "name": student.get("name"),
                "reason": "low_attendance",
                "attendance_pct": student.get("percentage", student.get("attendance_pct", 0)),
                "mode": "dry_run",
            }
            record_notification_event(
                "low_attendance_alert",
                student.get("reg_no"),
                f"Low attendance alert for {student.get('name')}",
                payload,
            )
            events.append(payload)

        return jsonify({"mode": "dry_run", "generated": len(events), "events": events})
