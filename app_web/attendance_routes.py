"""Attendance REST route registrations."""

from flask import jsonify, request

from core.models import AttendanceDAO, AttendanceSessionDAO, StudentDAO
from app_web.decorators import require_roles

def _build_daos(db_module):
    return (
        AttendanceDAO(db_module),
        AttendanceSessionDAO(db_module),
        StudentDAO(db_module),
    )


def register_attendance_routes(bp):
    def _session_payload(doc: dict | None) -> dict | None:
        if doc is None:
            return None
        return {
            "id": str(doc.get("_id")),
            "course_id": doc.get("course_id"),
            "camera_id": doc.get("camera_id"),
            "start_time": doc.get("start_time").isoformat() if doc.get("start_time") else None,
            "end_time": doc.get("end_time").isoformat() if doc.get("end_time") else None,
            "status": doc.get("status"),
            "last_activity_at": (
                doc.get("last_activity_at").isoformat()
                if doc.get("last_activity_at")
                else None
            ),
        }

    @bp.route("/api/attendance/sessions", methods=["POST"])
    @require_roles("admin", "teacher")
    def api_attendance_session_start():
        from app_web import routes as routes_module
        attendance_dao, session_dao, student_dao = _build_daos(routes_module.database)

        if not routes_module._check_rate_limit("attendance_session_start"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        errors, data = routes_module.validate_required_fields(
            request.get_json(silent=True),
            ["course_id", "camera_id"],
        )
        error_response = routes_module._first_error_response(errors)
        if error_response is not None:
            return error_response

        course_id = routes_module.sanitize_string(str(data["course_id"])).strip()
        camera_id = routes_module.sanitize_string(str(data["camera_id"])).strip()
        if not course_id:
            return routes_module._api_error("course_id is required.")
        if not camera_id:
            return routes_module._api_error("camera_id is required.")

        try:
            session_id = session_dao.create(
                course_id=course_id,
                camera_id=camera_id,
            )
            session = session_dao.get_by_id(session_id)
            return jsonify({"created": True, "session": _session_payload(session)}), 201
        except ValueError as exc:
            return routes_module._api_error(str(exc), 409)
        except RuntimeError as exc:
            return routes_module._api_error(f"Database unavailable: {exc}", 503)

    @bp.route("/api/attendance/sessions/<session_id>/end", methods=["POST"])
    @require_roles("admin", "teacher")
    def api_attendance_session_end(session_id: str):
        from app_web import routes as routes_module
        attendance_dao, session_dao, student_dao = _build_daos(routes_module.database)

        if not routes_module._check_rate_limit("attendance_session_end"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        try:
            ended = session_dao.end(session_id)
        except Exception:
            return routes_module._api_error("Invalid session_id.", 400)

        if not ended:
            return routes_module._api_error("Active session not found.", 404)

        session = session_dao.get_by_id(session_id)
        return jsonify({"ended": True, "session": _session_payload(session)})

    @bp.route("/api/attendance/sessions/active", methods=["GET"])
    @require_roles("admin", "teacher")
    def api_attendance_session_active():
        from app_web import routes as routes_module
        attendance_dao, session_dao, student_dao = _build_daos(routes_module.database)

        if not routes_module._check_rate_limit("attendance_session_active"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        camera_id = routes_module.sanitize_string(
            request.args.get("camera_id", "").strip(),
        )
        if not camera_id:
            return routes_module._api_error("camera_id is required.")

        try:
            session_dao.auto_close_idle()
            session = session_dao.get_active(camera_id)
        except RuntimeError as exc:
            return routes_module._api_error(f"Database unavailable: {exc}", 503)

        return jsonify({"active": session is not None, "session": _session_payload(session)})

    @bp.route("/api/attendance/bulk", methods=["POST"])
    @require_roles("admin", "teacher")
    def api_attendance_bulk():
        from app_web import routes as routes_module
        attendance_dao, session_dao, student_dao = _build_daos(routes_module.database)

        if not routes_module._check_rate_limit("attendance_bulk"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        errors, data = routes_module.validate_required_fields(
            request.get_json(silent=True),
            ["student_ids"],
        )
        error_response = routes_module._first_error_response(errors)
        if error_response is not None:
            return error_response

        reg_nos = data["student_ids"]
        status = data.get("status", "Present")
        if status not in ("Present", "Absent"):
            return routes_module._api_error("status must be 'Present' or 'Absent'.")

        entries = []
        not_found = []
        for reg_no in reg_nos:
            student = student_dao.get_by_registration_number(reg_no)
            if student is None:
                not_found.append(reg_no)
                continue
            entries.append(
                {
                    "student_id": student["_id"],
                    "status": status,
                    "confidence_score": 0.0,
                }
            )

        updated = 0
        if entries:
            try:
                updated = attendance_dao.bulk_upsert(entries)
            except RuntimeError as exc:
                return routes_module._api_error(
                    f"Database unavailable: {exc}",
                    503,
                )

        result = {"updated": updated}
        if not_found:
            result["not_found"] = not_found
        return jsonify(result)

    @bp.route("/api/attendance", methods=["GET"])
    @require_roles("admin", "teacher")
    def api_attendance_list():
        from app_web import routes as routes_module
        attendance_dao, session_dao, student_dao = _build_daos(routes_module.database)

        if not routes_module._check_rate_limit("attendance_list"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        reg_no = request.args.get("reg_no", "").strip()
        start_date = request.args.get("start_date", "").strip()
        end_date = request.args.get("end_date", "").strip()
        date = request.args.get("date", "").strip()

        if reg_no:
            records = attendance_dao.list_by_student(reg_no)
        elif start_date and end_date:
            sd, err1 = routes_module._validate_date_param(start_date, "start_date")
            ed, err2 = routes_module._validate_date_param(end_date, "end_date")
            if err1 or err2:
                return routes_module._api_error(err1 or err2)
            if sd > ed:
                return routes_module._api_error("start_date must be <= end_date.")
            records = attendance_dao.list_by_range(start_date, end_date)
        elif date:
            _, err = routes_module._validate_date_param(date, "date")
            if err:
                return routes_module._api_error(err)
            records = attendance_dao.list(date)
        else:
            records = attendance_dao.list()

        return jsonify(records)

    @bp.route("/api/attendance", methods=["POST"])
    @require_roles("admin", "teacher")
    def api_attendance_mark():
        from app_web import routes as routes_module
        attendance_dao, session_dao, student_dao = _build_daos(routes_module.database)

        if not routes_module._check_rate_limit("attendance_mark"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        errors, data = routes_module.validate_required_fields(
            request.get_json(silent=True),
            ["reg_no"],
        )
        error_response = routes_module._first_error_response(errors)
        if error_response is not None:
            return error_response

        reg_no = routes_module.sanitize_string(data["reg_no"])
        status = data.get("status", "Present")

        if status not in ("Present", "Absent"):
            return routes_module._api_error("status must be 'Present' or 'Absent'.")

        student = student_dao.get_by_registration_number(reg_no)
        if student is None:
            return routes_module._api_error("Student not found.", 404)

        entries = [
            {
                "student_id": student["_id"],
                "status": status,
                "confidence_score": 0.0,
            }
        ]
        try:
            count = attendance_dao.bulk_upsert(entries)
        except RuntimeError as exc:
            return routes_module._api_error(
                f"Database unavailable: {exc}",
                503,
            )

        return jsonify(
            {
                "marked": count > 0,
                "reg_no": reg_no,
                "status": status,
            }
        )