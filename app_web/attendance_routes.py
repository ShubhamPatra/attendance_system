"""Attendance REST route registrations."""

from flask import jsonify, request


def register_attendance_routes(bp):
    @bp.route("/api/attendance/bulk", methods=["POST"])
    def api_attendance_bulk():
        from app_web import routes as routes_module

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
            student = routes_module.database.get_student_by_reg_no(reg_no)
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
            updated = routes_module.database.bulk_upsert_attendance(entries)

        result = {"updated": updated}
        if not_found:
            result["not_found"] = not_found
        return jsonify(result)

    @bp.route("/api/attendance", methods=["GET"])
    def api_attendance_list():
        from app_web import routes as routes_module

        reg_no = request.args.get("reg_no", "").strip()
        start_date = request.args.get("start_date", "").strip()
        end_date = request.args.get("end_date", "").strip()
        date = request.args.get("date", "").strip()

        if reg_no:
            records = routes_module.database.get_attendance_by_student(reg_no)
        elif start_date and end_date:
            sd, err1 = routes_module._validate_date_param(start_date, "start_date")
            ed, err2 = routes_module._validate_date_param(end_date, "end_date")
            if err1 or err2:
                return routes_module._api_error(err1 or err2)
            if sd > ed:
                return routes_module._api_error("start_date must be <= end_date.")
            records = routes_module.database.get_attendance_by_date_range(start_date, end_date)
        elif date:
            _, err = routes_module._validate_date_param(date, "date")
            if err:
                return routes_module._api_error(err)
            records = routes_module.database.get_attendance(date)
        else:
            records = routes_module.database.get_attendance()

        return jsonify(records)

    @bp.route("/api/attendance", methods=["POST"])
    def api_attendance_mark():
        from app_web import routes as routes_module

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

        student = routes_module.database.get_student_by_reg_no(reg_no)
        if student is None:
            return routes_module._api_error("Student not found.", 404)

        entries = [
            {
                "student_id": student["_id"],
                "status": status,
                "confidence_score": 0.0,
            }
        ]
        count = routes_module.database.bulk_upsert_attendance(entries)

        return jsonify(
            {
                "marked": count > 0,
                "reg_no": reg_no,
                "status": status,
            }
        )