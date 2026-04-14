"""Student portal and student REST route registrations."""

import cv2

from flask import flash, jsonify, redirect, render_template, request, session, url_for


def register_student_routes(bp):
    @bp.route("/student", methods=["GET", "POST"])
    def student_portal():
        from app_web import routes as routes_module

        if request.method == "POST":
            reg_no = routes_module.sanitize_string(
                request.form.get("registration_number", "").strip(),
            )
            if not reg_no:
                flash("Please enter your registration number.", "danger")
                return render_template("student_login.html"), 400

            student = routes_module.database.get_student_by_reg_no(reg_no)
            if student is None:
                flash("Registration number not found.", "danger")
                return render_template("student_login.html"), 404

            session["student_reg_no"] = reg_no
            return redirect(url_for("main.student_portal"))

        reg_no = session.get("student_reg_no")
        if not reg_no:
            return render_template("student_login.html")

        summary = routes_module.database.get_student_attendance_summary(reg_no)
        if summary is None:
            session.pop("student_reg_no", None)
            flash("Student record not found. Please log in again.", "danger")
            return render_template("student_login.html")

        return render_template("student_portal.html", student=summary)

    @bp.route("/student/logout")
    def student_logout():
        session.pop("student_reg_no", None)
        return redirect(url_for("main.student_portal"))

    @bp.route("/api/at_risk")
    def api_at_risk():
        from app_web import routes as routes_module

        days = request.args.get("days", 30, type=int)
        threshold = request.args.get("threshold", None, type=int)
        data = routes_module.database.get_at_risk_students(days=days, threshold=threshold)
        return jsonify(data)

    @bp.route("/api/students", methods=["GET"])
    def api_students_list():
        from app_web import routes as routes_module

        students = routes_module.database.get_all_students()
        for student in students:
            student["_id"] = str(student["_id"])
        return jsonify(students)

    @bp.route("/api/students", methods=["POST"])
    def api_students_create():
        from app_web import routes as routes_module

        if not routes_module._check_rate_limit("students_create"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        data = request.get_json(silent=True)
        if not data:
            return routes_module._api_error("JSON body required.")

        req_errors, data = routes_module.validate_required_fields(
            data,
            ["name", "semester", "registration_number", "section", "image_paths"],
        )

        name = routes_module.sanitize_string(data.get("name", ""))
        semester_raw = data.get("semester")
        reg_no = routes_module.sanitize_string(data.get("registration_number", ""))
        section = routes_module.sanitize_string(data.get("section", ""))
        image_paths = data.get("image_paths", [])

        errors: list[str] = []
        errors.extend(req_errors)
        if not name:
            errors.append("name is required.")
        if not reg_no:
            errors.append("registration_number is required.")
        if not section:
            errors.append("section is required.")
        try:
            semester = int(semester_raw)
            if semester < 1 or semester > 12:
                errors.append("semester must be between 1 and 12.")
        except (ValueError, TypeError):
            errors.append("semester must be a valid integer.")
            semester = None
        if not image_paths:
            errors.append("image_paths is required (list of file paths).")

        if errors:
            return routes_module._api_errors(errors)

        encodings = []
        for idx, path in enumerate(image_paths, 1):
            resolved = routes_module._resolve_upload_reference(str(path))
            if resolved is None or not routes_module.os.path.isfile(resolved):
                errors.append(f"Image {idx}: invalid file path.")
                continue

            try:
                img_bgr = cv2.imread(resolved)
                if img_bgr is None:
                    errors.append(f"Image {idx}: could not read image file.")
                    continue

                ok, reason = routes_module.check_image_quality(img_bgr)
                if not ok:
                    errors.append(f"Image {idx}: {reason}")
                    continue

                enc = routes_module.generate_encoding(resolved)
                encodings.append(enc)
            except ValueError as exc:
                errors.append(f"Image {idx}: {exc}")

        if errors:
            return routes_module._api_errors(errors)

        if not encodings:
            return routes_module._api_error("No valid face encodings could be generated.")

        try:
            student_id = routes_module.database.insert_student(
                name,
                semester,
                reg_no,
                section,
                encodings,
            )
        except ValueError as exc:
            return routes_module._api_error(str(exc), 409)

        routes_module.encoding_cache.refresh()
        return jsonify(
            {
                "id": str(student_id),
                "name": name,
                "encodings_count": len(encodings),
            }
        ), 201

    @bp.route("/api/students/<reg_no>", methods=["GET"])
    def api_student_detail(reg_no):
        from app_web import routes as routes_module

        student = routes_module.database.get_student_by_reg_no(reg_no)
        if student is None:
            return routes_module._api_error("Student not found.", 404)

        student["_id"] = str(student["_id"])
        student.pop("encodings", None)
        student.pop("face_encoding", None)
        student.pop("created_at", None)
        return jsonify(student)

    @bp.route("/api/students/<reg_no>", methods=["DELETE"])
    def api_student_delete(reg_no):
        from app_web import routes as routes_module

        deleted = routes_module.database.delete_student(reg_no)
        if not deleted:
            return routes_module._api_error("Student not found.", 404)

        routes_module.encoding_cache.refresh()
        return jsonify({"deleted": True, "registration_number": reg_no})