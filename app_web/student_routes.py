"""Admin-facing student route registrations."""

from __future__ import annotations

import cv2
import os
import tempfile

import numpy as np
from flask import jsonify, render_template, request

from app_core.utils import setup_logging
from app_web.decorators import require_roles


logger = setup_logging()


def register_student_routes(bp):
    @bp.route("/api/at_risk")
    @require_roles("admin", "teacher")
    def api_at_risk():
        from app_web import routes as routes_module

        days = request.args.get("days", 30, type=int)
        threshold = request.args.get("threshold", None, type=int)
        data = routes_module.database.get_at_risk_students(days=days, threshold=threshold)
        for row in data:
            row["attendance_pct"] = row.get("percentage", row.get("attendance_pct", 0))
        return jsonify(data)

    @bp.route("/api/students", methods=["GET"])
    @require_roles("admin", "teacher")
    def api_students_list():
        from app_web import routes as routes_module

        if not routes_module._check_rate_limit("students_list"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        students = routes_module.database.get_all_students()
        for student in students:
            student["_id"] = str(student["_id"])
        return jsonify(students)

    @bp.route("/api/admin/students/pending", methods=["GET"])
    @require_roles("admin")
    def api_admin_students_pending():
        from app_web import routes as routes_module

        students = routes_module.database.get_pending_students()
        for student in students:
            student["_id"] = str(student["_id"])
        return jsonify(students)

    @bp.route("/api/students", methods=["POST"])
    @require_roles("admin", "teacher")
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
    @require_roles("admin", "teacher")
    def api_student_detail(reg_no):
        from app_web import routes as routes_module

        student = routes_module.database.get_student_by_reg_no(reg_no)
        if student is None:
            return routes_module._api_error("Student not found.", 404)

        student["_id"] = str(student["_id"])
        student.pop("encodings", None)
        student.pop("face_encoding", None)
        student.pop("created_at", None)
        student.pop("password_hash", None)
        return jsonify(student)

    @bp.route("/api/students/<reg_no>", methods=["DELETE"])
    @require_roles("admin")
    def api_student_delete(reg_no):
        from app_web import routes as routes_module

        deleted = routes_module.database.delete_student(reg_no)
        if not deleted:
            return routes_module._api_error("Student not found.", 404)

        routes_module.encoding_cache.refresh()
        return jsonify({"deleted": True, "registration_number": reg_no})

    @bp.route("/admin/students")
    @require_roles("admin")
    def admin_students():
        from app_web import routes as routes_module

        students = routes_module.database.get_all_students()
        for student in students:
            student["_id"] = str(student["_id"])
        return render_template("admin_students.html", students=students)

    @bp.route("/api/admin/students", methods=["GET"])
    @require_roles("admin")
    def api_admin_students_list():
        return api_students_list()

    @bp.route("/api/admin/students/<reg_no>", methods=["PATCH"])
    @require_roles("admin")
    def api_admin_students_update(reg_no):
        from app_web import routes as routes_module

        data = request.get_json(silent=True) or {}
        updates = {}
        if "name" in data:
            updates["name"] = routes_module.sanitize_string(str(data.get("name", "")))
        if "section" in data:
            updates["section"] = routes_module.sanitize_string(str(data.get("section", "")))
        if "email" in data:
            updates["email"] = routes_module.sanitize_string(str(data.get("email", ""))) or None
        if "semester" in data:
            try:
                updates["semester"] = int(data.get("semester"))
            except (TypeError, ValueError):
                return routes_module._api_error("semester must be a valid integer.")
        if "is_active" in data:
            updates["is_active"] = bool(data.get("is_active"))

        updated = routes_module.database.update_student(reg_no, updates)
        if not updated:
            return routes_module._api_error("Student not found or nothing to update.", 404)
        return jsonify({"updated": True, "registration_number": reg_no, "changes": updates})

    @bp.route("/api/admin/students/<reg_no>/recompute", methods=["POST"])
    @require_roles("admin")
    def api_admin_students_recompute(reg_no):
        from app_web import routes as routes_module

        payload = request.get_json(silent=True) or {}
        image_paths = list(payload.get("image_paths") or [])
        temp_paths: list[str] = []

        if not image_paths:
            files = request.files.getlist("images")
            for i, f in enumerate(files, 1):
                if f.filename == "":
                    continue
                suffix = os.path.splitext(f.filename)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    f.save(tmp)
                    temp_paths.append(tmp.name)
                    image_paths.append(tmp.name)

        if not image_paths:
            return routes_module._api_error("image_paths or images are required.")

        encodings: list[np.ndarray] = []
        try:
            for i, path in enumerate(image_paths, 1):
                raw_path = str(path)
                resolved = raw_path if raw_path in temp_paths else routes_module._resolve_upload_reference(raw_path)
                if not resolved or not routes_module.os.path.isfile(resolved):
                    return routes_module._api_error(f"Image {i}: invalid file path.")
                img_bgr = cv2.imread(resolved)
                if img_bgr is None:
                    return routes_module._api_error(f"Image {i}: could not read image file.")
                ok, reason = routes_module.check_image_quality(img_bgr)
                if not ok:
                    return routes_module._api_error(f"Image {i}: {reason}")
                encodings.append(routes_module.generate_encoding(resolved))
        except ValueError as exc:
            return routes_module._api_error(str(exc))
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        if not encodings:
            return routes_module._api_error("No valid face encodings could be generated.")

        student = routes_module.database.get_student_by_reg_no(reg_no)
        replaced = routes_module.database.replace_student_encodings(reg_no, encodings)
        if not replaced:
            return routes_module._api_error("Student not found.", 404)

        if student is not None:
            try:
                routes_module.encoding_cache.upsert_student(student["_id"], student.get("name", reg_no), encodings)
            except Exception as exc:
                # Fallback to full refresh if upsert fails; catch broad exceptions since cache operations can fail in many ways
                logger.debug("Cache upsert failed in recompute, falling back to refresh: %s", exc)
                routes_module.encoding_cache.refresh()
        else:
            routes_module.encoding_cache.refresh()
        return jsonify({"updated": True, "registration_number": reg_no, "encodings_count": len(encodings)})

    @bp.route("/api/admin/students/<reg_no>/approve", methods=["POST"])
    @require_roles("admin")
    def api_admin_students_approve(reg_no):
        from app_web import routes as routes_module
        from student_app.verification import evaluate_student_samples

        student = routes_module.database.get_student_by_reg_no(reg_no, include_sensitive=False)
        if student is None:
            return routes_module._api_error("Student not found.", 404)

        samples = list(student.get("face_samples") or [])
        if not samples:
            return routes_module._api_error("No face samples available for approval.")

        result = evaluate_student_samples(samples, registration_number=reg_no)
        if not result.encodings:
            return routes_module._api_error("Unable to generate encodings from stored samples.")

        updated = routes_module.database.update_student_verification(
            reg_no,
            "approved",
            score=result.score,
            reason="Approved by admin review.",
            face_samples=samples,
            encodings=result.encodings,
        )
        if not updated:
            return routes_module._api_error("Student not found.", 404)

        routes_module.encoding_cache.refresh()
        return jsonify({"approved": True, "registration_number": reg_no, "score": result.score})

    @bp.route("/api/admin/students/<reg_no>/reject", methods=["POST"])
    @require_roles("admin")
    def api_admin_students_reject(reg_no):
        from app_web import routes as routes_module

        payload = request.get_json(silent=True) or request.form or {}
        reason = str(payload.get("reason", "Rejected by admin review.")).strip() or "Rejected by admin review."
        updated = routes_module.database.update_student_verification(
            reg_no,
            "rejected",
            reason=reason,
        )
        if not updated:
            return routes_module._api_error("Student not found.", 404)
        return jsonify({"rejected": True, "registration_number": reg_no, "reason": reason})