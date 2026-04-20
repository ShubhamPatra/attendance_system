"""Student registration and webcam capture route registrations."""

import csv
import base64
import os
import tempfile
import zipfile
import uuid
from io import StringIO

import cv2
import numpy as np
from flask import flash, jsonify, redirect, render_template, request, url_for

from web.decorators import require_roles


BATCH_COLUMNS = ["registration_number", "name", "semester", "section", "email"]


def _parse_batch_rows(csv_storage):
    raw = csv_storage.read().decode("utf-8-sig")
    reader = csv.DictReader(StringIO(raw))
    if not reader.fieldnames:
        return None, ["CSV file is empty."]
    headers = [header.strip() for header in reader.fieldnames]
    missing = [column for column in BATCH_COLUMNS if column not in headers]
    if missing:
        return None, [f"CSV missing required column(s): {', '.join(missing)}."]

    rows = []
    errors = []
    for index, row in enumerate(reader, 2):
        reg_no = row.get("registration_number", "").strip()
        name = row.get("name", "").strip()
        section = row.get("section", "").strip()
        email = row.get("email", "").strip() or None
        try:
            semester = int(row.get("semester", ""))
        except (TypeError, ValueError):
            errors.append(f"Row {index}: semester must be an integer.")
            continue
        if semester < 1 or semester > 12:
            errors.append(f"Row {index}: semester must be between 1 and 12.")
            continue
        if not reg_no or not name or not section:
            errors.append(f"Row {index}: registration_number, name, and section are required.")
            continue
        rows.append({
            "registration_number": reg_no,
            "name": name,
            "semester": semester,
            "section": section,
            "email": email,
        })
    return rows, errors


def _zip_matches_for_reg_no(zip_file: zipfile.ZipFile, reg_no: str) -> list[str]:
    matches: list[str] = []
    for info in zip_file.infolist():
        if info.is_dir():
            continue
        normalized = info.filename.replace("\\", "/")
        base = os.path.basename(normalized)
        stem, _ = os.path.splitext(base)
        parts = normalized.split("/")
        if (
            parts[0] == reg_no
            or base == reg_no
            or stem == reg_no
            or stem.startswith(f"{reg_no}_")
            or stem.startswith(f"{reg_no}-")
            or normalized.startswith(f"{reg_no}/")
        ):
            matches.append(info.filename)
    return matches


def register_registration_routes(bp):
    @bp.route("/register", methods=["GET", "POST"])
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def register():
        from web import routes as routes_module

        if request.method == "GET":
            return render_template("register.html")

        name = routes_module.sanitize_string(request.form.get("name", ""))
        semester_raw = request.form.get("semester", "")
        reg_no = routes_module.sanitize_string(request.form.get("registration_number", ""))
        section = routes_module.sanitize_string(request.form.get("section", ""))

        errors: list[str] = []
        if not name:
            errors.append("Name is required.")
        if not reg_no:
            errors.append("Registration number is required.")
        if not section:
            errors.append("Section is required.")
        try:
            semester = int(semester_raw)
            if semester < 1 or semester > 12:
                errors.append("Semester must be between 1 and 12.")
        except (ValueError, TypeError):
            errors.append("Semester must be a valid integer.")
            semester = None

        webcam_mode = request.form.get("webcam_mode", "0") == "1"

        if webcam_mode:
            webcam_paths: list[str] = []
            for i in range(5):
                ref = request.form.get(f"webcam_frame_{i}", "").strip()
                if not ref:
                    continue
                path = routes_module._resolve_upload_reference(ref)
                if path is None or not os.path.isfile(path):
                    errors.append(f"Webcam frame {i}: invalid file path.")
                    continue
                webcam_paths.append(path)
            if not webcam_paths:
                errors.append("No webcam frames provided or files not found.")
        else:
            files = request.files.getlist("images")
            if not files or all(f.filename == "" for f in files):
                errors.append("Please upload at least one face image.")
            else:
                if len(files) > routes_module.config.MAX_REGISTRATION_IMAGES:
                    errors.append(
                        f"Maximum {routes_module.config.MAX_REGISTRATION_IMAGES} images allowed."
                    )
                for i, f in enumerate(files, 1):
                    if f.filename == "":
                        continue
                    if not routes_module.allowed_file(f.filename):
                        errors.append(f"Image {i}: only PNG, JPG, JPEG are allowed.")
                        continue
                    detected = routes_module._validate_image_mime(f)
                    if detected is None:
                        errors.append(
                            f"Image {i}: file content does not match a valid image format (PNG or JPEG)."
                        )
                        continue
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(0)
                    if size > routes_module.config.UPLOAD_MAX_SIZE:
                        errors.append(f"Image {i}: must be smaller than 5 MB.")

        if errors:
            for error in errors:
                flash(error, "danger")
            return render_template("register.html"), 400

        encodings: list[np.ndarray] = []
        if webcam_mode:
            for i, path in enumerate(webcam_paths, 1):
                try:
                    img_bgr = cv2.imread(path)
                    if img_bgr is None:
                        flash(f"Webcam frame {i}: could not read image file.", "danger")
                        continue
                    ok, reason = routes_module.check_image_quality(img_bgr)
                    if not ok:
                        flash(f"Webcam frame {i}: {reason}", "danger")
                        continue
                    enc = routes_module.generate_encoding(path)
                    encodings.append(enc)
                except ValueError as exc:
                    flash(f"Webcam frame {i}: {exc}", "danger")
        else:
            for i, f in enumerate(files, 1):
                if f.filename == "":
                    continue
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        f.save(tmp)
                        tmp_path = tmp.name

                    img_bgr = cv2.imread(tmp_path)
                    if img_bgr is None:
                        flash(f"Image {i}: could not read image file.", "danger")
                        continue

                    ok, reason = routes_module.check_image_quality(img_bgr)
                    if not ok:
                        flash(f"Image {i}: {reason}", "danger")
                        continue

                    enc = routes_module.generate_encoding(tmp_path)
                    encodings.append(enc)
                except ValueError as exc:
                    flash(f"Image {i}: {exc}", "danger")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        if not encodings:
            flash(
                "No valid face encodings could be generated. Please upload clear, well-lit images with exactly one face.",
                "danger",
            )
            return render_template("register.html"), 400

        try:
            student_id = routes_module.database.insert_student(name, semester, reg_no, section, encodings)
        except ValueError as exc:
            flash(str(exc), "danger")
            return render_template("register.html"), 409

        try:
            routes_module.encoding_cache.upsert_student(student_id, name, encodings)
        except Exception as exc:
            # Fallback to full refresh if upsert fails; catch broad exceptions since cache operations can fail in many ways
            logger.debug("Cache upsert failed, falling back to refresh: %s", exc)
            routes_module.encoding_cache.refresh()
        flash(
            f"Student '{name}' registered successfully with {len(encodings)} face encoding(s)!",
            "success",
        )
        return redirect(url_for("main.register"))

    @bp.route("/register/batch", methods=["GET", "POST"])
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def register_batch():
        from web import routes as routes_module

        if request.method == "GET":
            return render_template("batch_register.html")

        csv_file = request.files.get("csv_file")
        images_zip = request.files.get("images_zip")
        if csv_file is None or images_zip is None:
            flash("CSV file and ZIP archive are required.", "danger")
            return render_template("batch_register.html"), 400

        rows, errors = _parse_batch_rows(csv_file)
        if errors:
            for error in errors:
                flash(error, "danger")
            return render_template("batch_register.html"), 400

        results = []
        success_count = 0
        failure_count = 0
        with zipfile.ZipFile(images_zip.stream) as archive, tempfile.TemporaryDirectory() as tmpdir:
            for row in rows or []:
                reg_no = row["registration_number"]
                matches = _zip_matches_for_reg_no(archive, reg_no)
                if not matches:
                    failure_count += 1
                    results.append({"registration_number": reg_no, "status": "failed", "error": "No images found in ZIP."})
                    continue

                encodings: list[np.ndarray] = []
                row_errors: list[str] = []
                for index, member in enumerate(matches[:routes_module.config.MAX_REGISTRATION_IMAGES], 1):
                    suffix = os.path.splitext(member)[1] or ".jpg"
                    tmp_path = os.path.join(tmpdir, f"{reg_no}_{index}{suffix}")
                    with open(tmp_path, "wb") as fh:
                        fh.write(archive.read(member))

                    img_bgr = cv2.imread(tmp_path)
                    if img_bgr is None:
                        row_errors.append(f"{member}: could not read image file.")
                        continue
                    ok, reason = routes_module.check_image_quality(img_bgr)
                    if not ok:
                        row_errors.append(f"{member}: {reason}")
                        continue
                    try:
                        encodings.append(routes_module.generate_encoding(tmp_path))
                    except ValueError as exc:
                        row_errors.append(f"{member}: {exc}")

                if not encodings:
                    failure_count += 1
                    results.append({"registration_number": reg_no, "status": "failed", "error": "; ".join(row_errors) or "No valid encodings generated."})
                    continue

                try:
                    student_id = routes_module.database.insert_student(
                        row["name"],
                        row["semester"],
                        reg_no,
                        row["section"],
                        encodings,
                        email=row["email"],
                    )
                    try:
                        routes_module.encoding_cache.upsert_student(student_id, row["name"], encodings)
                    except Exception as exc:
                        # Fallback to full refresh if upsert fails; catch broad exceptions since cache operations can fail in many ways
                        logger.debug("Cache upsert failed for batch, falling back to refresh: %s", exc)
                        routes_module.encoding_cache.refresh()
                    success_count += 1
                    results.append({"registration_number": reg_no, "status": "created", "encodings_count": len(encodings)})
                except ValueError as exc:
                    failure_count += 1
                    results.append({"registration_number": reg_no, "status": "failed", "error": str(exc)})

        if request.path.startswith("/api/"):
            return jsonify({"created": success_count, "failed": failure_count, "results": results}), 201 if success_count else 400

        flash(f"Batch processed: {success_count} created, {failure_count} failed.", "success" if success_count else "danger")
        return render_template("batch_register.html", results=results)

    @bp.route("/api/register/capture", methods=["POST"])
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def api_register_capture():
        from web import routes as routes_module

        if not routes_module._check_rate_limit("register_capture"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        errors, data = routes_module.validate_required_fields(request.get_json(silent=True), ["frame"])
        error_response = routes_module._first_error_response(errors)
        if error_response is not None:
            return error_response

        frame_b64 = data["frame"]
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(frame_b64)
        except (ValueError, TypeError) as exc:
            logger.debug("Base64 decode error: %s", exc)
            return routes_module._api_error("Invalid base64 data.")

        if not raw_bytes.startswith(b"\xff\xd8\xff"):
            return routes_module._api_error("Invalid image data: only JPEG is accepted.")

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        if cv2.imdecode(arr, cv2.IMREAD_COLOR) is None:
            return routes_module._api_error("Invalid image data: JPEG decode failed.")

        os.makedirs(routes_module.config.UPLOAD_DIR, exist_ok=True)
        filename = f"webcam_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(routes_module.config.UPLOAD_DIR, filename)

        with open(filepath, "wb") as fh:
            fh.write(raw_bytes)

        return jsonify({"path": filename})

    @bp.route("/api/register/batch", methods=["POST"])
    @require_roles("admin", "teacher")  # Configurable via AUTH_REQUIRED env var
    def api_register_batch():
        return register_batch()