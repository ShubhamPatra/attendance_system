from __future__ import annotations

import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_file, url_for
from flask_login import login_required

from admin_app.forms import ReportForm
from admin_app.routes.auth import role_required
from tasks.report_tasks import generate_report_task


reports_bp = Blueprint("admin_reports", __name__)


def _reports_dir() -> Path:
	path = Path(current_app.root_path).parent / "reports"
	path.mkdir(parents=True, exist_ok=True)
	return path


@reports_bp.get("/reports")
@login_required
@role_required("super_admin", "admin", "viewer")
def reports_home():
	return render_template("reports/reports.html", form=ReportForm())


@reports_bp.post("/reports/generate")
@login_required
@role_required("super_admin", "admin")
def generate_report():
	form = ReportForm()
	if not form.validate_on_submit():
		return jsonify({"error": "invalid form"}), 400
	filters = {
		"date_from": form.date_from.data,
		"date_to": form.date_to.data,
		"course_id": form.course_id.data,
		"student_id": form.student_id.data,
	}
	report_id = uuid.uuid4().hex
	output_path = _reports_dir() / f"report_{report_id}.csv"
	task = generate_report_task.delay(form.report_type.data, filters, str(output_path))
	return jsonify({"task": task, "report_id": report_id, "download_url": url_for("admin_reports.download_report", report_id=report_id)})


@reports_bp.get("/reports/download/<report_id>")
@login_required
@role_required("super_admin", "admin", "viewer")
def download_report(report_id: str):
	path = _reports_dir() / f"report_{report_id}.csv"
	if not path.exists():
		return jsonify({"error": "report not found"}), 404
	return send_file(path, as_attachment=True, download_name=path.name)
