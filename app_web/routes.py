"""
Flask routes blueprint -- all web endpoints.
"""

import os

from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
    url_for,
)

import core.config as config
import core.database as database
from core.auth import authenticate_user
from app_web.health_routes import (
  _check_celery_ready,
  _check_model_artifacts,
  _check_mongo_ready,
  register_health_routes,
)
from app_web.auth_routes import register_auth_routes
from app_web.attendance_routes import register_attendance_routes
from app_web.camera_routes import register_camera_routes
from app_web.overview_routes import register_overview_routes
from app_web.public_routes import register_public_routes
from app_web.ops_routes import register_ops_routes
from app_web.notification_routes import register_notification_routes
from app_web.registration_routes import register_registration_routes
from app_web.report_routes import register_report_routes
from app_web.student_routes import register_student_routes
from recognition.embedder import encoding_cache, generate_encoding
from core.profiling import tracker
from core.utils import (
    allowed_file,
    check_image_quality,
    sanitize_string,
    setup_logging,
    today_str,
    validate_required_fields,
    validate_semester,
    validate_section,
)
from app_web.routes_helpers import (
    _api_error,
    _api_errors,
    _check_rate_limit,
    _first_error_response,
    _is_truthy_flag,
    _is_within_directory,
    _parse_report_filters,
    _rate_limit_key,
    _resolve_upload_reference,
    _validate_date_param,
    _validate_image_mime,
)

logger = setup_logging()

bp = Blueprint("main", __name__)
register_auth_routes(bp)
register_health_routes(bp)
register_public_routes(bp)
register_student_routes(bp)
register_report_routes(bp)
register_attendance_routes(bp)
register_camera_routes(bp)
register_ops_routes(bp)
register_notification_routes(bp)
register_registration_routes(bp)
register_overview_routes(bp)


# ── API endpoints ─────────────────────────────────────────────────────────


