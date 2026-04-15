from __future__ import annotations

import time

import redis

from flask import Flask, jsonify, render_template, request

from admin_app.routes import attendance_bp, auth_bp, courses_bp, dashboard_bp, reports_bp, students_bp
from core import limiter, load_config, login_manager, socketio
from core.database import ensure_indexes, get_mongo_client
from core.models import AdminDAO, CourseDAO, StudentDAO


_admin_dao: AdminDAO | None = None
_student_dao: StudentDAO | None = None
_course_dao: CourseDAO | None = None


def get_admin_dao() -> AdminDAO:
	if _admin_dao is None:
		raise RuntimeError("AdminDAO has not been initialized")
	return _admin_dao


def get_student_dao() -> StudentDAO:
	if _student_dao is None:
		raise RuntimeError("StudentDAO has not been initialized")
	return _student_dao


def get_course_dao() -> CourseDAO:
	if _course_dao is None:
		raise RuntimeError("CourseDAO has not been initialized")
	return _course_dao


def create_app(config_name: str | None = None) -> Flask:
	del config_name  # environment-based config loading is used for now

	app = Flask(__name__)
	cfg = load_config()
	app.config.from_mapping(
		SECRET_KEY=cfg.SECRET_KEY,
		ENV=cfg.ENV,
		DEBUG=cfg.DEBUG,
		SESSION_COOKIE_SECURE=cfg.SESSION_COOKIE_SECURE,
		SESSION_COOKIE_HTTPONLY=cfg.SESSION_COOKIE_HTTPONLY,
		SESSION_COOKIE_SAMESITE=cfg.SESSION_COOKIE_SAMESITE,
		PERMANENT_SESSION_LIFETIME=cfg.PERMANENT_SESSION_LIFETIME,
	)

	login_manager.init_app(app)
	login_manager.login_view = "admin_auth.login"
	from admin_app.routes.auth import load_user as load_admin_user

	login_manager.user_loader(load_admin_user)
	limiter.init_app(app)
	socketio.init_app(app)

	app.register_blueprint(auth_bp, url_prefix="/admin")
	app.register_blueprint(dashboard_bp, url_prefix="/admin")
	app.register_blueprint(students_bp, url_prefix="/admin")
	app.register_blueprint(courses_bp, url_prefix="/admin")
	app.register_blueprint(attendance_bp, url_prefix="/admin")
	app.register_blueprint(reports_bp, url_prefix="/admin")

	@app.before_request
	def start_timer():
		request._start_time = time.perf_counter()

	@app.after_request
	def request_logger(response):
		start = getattr(request, "_start_time", None)
		if start is not None:
			duration_ms = (time.perf_counter() - start) * 1000
			app.logger.info(
				"%s %s -> %s (%.2fms)",
				request.method,
				request.path,
				response.status_code,
				duration_ms,
			)
		return response

	def _error_payload(code: int, message: str):
		wants_json = request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html
		if wants_json:
			return jsonify({"error": message, "status": code}), code
		return render_template(f"{code}.html"), code

	@app.errorhandler(400)
	def bad_request(_):
		return _error_payload(400, "Bad Request")

	@app.errorhandler(403)
	def forbidden(_):
		return _error_payload(403, "Forbidden")

	@app.errorhandler(404)
	def not_found(_):
		return _error_payload(404, "Not Found")

	@app.errorhandler(500)
	def internal_error(_):
		return _error_payload(500, "Internal Server Error")

	try:
		global _admin_dao, _student_dao, _course_dao
		mongo = get_mongo_client(cfg)
		db = mongo.get_db()
		ensure_indexes(db)
		_admin_dao = AdminDAO(db)
		_student_dao = StudentDAO(db)
		_course_dao = CourseDAO(db)
	except Exception as exc:  # pragma: no cover - environment dependent
		app.logger.warning("MongoDB startup checks skipped: %s", exc)

	def _dependency_health() -> tuple[dict[str, str], int]:
		mongo_status = "connected"
		redis_status = "connected"
		status_code = 200

		try:
			mongo_client = get_mongo_client(cfg)
			mongo_client.get_db().command("ping")
		except Exception as exc:  # pragma: no cover - environment dependent
			app.logger.warning("MongoDB health check failed: %s", exc)
			mongo_status = "disconnected"
			status_code = 503

		try:
			redis.Redis.from_url(cfg.REDIS_URL).ping()
		except Exception as exc:  # pragma: no cover - environment dependent
			app.logger.warning("Redis health check failed: %s", exc)
			redis_status = "disconnected"
			status_code = 503

		return {"db": mongo_status, "redis": redis_status}, status_code

	@app.get("/admin/health")
	def health():
		payload, status_code = _dependency_health()
		payload["status"] = "ok" if status_code == 200 else "degraded"
		payload["service"] = "admin"
		return payload, status_code

	return app
