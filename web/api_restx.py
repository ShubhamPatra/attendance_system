"""Flask-RESTX API namespaces for structured /api documentation."""

from __future__ import annotations

from flask import request
from flask_login import login_user, logout_user
from flask_restx import Api, Namespace, Resource, fields

import core.database as database
from core.auth import authenticate_user, generate_jwt_token, verify_jwt_token
from core.performance import tracker
from vision.face_engine import get_embedding_backend_name
from web.decorators import require_roles
from web.routes_helpers import _validate_date_param


def init_restx_api(app):
    """Attach Flask-RESTX API with initial namespaces."""
    api = Api(
        app,
        version="2.0",
        title="AutoAttendance API",
        description="Structured REST API for AutoAttendance",
        doc="/api/v2/docs",
        prefix="/api/v2",
    )

    auth_ns = Namespace("auth", description="Authentication")
    health_ns = Namespace("health", description="Health checks")
    public_ns = Namespace("public", description="Read-only public metrics")
    attendance_ns = Namespace("attendance", description="Attendance operations")
    students_ns = Namespace("students", description="Student management")
    report_ns = Namespace("report", description="Attendance reports")
    ops_ns = Namespace("ops", description="Operational metrics")
    camera_ns = Namespace("cameras", description="Multi-camera diagnostics")
    analytics_ns = Namespace("analytics", description="Attendance analytics")

    login_request = auth_ns.model(
        "LoginRequest",
        {
            "username": fields.String(required=True),
            "password": fields.String(required=True),
        },
    )

    login_response = auth_ns.model(
        "LoginResponse",
        {
            "ok": fields.Boolean,
            "username": fields.String,
            "role": fields.String,
            "access_token": fields.String,
            "refresh_token": fields.String,
            "token_type": fields.String,
            "expires_in": fields.Integer,
        },
    )

    attendance_mark_model = attendance_ns.model(
        "AttendanceMarkRequest",
        {
            "reg_no": fields.String(required=True),
            "status": fields.String(required=False, enum=["Present", "Absent"]),
        },
    )

    attendance_session_start_model = attendance_ns.model(
        "AttendanceSessionStartRequest",
        {
            "course_id": fields.String(required=True),
            "camera_id": fields.String(required=True),
        },
    )

    attendance_session_model = attendance_ns.model(
        "AttendanceSession",
        {
            "id": fields.String,
            "course_id": fields.String,
            "camera_id": fields.String,
            "start_time": fields.String,
            "end_time": fields.String,
            "status": fields.String,
            "last_activity_at": fields.String,
        },
    )

    student_model = students_ns.model(
        "Student",
        {
            "_id": fields.String,
            "name": fields.String,
            "registration_number": fields.String,
            "semester": fields.Integer,
            "section": fields.String,
        },
    )

    report_payload_model = report_ns.model(
        "ReportExportRequest",
        {
            "reg_no": fields.String,
            "start_date": fields.String,
            "end_date": fields.String,
            "date": fields.String,
            "full": fields.Boolean,
        },
    )

    trend_model = analytics_ns.model(
        "AttendanceTrend",
        {
            "date": fields.String,
            "present": fields.Integer,
            "total_students": fields.Integer,
            "attendance_pct": fields.Float,
        },
    )

    at_risk_model = analytics_ns.model(
        "AtRiskStudent",
        {
            "name": fields.String,
            "reg_no": fields.String,
            "percentage": fields.Float,
            "attendance_pct": fields.Float,
            "days_present": fields.Integer,
            "days_total": fields.Integer,
        },
    )

    @auth_ns.route("/login")
    class AuthLogin(Resource):
        @auth_ns.expect(login_request)
        @auth_ns.response(200, "Logged in", login_response)
        @auth_ns.response(401, "Invalid credentials")
        def post(self):
            payload = request.get_json(silent=True) or {}
            username = (payload.get("username") or "").strip()
            password = payload.get("password") or ""
            
            # Record login attempt and check lockout
            user = authenticate_user(username, password)
            if user is None:
                # Log failed attempt
                database.log_auth_event(
                    event_type='API_LOGIN_FAILED',
                    status='invalid_credentials',
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', ''),
                    details={'username': username}
                )
                return {"error": "Invalid credentials."}, 401
            
            # Log successful login
            database.log_auth_event(
                event_type='API_LOGIN',
                user_id=user.id,
                status='success',
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', ''),
            )
            
            # Generate JWT tokens
            access_token = generate_jwt_token(user.id, user.role, expires_in_hours=1)
            refresh_token = generate_jwt_token(user.id, user.role, expires_in_hours=24)
            
            login_user(user, remember=True)
            return {
                "ok": True,
                "username": user.username,
                "role": user.role,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": 3600,  # 1 hour in seconds
            }

    @auth_ns.route("/logout")
    class AuthLogout(Resource):
        def post(self):
            database.log_auth_event(
                event_type='API_LOGOUT',
                status='success',
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', ''),
            )
            logout_user()
            return {"ok": True}

    refresh_request = auth_ns.model(
        "RefreshTokenRequest",
        {
            "refresh_token": fields.String(required=True),
        },
    )

    @auth_ns.route("/refresh")
    class AuthRefresh(Resource):
        @auth_ns.expect(refresh_request)
        @auth_ns.response(200, "Token refreshed", login_response)
        @auth_ns.response(401, "Invalid token")
        def post(self):
            payload = request.get_json(silent=True) or {}
            refresh_token = payload.get("refresh_token", "")
            
            # Verify refresh token
            token_data = verify_jwt_token(refresh_token)
            if not token_data:
                database.log_auth_event(
                    event_type='TOKEN_REFRESH_FAILED',
                    status='invalid_token',
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent', ''),
                )
                return {"error": "Invalid or expired refresh token"}, 401
            
            # Generate new access token
            user_id = token_data.get('user_id')
            user_role = token_data.get('role', 'student')
            
            access_token = generate_jwt_token(user_id, user_role, expires_in_hours=1)
            
            database.log_auth_event(
                event_type='TOKEN_REFRESHED',
                user_id=user_id,
                status='success',
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', ''),
            )
            
            return {
                "ok": True,
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": 3600,
            }

    @health_ns.route("/health")
    class Health(Resource):
        def get(self):
            return {"status": "ok", "service": "autoattendance"}

    @health_ns.route("/ready")
    class Ready(Resource):
        def get(self):
            return {"status": "ready" if database.get_client() else "degraded"}

    @public_ns.route("/attendance-activity")
    class AttendanceActivity(Resource):
        def get(self):
            date = request.args.get("date") or ""
            if not date:
                from core.utils import today_str

                date = today_str()
            _, err = _validate_date_param(date, "date")
            if err:
                return {"error": err}, 400
            return {"date": date, "hours": database.get_attendance_by_hour(date)}

    @public_ns.route("/heatmap")
    class Heatmap(Resource):
        def get(self):
            return database.get_attendance_heatmap_data()

    @attendance_ns.route("")
    class AttendanceList(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        def get(self):
            reg_no = (request.args.get("reg_no") or "").strip()
            date = (request.args.get("date") or "").strip()
            if reg_no:
                return database.get_attendance_by_student(reg_no)
            if date:
                return database.get_attendance(date)
            return database.get_attendance()

        @attendance_ns.expect(attendance_mark_model)
        def post(self):
            payload = request.get_json(silent=True) or {}
            reg_no = (payload.get("reg_no") or "").strip()
            status = payload.get("status", "Present")
            if not reg_no:
                return {"error": "reg_no is required."}, 400
            if status not in ("Present", "Absent"):
                return {"error": "status must be 'Present' or 'Absent'."}, 400

            student = database.get_student_by_reg_no(reg_no)
            if student is None:
                return {"error": "Student not found."}, 404

            changed = database.bulk_upsert_attendance(
                [
                    {
                        "student_id": student["_id"],
                        "status": status,
                        "confidence_score": 0.0,
                    }
                ]
            )
            return {"marked": changed > 0, "reg_no": reg_no, "status": status}

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

    @attendance_ns.route("/sessions")
    class AttendanceSessionStart(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        @attendance_ns.expect(attendance_session_start_model)
        @attendance_ns.marshal_with(
            attendance_ns.model(
                "AttendanceSessionStartResponse",
                {
                    "created": fields.Boolean,
                    "session": fields.Nested(attendance_session_model),
                },
            )
        )
        def post(self):
            payload = request.get_json(silent=True) or {}
            course_id = (payload.get("course_id") or "").strip()
            camera_id = (payload.get("camera_id") or "").strip()
            if not course_id:
                return {"error": "course_id is required."}, 400
            if not camera_id:
                return {"error": "camera_id is required."}, 400

            try:
                sid = database.create_attendance_session(course_id=course_id, camera_id=camera_id)
                return {
                    "created": True,
                    "session": _session_payload(database.get_attendance_session_by_id(sid)),
                }, 201
            except ValueError as exc:
                return {"error": str(exc)}, 409

    @attendance_ns.route("/sessions/<string:session_id>/end")
    class AttendanceSessionEnd(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        def post(self, session_id: str):
            try:
                ended = database.end_attendance_session(session_id)
            except Exception:
                return {"error": "Invalid session_id."}, 400
            if not ended:
                return {"error": "Active session not found."}, 404
            return {
                "ended": True,
                "session": _session_payload(database.get_attendance_session_by_id(session_id)),
            }

    @attendance_ns.route("/sessions/active")
    class AttendanceSessionActive(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        def get(self):
            camera_id = (request.args.get("camera_id") or "").strip()
            if not camera_id:
                return {"error": "camera_id is required."}, 400

            database.auto_close_idle_attendance_sessions()
            session = database.get_active_attendance_session(camera_id)
            return {
                "active": session is not None,
                "session": _session_payload(session),
            }

    @students_ns.route("")
    class StudentsList(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        @students_ns.marshal_list_with(student_model)
        def get(self):
            students = database.get_all_students()
            for row in students:
                row["_id"] = str(row["_id"])
            return students

    @ops_ns.route("/metrics")
    class OpsMetrics(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        def get(self):
            from camera.camera import get_camera_diagnostics

            payload = tracker.metrics()
            payload["embedding_backend"] = get_embedding_backend_name()
            payload["camera_diagnostics"] = get_camera_diagnostics()
            return payload

    @camera_ns.route("")
    class CameraDiagnostics(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        def get(self):
            from camera.camera import get_camera_diagnostics

            return get_camera_diagnostics()

    @report_ns.route("/csv/async")
    class ReportCsvAsync(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        @report_ns.expect(report_payload_model)
        def post(self):
            payload = request.get_json(silent=True) or {}
            reg_no = (payload.get("reg_no") or "").strip()
            start_date = (payload.get("start_date") or "").strip()
            end_date = (payload.get("end_date") or "").strip()
            date = (payload.get("date") or "").strip()
            full = bool(payload.get("full"))
            if full:
                return {"task_id": "full"}
            if reg_no:
                return {"task_id": f"student:{reg_no}"}
            if start_date and end_date:
                return {"task_id": f"range:{start_date}:{end_date}"}
            return {"task_id": f"date:{date or 'today'}"}

    @analytics_ns.route("/trends")
    class AnalyticsTrends(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        @analytics_ns.marshal_list_with(trend_model)
        def get(self):
            days = request.args.get("days", 14, type=int)
            return database.get_attendance_trends(days=days)

    @analytics_ns.route("/at-risk")
    class AnalyticsAtRisk(Resource):
        method_decorators = [require_roles("admin", "teacher")]

        @analytics_ns.marshal_list_with(at_risk_model)
        def get(self):
            days = request.args.get("days", 30, type=int)
            threshold = request.args.get("threshold", None, type=int)
            data = database.get_at_risk_students(days=days, threshold=threshold)
            for row in data:
                row["attendance_pct"] = row.get("percentage", row.get("attendance_pct", 0))
            return data

    api.add_namespace(auth_ns)
    api.add_namespace(health_ns)
    api.add_namespace(public_ns)
    api.add_namespace(attendance_ns)
    api.add_namespace(students_ns)
    api.add_namespace(report_ns)
    api.add_namespace(ops_ns)
    api.add_namespace(camera_ns)
    api.add_namespace(analytics_ns)

    return api
