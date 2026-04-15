import base64

import cv2
import mongomock
import numpy as np
import student_app as student_app_module
from core import socketio
from core.models import AttendanceDAO, CourseDAO, StudentDAO
from student_app import create_app


def build_student_client():
	app = create_app()
	app.config["TESTING"] = True
	app.config["WTF_CSRF_ENABLED"] = False

	student_app_module._student_dao = StudentDAO(mongomock.MongoClient()["autoattendance_test"])
	student_app_module._student_dao.collection.delete_many({})

	return app, app.test_client(), student_app_module._student_dao


def test_student_factory_registers_blueprints_and_health_endpoint():
	app = create_app()
	client = app.test_client()

	response = client.get("/student/health")
	assert response.status_code == 200
	assert response.get_json() == {
		"status": "ok",
		"service": "student",
		"db": "connected",
		"redis": "connected",
	}

	registered = set(app.blueprints.keys())
	assert "student_auth" in registered
	assert "student_attendance" in registered
	assert "student_profile" in registered


def test_student_webcam_routes_include_cors_headers():
	_, client, _ = build_student_client()

	response = client.get("/student/attendance/mark")
	assert response.status_code == 302
	assert "/student/login" in response.headers.get("Location", "")
	assert response.headers.get("Access-Control-Allow-Origin") == "*"
	allow_methods = response.headers.get("Access-Control-Allow-Methods", "")
	assert "GET" in allow_methods
	assert "POST" in allow_methods


def test_student_login_view_is_student_login_route():
	app = create_app()

	assert app.login_manager.login_view == "student_auth.login"


def test_student_register_login_logout_flow():
	_, client, dao = build_student_client()

	register_response = client.post(
		"/student/register",
		data={
			"name": "Student One",
			"student_id": "CS2026101",
			"email": "student1@example.com",
			"password": "Password123",
			"confirm_password": "Password123",
			"department": "CSE",
			"year": "2",
			"face_photo_path": "uploads/students/CS2026101_face.png",
		},
	)
	assert register_response.status_code == 302
	assert "/student/profile" in register_response.headers.get("Location", "")

	created = dao.get_by_student_id("CS2026101")
	assert created is not None
	assert created["face_photo_path"] == "uploads/students/CS2026101_face.png"

	logout_response = client.get("/student/logout")
	assert logout_response.status_code == 302
	assert "/student/login" in logout_response.headers.get("Location", "")

	login_response = client.post(
		"/student/login",
		data={"student_id": "CS2026101", "password": "Password123"},
	)
	assert login_response.status_code == 302
	assert "/student/profile" in login_response.headers.get("Location", "")


def test_student_verification_socket_flow_uses_verifier_integration(monkeypatch):
	app, client, dao = build_student_client()

	client.post(
		"/student/register",
		data={
			"name": "Socket Student",
			"student_id": "CS2026202",
			"email": "socket.student@example.com",
			"password": "Password123",
			"confirm_password": "Password123",
			"department": "CSE",
			"year": "3",
			"face_photo_path": "",
		},
	)

	course_dao = CourseDAO(dao.db)
	course_id = course_dao.create_course(
		course_code="CSE202",
		course_name="Computer Vision",
		department="CSE",
		instructor="Dr. Vision",
		schedule=[{"day": "Tue", "start": "10:00", "end": "11:00"}],
	)
	assert course_dao.enroll_student(course_id, "CS2026202") is True

	class _FakeSessionManager:
		def delete_session(self, _session_id):
			return True

	expected_course_id = course_id

	class _FakeVerifier:
		def __init__(self):
			self.session_manager = _FakeSessionManager()
			self._student_id = None
			self._course_id = None
			self._attendance_dao = AttendanceDAO(dao.db)

		def start_session(self, student_id, course_id):
			assert student_id == "CS2026202"
			assert course_id == expected_course_id
			self._student_id = student_id
			self._course_id = course_id
			return "session-1"

		def process_frame(self, session_id, frame):
			assert session_id == "session-1"
			assert frame is not None
			return {
				"status": "processing",
				"recognition": {
					"status": "matched",
					"matched": True,
					"student_id": "CS2026202",
					"vote_count": 3,
				},
				"liveness": {
					"is_real": True,
					"score": 0.9,
					"confidence_level": "HIGH",
					"details": {},
				},
			}

		def finalize(self, session_id):
			assert session_id == "session-1"
			insert_result = self._attendance_dao.record_attendance(
				student_id=self._student_id,
				course_id=self._course_id,
				status="present",
				confidence=0.94,
				liveness_score=0.96,
				verification_method="face",
				session_id=session_id,
			)

			class _FakeResult:
				status = "success" if insert_result.get("inserted") else "failed"
				message = "Attendance marked" if insert_result.get("inserted") else "Attendance record not inserted"
				confidence = 0.94
				liveness_score = 0.96
				attendance_recorded = bool(insert_result.get("inserted"))

			return _FakeResult()

	import student_app.routes.attendance as attendance_routes

	fake_verifier = _FakeVerifier()
	monkeypatch.setattr(attendance_routes, "_get_verifier", lambda: fake_verifier)

	socket_client = socketio.test_client(app, flask_test_client=client)
	assert socket_client.is_connected()

	socket_client.emit("start_verification")
	received = socket_client.get_received()
	assert any(item["name"] == "verification_status" and item["args"][0]["status"] == "session_started" for item in received)

	socket_client.emit("frame", {"frame": "invalid-data"})
	received = socket_client.get_received()
	assert any(item["name"] == "verification_status" and item["args"][0]["status"] == "invalid_frame" for item in received)

	img = np.zeros((16, 16, 3), dtype=np.uint8)
	encoded_ok, encoded = cv2.imencode(".jpg", img)
	assert encoded_ok is True
	frame_payload = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

	socket_client.emit("frame", {"frame": frame_payload})
	received = socket_client.get_received()
	statuses = [item["args"][0]["status"] for item in received if item["name"] == "verification_status"]
	assert "identity_verified" in statuses
	assert "attendance_marked" in statuses

	student = dao.get_by_student_id("CS2026202")
	assert student is not None
	record = dao.db.attendance_records.find_one(
		{
			"student_id": student["_id"],
			"course_id": student["enrolled_courses"][0],
			"status": "present",
		}
	)
	assert record is not None
	assert record.get("verification_method") == "face"

	socket_client.emit("cancel_verification")
	received = socket_client.get_received()
	assert any(item["name"] == "verification_status" and item["args"][0]["status"] == "cancelled" for item in received)

	socket_client.disconnect()


def test_student_profile_and_attendance_views_flow():
	_, client, dao = build_student_client()

	client.post(
		"/student/register",
		data={
			"name": "Profile Student",
			"student_id": "CS2026303",
			"email": "profile.student@example.com",
			"password": "Password123",
			"confirm_password": "Password123",
			"department": "CSE",
			"year": "3",
			"face_photo_path": "",
		},
	)

	course_dao = CourseDAO(dao.db)
	attendance_dao = AttendanceDAO(dao.db)
	course_id = course_dao.create_course(
		course_code="CSE101",
		course_name="Algorithms",
		department="CSE",
		instructor="Dr. Ada",
		schedule=[{"day": "Mon", "start": "09:00", "end": "10:00"}],
	)
	assert course_dao.enroll_student(course_id, "CS2026303") is True

	record = attendance_dao.record_attendance(
		student_id="CS2026303",
		course_id=course_id,
		status="present",
		confidence=0.92,
		liveness_score=0.95,
	)
	assert record["inserted"] is True

	history_response = client.get("/student/attendance/history")
	assert history_response.status_code == 200
	assert b"Attendance History" in history_response.data

	status_response = client.get("/student/attendance/status")
	assert status_response.status_code == 200
	assert b"CSE101" in status_response.data

	courses_response = client.get("/student/profile/courses")
	assert courses_response.status_code == 200
	assert b"Algorithms" in courses_response.data

	update_face_response = client.post(
		"/student/profile/update-face",
		data={"face_photo_path": "uploads/students/CS2026303_face.png"},
	)
	assert update_face_response.status_code == 302

	updated_student = dao.get_by_student_id("CS2026303")
	assert updated_student is not None
	assert updated_student["face_photo_path"] == "uploads/students/CS2026303_face.png"

	profile_response = client.get("/student/profile")
	assert profile_response.status_code == 200
	assert b"Student Profile" in profile_response.data
