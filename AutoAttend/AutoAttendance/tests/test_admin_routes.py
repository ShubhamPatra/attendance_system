from io import BytesIO

import admin_app as admin_app_module
import mongomock
from bson import ObjectId

from admin_app import create_app
from core.models import AdminDAO, CourseDAO, StudentDAO


def build_logged_in_client():
	app = create_app()
	app.config["TESTING"] = True
	app.config["WTF_CSRF_ENABLED"] = False
	db = mongomock.MongoClient()["autoattendance"]

	admin_app_module._admin_dao = AdminDAO(db)
	admin_app_module._student_dao = StudentDAO(db)
	admin_app_module._course_dao = CourseDAO(db)

	admin_app_module._admin_dao.create_admin("admin@example.com", "Secret123!", "Admin", "admin")

	client = app.test_client()
	login = client.post(
		"/admin/login",
		data={"email": "admin@example.com", "password": "Secret123!", "remember_me": "y"},
		follow_redirects=True,
	)
	assert login.status_code == 200
	return client, db


def test_course_management_flow():
	client, db = build_logged_in_client()
	admin_app_module._course_dao = CourseDAO(db)
	student_dao = StudentDAO(db)
	student_dao.create_student("CS2026008", "s8@example.com", "Secret123!", "Student Eight", "CS", 2)

	response = client.get("/admin/courses")
	assert response.status_code == 200

	add = client.post(
		"/admin/courses/add",
		data={
			"course_code": "CS201",
			"course_name": "Programming II",
			"department": "CS",
			"instructor": "Prof. Knuth",
			"schedule_json": '[{"day":"Monday","start_time":"10:00","end_time":"11:00","room":"C1"}]',
		},
		follow_redirects=True,
	)
	assert add.status_code == 200
	course = db.courses.find_one({"course_code": "CS201"})
	assert course is not None

	edit = client.post(
		f"/admin/courses/{course['_id']}/edit",
		data={
			"course_code": "CS201",
			"course_name": "Programming II Updated",
			"department": "CS",
			"instructor": "Prof. Knuth",
			"schedule_json": '[{"day":"Tuesday","start_time":"12:00","end_time":"13:00","room":"C2"}]',
		},
		follow_redirects=True,
	)
	assert edit.status_code == 200
	assert db.courses.find_one({"_id": course["_id"]})["course_name"] == "Programming II Updated"

	enroll = client.post(
		f"/admin/courses/{course['_id']}/enroll",
		data={"student_ids": ["CS2026008"]},
		follow_redirects=True,
	)
	assert enroll.status_code == 200
	assert db.courses.find_one({"_id": course["_id"]})["enrolled_students"]

	open_session = client.post(
		f"/admin/courses/{course['_id']}/session/open",
		data={"recognition_mode": "BALANCED", "anti_spoofing_enabled": "1", "late_threshold_minutes": 15},
		follow_redirects=True,
	)
	assert open_session.status_code == 200
	assert db.attendance_sessions.find_one({"course_id": course["_id"], "status": "open"}) is not None

	close_session = client.post(f"/admin/courses/{course['_id']}/session/close", follow_redirects=True)
	assert close_session.status_code == 200
	assert db.attendance_sessions.find_one({"course_id": course["_id"], "status": "closed"}) is not None


def test_attendance_management_flow():
	client, db = build_logged_in_client()
	admin_app_module._course_dao = CourseDAO(db)
	student_dao = StudentDAO(db)
	student_dao.create_student("CS2026009", "s9@example.com", "Secret123!", "Student Nine", "CS", 2)
	course_id = admin_app_module._course_dao.create_course(
		"CS301",
		"Operating Systems",
		"CS",
		"Prof. Tannenbaum",
		[{"day": "Wednesday", "start_time": "14:00", "end_time": "15:00", "room": "D1"}],
	)
	admin_app_module._course_dao.enroll_student(course_id, "CS2026009")

	client.post(
		f"/admin/courses/{course_id}/session/open",
		data={"recognition_mode": "BALANCED", "anti_spoofing_enabled": "1", "late_threshold_minutes": 15},
		follow_redirects=True,
	)

	session = db.attendance_sessions.find_one({"course_id": ObjectId(course_id), "status": "open"})
	overview = client.get("/admin/attendance")
	assert overview.status_code == 200

	live = client.get(f"/admin/attendance/session/{session['_id']}")
	assert live.status_code == 200

	manual = client.post(
		"/admin/attendance/manual",
		data={"student_id": "CS2026009", "course_id": course_id, "status": "present", "session_id": str(session["_id"])},
		follow_redirects=True,
	)
	assert manual.status_code == 200

	record = db.attendance_records.find_one({"session_id": str(session["_id"])})
	assert record is not None

	override = client.post(
		f"/admin/attendance/{record['_id']}/override",
		data={"status": "late", "reason": "updated after review"},
		follow_redirects=True,
	)
	assert override.status_code == 200
	assert db.attendance_records.find_one({"_id": record["_id"]})["status"] == "late"


def test_dashboard_renders_with_stats():
	client, db = build_logged_in_client()
	admin_app_module._course_dao = CourseDAO(db)
	student_dao = StudentDAO(db)
	student_dao.create_student("CS2026010", "s10@example.com", "Secret123!", "Student Ten", "CS", 2)
	course_id = admin_app_module._course_dao.create_course(
		"CS401",
		"Distributed Systems",
		"CS",
		"Prof. Lamport",
		[{"day": "Thursday", "start_time": "16:00", "end_time": "17:00", "room": "E1"}],
	)
	admin_app_module._course_dao.enroll_student(course_id, "CS2026010")
	from core.models import AttendanceDAO

	AttendanceDAO(db).record_attendance("CS2026010", course_id, "present", 0.95, 0.93)

	response = client.get("/admin/dashboard")
	assert response.status_code == 200
	assert b"Admin Dashboard" in response.data


def test_reports_generation_and_download():
	client, db = build_logged_in_client()
	admin_app_module._course_dao = CourseDAO(db)

	response = client.get("/admin/reports")
	assert response.status_code == 200

	generated = client.post(
		"/admin/reports/generate",
		data={
			"report_type": "course",
			"date_from": "2026-04-01",
			"date_to": "2026-04-15",
			"course_id": "",
			"student_id": "",
		},
	)
	assert generated.status_code == 200
	payload = generated.get_json()
	assert payload["report_id"]

	download = client.get(f"/admin/reports/download/{payload['report_id']}")
	assert download.status_code == 200
	assert download.data.startswith(b"date,course_code,course_name")


def test_student_list_and_detail():
	client, db = build_logged_in_client()
	student_dao = StudentDAO(db)
	student_dao.create_student("CS2026004", "s4@example.com", "Secret123!", "Student Four", "CS", 2)

	response = client.get("/admin/students")
	assert response.status_code == 200
	assert b"CS2026004" in response.data

	detail = client.get("/admin/students/CS2026004")
	assert detail.status_code == 200
	assert b"Student Four" in detail.data


def test_student_add_edit_delete_flow():
	client, db = build_logged_in_client()

	add_response = client.post(
		"/admin/students/add",
		data={
			"student_id": "CS2026005",
			"name": "Student Five",
			"email": "s5@example.com",
			"department": "CS",
			"year": 1,
			"face_photo_path": "",
		},
		follow_redirects=True,
	)
	assert add_response.status_code == 200
	assert db.students.find_one({"student_id": "CS2026005"}) is not None

	edit_response = client.post(
		"/admin/students/CS2026005/edit",
		data={
			"name": "Student Five Updated",
			"email": "s5@example.com",
			"department": "CS",
			"year": 2,
			"face_photo_path": "",
		},
		follow_redirects=True,
	)
	assert edit_response.status_code == 200
	assert db.students.find_one({"student_id": "CS2026005"})["name"] == "Student Five Updated"

	delete_response = client.post("/admin/students/CS2026005/delete", follow_redirects=True)
	assert delete_response.status_code == 200
	assert db.students.find_one({"student_id": "CS2026005"})["is_active"] is False


def test_student_face_upload_and_bulk_import():
	client, db = build_logged_in_client()
	student_dao = StudentDAO(db)
	student_dao.create_student("CS2026006", "s6@example.com", "Secret123!", "Student Six", "CS", 3)

	upload = client.post(
		"/admin/students/CS2026006/upload-face",
		data={"face_photo": (BytesIO(b"fake image bytes"), "face.png")},
		content_type="multipart/form-data",
	)
	assert upload.status_code == 200
	assert db.students.find_one({"student_id": "CS2026006"})["face_photo_path"]

	bulk_csv = BytesIO(
		b"student_id,email,password,name,department,year\nCS2026007,s7@example.com,Secret123!,Student Seven,CS,2\n"
	)
	bulk = client.post(
		"/admin/students/bulk-upload",
		data={"csv_file": (bulk_csv, "students.csv")},
		content_type="multipart/form-data",
	)
	assert bulk.status_code == 200
	assert db.students.find_one({"student_id": "CS2026007"}) is not None
