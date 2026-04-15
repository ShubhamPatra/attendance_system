from datetime import datetime, timezone

import mongomock
import numpy as np
from pymongo.errors import PyMongoError

from core.database import ensure_indexes, retry_operation
from core.models import AdminDAO, AttendanceDAO, CourseDAO, LogDAO, SessionDAO, StudentDAO


def test_retry_operation_eventual_success():
	state = {"tries": 0}

	@retry_operation(max_attempts=3, base_delay=0)
	def flaky():
		state["tries"] += 1
		if state["tries"] < 3:
			raise PyMongoError("temporary")
		return "ok"

	assert flaky() == "ok"
	assert state["tries"] == 3


def test_ensure_indexes_idempotent():
	client = mongomock.MongoClient()
	db = client["autoattendance"]

	ensure_indexes(db)
	ensure_indexes(db)

	students_indexes = db.students.index_information()
	courses_indexes = db.courses.index_information()
	logs_indexes = db.system_logs.index_information()

	assert any("student_id" in name for name in students_indexes)
	assert any("email" in name for name in students_indexes)
	assert any("course_code" in name for name in courses_indexes)
	assert any("timestamp" in name for name in logs_indexes)


def test_admin_dao_crud_and_pagination():
	db = mongomock.MongoClient()["autoattendance"]
	dao = AdminDAO(db)

	admin_id = dao.create_admin("admin@example.com", "Secret123!", "Main Admin", "admin")
	assert admin_id

	by_email = dao.get_by_email("admin@example.com")
	assert by_email is not None
	assert by_email["email"] == "admin@example.com"
	assert by_email["password_hash"] != "Secret123!"

	by_id = dao.get_by_id(admin_id)
	assert by_id is not None
	assert by_id["name"] == "Main Admin"

	assert dao.update_admin(admin_id, {"name": "Updated Name"})
	assert dao.get_by_id(admin_id)["name"] == "Updated Name"

	assert dao.update_last_login(admin_id)
	assert dao.get_by_id(admin_id)["last_login"] is not None

	listing = dao.list_admins(page=1, per_page=10)
	assert listing["total"] == 1
	assert len(listing["items"]) == 1

	assert dao.deactivate_admin(admin_id)
	assert dao.get_by_email("admin@example.com") is None


def test_student_dao_embeddings_and_search():
	db = mongomock.MongoClient()["autoattendance"]
	dao = StudentDAO(db)

	sid = dao.create_student(
		student_id="CS2026001",
		email="s1@example.com",
		password="Secret123!",
		name="Student One",
		department="CS",
		year=2,
	)
	assert sid

	emb = np.ones(512, dtype=np.float32)
	assert dao.add_embedding("CS2026001", emb, 0.9, "admin_upload")

	fetched = dao.get_embeddings("CS2026001")
	assert len(fetched) == 1
	assert fetched[0]["embedding"].shape[0] == 512
	assert np.allclose(fetched[0]["embedding"], emb)

	student_doc = db.students.find_one({"student_id": "CS2026001"})
	course_id = db.courses.insert_one(
		{
			"course_code": "CS101",
			"course_name": "Intro CS",
			"enrolled_students": [student_doc["_id"]],
		}
	).inserted_id

	roster = dao.get_roster_embeddings(str(course_id))
	assert len(roster) == 1
	assert roster[0]["student_id"] == "CS2026001"

	results = dao.search_students(query="Student", department="CS", year=2, page=1, per_page=10)
	assert results["total"] == 1
	assert len(results["items"]) == 1


def test_student_dao_bulk_create_and_soft_delete():
	db = mongomock.MongoClient()["autoattendance"]
	dao = StudentDAO(db)

	inserted = dao.bulk_create_students(
		[
			{
				"student_id": "EE2026001",
				"email": "ee1@example.com",
				"password": "Secret123!",
				"name": "EE Student",
				"department": "EE",
				"year": 3,
			},
			{
				"student_id": "ME2026001",
				"email": "me1@example.com",
				"password": "Secret123!",
				"name": "ME Student",
				"department": "ME",
				"year": 1,
			},
		]
	)
	assert inserted == 2

	assert dao.get_by_student_id("EE2026001") is not None
	assert dao.get_by_email("me1@example.com") is not None
	assert dao.soft_delete_student("EE2026001")
	assert dao.get_by_student_id("EE2026001") is None


def test_course_dao_enrollment_and_queries():
	db = mongomock.MongoClient()["autoattendance"]
	student_dao = StudentDAO(db)
	course_dao = CourseDAO(db)

	student_dao.create_student("CS2026002", "s2@example.com", "Secret123!", "Student Two", "CS", 2)
	course_id = course_dao.create_course(
		"CS102",
		"Data Structures",
		"CS",
		"Prof. Ada",
		[{"day": "Monday", "start_time": "09:00", "end_time": "10:00", "room": "A1"}],
	)

	assert course_dao.enroll_student(course_id, "CS2026002")
	enrolled = course_dao.get_enrolled_students(course_id, page=1, per_page=10)
	assert enrolled["total"] == 1
	assert len(enrolled["items"]) == 1

	active = course_dao.get_active_courses_for_student("CS2026002")
	assert len(active) == 1
	assert active[0]["course_code"] == "CS102"

	db.attendance_sessions.insert_one(
		{
			"course_id": active[0]["_id"],
			"status": "open",
			"opened_at": 1,
		}
	)
	assert course_dao.get_current_session(course_id) is not None

	assert course_dao.unenroll_student(course_id, "CS2026002")
	enrolled_after = course_dao.get_enrolled_students(course_id, page=1, per_page=10)
	assert enrolled_after["total"] == 0


def test_attendance_and_session_flow():
	db = mongomock.MongoClient()["autoattendance"]
	student_dao = StudentDAO(db)
	course_dao = CourseDAO(db)
	attendance_dao = AttendanceDAO(db)
	session_dao = SessionDAO(db)

	student_dao.create_student("CS2026003", "s3@example.com", "Secret123!", "Student Three", "CS", 1)
	course_id = course_dao.create_course(
		"CS103",
		"Algorithms",
		"CS",
		"Prof. Turing",
		[{"day": "Tuesday", "start_time": "11:00", "end_time": "12:00", "room": "B2"}],
	)
	course_dao.enroll_student(course_id, "CS2026003")

	admin_id = db.admins.insert_one({"email": "a@example.com", "is_active": True}).inserted_id
	session = session_dao.open_session(
		course_id,
		str(admin_id),
		{"recognition_mode": "BALANCED", "anti_spoofing_enabled": True, "late_threshold_minutes": 15},
	)
	assert session["opened"]
	assert session_dao.get_open_session(course_id) is not None

	rec1 = attendance_dao.record_attendance("CS2026003", course_id, "present", 0.92, 0.88)
	assert rec1["inserted"]
	rec_dup = attendance_dao.record_attendance("CS2026003", course_id, "present", 0.90, 0.85)
	assert not rec_dup["inserted"]
	assert rec_dup["reason"] == "duplicate"

	today = str(datetime.now(timezone.utc).date())
	by_course = attendance_dao.get_for_course(course_id, today)
	assert len(by_course) == 1

	by_student = attendance_dao.get_for_student("CS2026003", today, today)
	assert len(by_student) == 1

	stats = attendance_dao.get_stats(course_id)
	assert stats["present"] == 1

	assert attendance_dao.override_status(rec1["id"], "late")
	updated_stats = attendance_dao.get_stats(course_id)
	assert updated_stats["late"] == 1

	assert session_dao.close_session(session["session_id"])


def test_log_dao_queries():
	db = mongomock.MongoClient()["autoattendance"]
	log_dao = LogDAO(db)

	actor_id = str(db.admins.insert_one({"email": "auditor@example.com", "is_active": True}).inserted_id)
	first_id = log_dao.log_event("login", actor_id, "admin", {"ok": True}, "127.0.0.1")
	second_id = log_dao.log_event("attendance_marked", None, "system", {"count": 1}, None)

	assert first_id and second_id

	recent = log_dao.get_recent_logs(limit=10)
	assert len(recent) == 2

	login_logs = log_dao.get_recent_logs(event_type="login", limit=10)
	assert len(login_logs) == 1
	assert login_logs[0]["event_type"] == "login"

	actor_logs = log_dao.get_logs_by_actor(actor_id, limit=10)
	assert len(actor_logs) == 1


def test_invalid_id_edge_cases():
	db = mongomock.MongoClient()["autoattendance"]
	admin_dao = AdminDAO(db)
	course_dao = CourseDAO(db)
	attendance_dao = AttendanceDAO(db)
	session_dao = SessionDAO(db)
	log_dao = LogDAO(db)

	assert admin_dao.get_by_id("not-an-id") is None
	assert not admin_dao.update_admin("not-an-id", {"name": "x"})
	assert not course_dao.update_course("not-an-id", {"course_name": "x"})
	assert attendance_dao.get_for_student("missing", "2026-01-01", "2026-01-02") == []
	assert not attendance_dao.override_status("not-an-id", "present")
	assert session_dao.get_open_session("not-an-id") is None
	assert log_dao.get_logs_by_actor("not-an-id") == []
