"""Seed MongoDB with sample data for development and testing."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from bson import ObjectId

from core.config import load_config
from core.database import ensure_indexes, get_mongo_client
from core.models import AdminDAO, AttendanceDAO, CourseDAO, StudentDAO


DEPARTMENTS = ["CS", "EE", "ME"]


def random_name(prefix: str, index: int) -> str:
	return f"{prefix} {index:03d}"


def seed_admins(admin_dao: AdminDAO) -> int:
	roles = [
		("super_admin@autoattendance.com", "Super Admin", "super_admin"),
		("admin@autoattendance.com", "Admin User", "admin"),
		("viewer@autoattendance.com", "Viewer User", "viewer"),
	]
	for email, name, role in roles:
		admin_dao.create_admin(email, "ChangeMe123!", name, role)
	return len(roles)


def seed_students(student_dao: StudentDAO) -> list[str]:
	student_ids: list[str] = []
	for i in range(1, 51):
		dept = DEPARTMENTS[(i - 1) % len(DEPARTMENTS)]
		sid = f"{dept}{2026}{i:03d}"
		student_dao.create_student(
			student_id=sid,
			email=f"student{i:03d}@example.com",
			password="Student123!",
			name=random_name("Student", i),
			department=dept,
			year=random.randint(1, 4),
		)
		student_ids.append(sid)
	return student_ids


def seed_courses(course_dao: CourseDAO) -> list[str]:
	courses = [
		("CS101", "Intro to Computing", "CS"),
		("CS102", "Data Structures", "CS"),
		("CS103", "Algorithms", "CS"),
		("EE101", "Circuit Fundamentals", "EE"),
		("EE102", "Signals and Systems", "EE"),
		("EE103", "Electromagnetics", "EE"),
		("ME101", "Engineering Mechanics", "ME"),
		("ME102", "Thermodynamics", "ME"),
		("ME103", "Fluid Mechanics", "ME"),
		("GEN201", "Professional Ethics", "CS"),
	]

	ids: list[str] = []
	for idx, (code, name, dept) in enumerate(courses, start=1):
		day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][idx % 5]
		start_hour = 8 + (idx % 6)
		schedule = [
			{
				"day": day,
				"start_time": f"{start_hour:02d}:00",
				"end_time": f"{start_hour + 1:02d}:00",
				"room": f"R{100 + idx}",
			}
		]
		course_id = course_dao.create_course(code, name, dept, f"Instructor {idx}", schedule)
		ids.append(course_id)
	return ids


def seed_enrollments(course_dao: CourseDAO, student_ids: list[str], course_ids: list[str]) -> int:
	enrollments = 0
	for course_id in course_ids:
		for sid in random.sample(student_ids, random.randint(15, 30)):
			if course_dao.enroll_student(course_id, sid):
				enrollments += 1
	return enrollments


def seed_attendance(attendance_dao: AttendanceDAO, db, course_ids: list[str]) -> int:
	records = 0
	today = datetime.now(timezone.utc).date()

	for days_ago in range(30):
		date_value = str(today - timedelta(days=days_ago))
		for course_id in course_ids:
			course = db.courses.find_one({"_id": ObjectId(course_id)})
			if not course:
				continue
			enrolled_ids = course.get("enrolled_students", [])
			for sid_obj in enrolled_ids:
				student = db.students.find_one({"_id": sid_obj})
				if not student:
					continue
				roll = random.random()
				if roll < 0.8:
					status = "present"
				elif roll < 0.9:
					status = "late"
				else:
					status = "absent"
				result = attendance_dao.record_attendance(
					student_id=student["student_id"],
					course_id=course_id,
					status=status,
					confidence=round(random.uniform(0.75, 0.99), 2),
					liveness_score=round(random.uniform(0.7, 0.99), 2),
					verification_method="face",
					check_date=date_value,
				)
				if result.get("inserted"):
					records += 1
	return records


def main() -> None:
	cfg = load_config()
	mongo = get_mongo_client(cfg)
	db = mongo.get_db()

	# Idempotent reset
	db.admins.delete_many({})
	db.students.delete_many({})
	db.courses.delete_many({})
	db.attendance_records.delete_many({})
	db.attendance_sessions.delete_many({})
	db.system_logs.delete_many({})

	ensure_indexes(db)

	admin_dao = AdminDAO(db)
	student_dao = StudentDAO(db)
	course_dao = CourseDAO(db)
	attendance_dao = AttendanceDAO(db)

	admins_count = seed_admins(admin_dao)
	student_ids = seed_students(student_dao)
	course_ids = seed_courses(course_dao)
	enrollment_count = seed_enrollments(course_dao, student_ids, course_ids)
	attendance_count = seed_attendance(attendance_dao, db, course_ids)

	print("Seeding complete")
	print(f"Admins: {admins_count}")
	print(f"Students: {len(student_ids)}")
	print(f"Courses: {len(course_ids)}")
	print(f"Enrollments: {enrollment_count}")
	print(f"Attendance records: {attendance_count}")


if __name__ == "__main__":
	main()
