"""Data access objects for MongoDB collections."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from bson import ObjectId
from bson.binary import Binary
from pymongo import InsertOne
from pymongo.database import Database

from core.config import Config
from core.database import get_mongo_client
from core.utils import hash_password


def _now() -> datetime:
	return datetime.now(tz=timezone.utc)


def _as_object_id(value: str) -> ObjectId | None:
	try:
		return ObjectId(value)
	except Exception:
		return None


class AdminDAO:
	def __init__(self, db: Database) -> None:
		self.collection = db.admins

	@classmethod
	def from_config(cls, config: Config) -> "AdminDAO":
		client = get_mongo_client(config)
		return cls(client.get_db())

	def create_admin(self, email: str, password: str, name: str, role: str) -> str:
		now = _now()
		doc = {
			"email": email.lower().strip(),
			"password_hash": hash_password(password),
			"name": name,
			"role": role,
			"is_active": True,
			"created_at": now,
			"updated_at": now,
			"last_login": None,
		}
		result = self.collection.insert_one(doc)
		return str(result.inserted_id)

	def get_by_email(self, email: str) -> dict[str, Any] | None:
		return self.collection.find_one({"email": email.lower().strip(), "is_active": True})

	def get_by_id(self, admin_id: str) -> dict[str, Any] | None:
		oid = _as_object_id(admin_id)
		if oid is None:
			return None
		return self.collection.find_one({"_id": oid, "is_active": True})

	def update_admin(self, admin_id: str, fields: dict[str, Any]) -> bool:
		if not fields:
			return False
		oid = _as_object_id(admin_id)
		if oid is None:
			return False
		fields = dict(fields)
		fields["updated_at"] = _now()
		result = self.collection.update_one({"_id": oid}, {"$set": fields})
		return result.modified_count > 0

	def update_last_login(self, admin_id: str) -> bool:
		oid = _as_object_id(admin_id)
		if oid is None:
			return False
		result = self.collection.update_one(
			{"_id": oid},
			{"$set": {"last_login": _now(), "updated_at": _now()}},
		)
		return result.modified_count > 0

	def list_admins(self, page: int = 1, per_page: int = 20) -> dict[str, Any]:
		page = max(page, 1)
		per_page = max(per_page, 1)
		query = {"is_active": True}
		total = self.collection.count_documents(query)
		cursor = (
			self.collection.find(query)
			.sort("created_at", -1)
			.skip((page - 1) * per_page)
			.limit(per_page)
		)
		return {
			"items": list(cursor),
			"page": page,
			"per_page": per_page,
			"total": total,
		}

	def deactivate_admin(self, admin_id: str) -> bool:
		oid = _as_object_id(admin_id)
		if oid is None:
			return False
		result = self.collection.update_one(
			{"_id": oid},
			{"$set": {"is_active": False, "updated_at": _now()}},
		)
		return result.modified_count > 0


class StudentDAO:
	def __init__(self, db: Database) -> None:
		self.db = db
		self.collection = db.students

	def create_student(
		self,
		student_id: str,
		email: str,
		password: str,
		name: str,
		department: str,
		year: int,
	) -> str:
		now = _now()
		doc = {
			"student_id": student_id,
			"email": email.lower().strip(),
			"password_hash": hash_password(password),
			"name": name,
			"department": department,
			"year": int(year),
			"enrolled_courses": [],
			"face_embeddings": [],
			"face_photo_path": None,
			"is_active": True,
			"created_at": now,
			"updated_at": now,
		}
		result = self.collection.insert_one(doc)
		return str(result.inserted_id)

	def get_by_student_id(self, student_id: str) -> dict[str, Any] | None:
		return self.collection.find_one({"student_id": student_id, "is_active": True})

	def get_by_email(self, email: str) -> dict[str, Any] | None:
		return self.collection.find_one({"email": email.lower().strip(), "is_active": True})

	def update_student(self, student_id: str, fields: dict[str, Any]) -> bool:
		if not fields:
			return False
		fields = dict(fields)
		fields["updated_at"] = _now()
		result = self.collection.update_one({"student_id": student_id}, {"$set": fields})
		return result.modified_count > 0

	def soft_delete_student(self, student_id: str) -> bool:
		result = self.collection.update_one(
			{"student_id": student_id},
			{"$set": {"is_active": False, "updated_at": _now()}},
		)
		return result.modified_count > 0

	def add_embedding(
		self,
		student_id: str,
		embedding_array: np.ndarray,
		quality_score: float,
		source: str,
	) -> bool:
		arr = np.asarray(embedding_array, dtype=np.float32)
		payload = {
			"embedding": Binary(arr.tobytes()),
			"quality_score": float(quality_score),
			"created_at": _now(),
			"source": source,
		}
		result = self.collection.update_one(
			{"student_id": student_id, "is_active": True},
			{"$push": {"face_embeddings": payload}, "$set": {"updated_at": _now()}},
		)
		return result.modified_count > 0

	def get_embeddings(self, student_id: str) -> list[dict[str, Any]]:
		student = self.collection.find_one({"student_id": student_id, "is_active": True}, {"face_embeddings": 1})
		if not student:
			return []

		output: list[dict[str, Any]] = []
		for item in student.get("face_embeddings", []):
			emb = np.frombuffer(item["embedding"], dtype=np.float32)
			output.append(
				{
					"embedding": emb,
					"quality_score": item.get("quality_score"),
					"created_at": item.get("created_at"),
					"source": item.get("source"),
				}
			)
		return output

	def get_roster_embeddings(self, course_id: str) -> list[dict[str, Any]]:
		oid = _as_object_id(course_id)
		if oid is None:
			return []
		course = self.db.courses.find_one({"_id": oid})
		if not course:
			return []

		student_ids = course.get("enrolled_students", [])
		cursor = self.collection.find(
			{"_id": {"$in": student_ids}, "is_active": True},
			{"student_id": 1, "name": 1, "face_embeddings": 1},
		)

		results: list[dict[str, Any]] = []
		for student in cursor:
			for emb in student.get("face_embeddings", []):
				results.append(
					{
						"student_id": student.get("student_id"),
						"name": student.get("name"),
						"embedding": np.frombuffer(emb["embedding"], dtype=np.float32),
						"quality_score": emb.get("quality_score"),
					}
				)
		return results

	def search_students(
		self,
		query: str | None = None,
		department: str | None = None,
		year: int | None = None,
		page: int = 1,
		per_page: int = 20,
	) -> dict[str, Any]:
		page = max(page, 1)
		per_page = max(per_page, 1)

		filters: dict[str, Any] = {"is_active": True}
		if query:
			regex = {"$regex": query, "$options": "i"}
			filters["$or"] = [{"name": regex}, {"student_id": regex}, {"email": regex}]
		if department:
			filters["department"] = department
		if year is not None:
			filters["year"] = int(year)

		total = self.collection.count_documents(filters)
		items = list(
			self.collection.find(filters)
			.sort("created_at", -1)
			.skip((page - 1) * per_page)
			.limit(per_page)
		)
		return {"items": items, "page": page, "per_page": per_page, "total": total}

	def bulk_create_students(self, students_list: list[dict[str, Any]]) -> int:
		if not students_list:
			return 0

		now = _now()
		operations = []
		for student in students_list:
			operations.append(
				InsertOne(
					{
						"student_id": student["student_id"],
						"email": student["email"].lower().strip(),
						"password_hash": hash_password(student["password"]),
						"name": student["name"],
						"department": student["department"],
						"year": int(student["year"]),
						"enrolled_courses": [],
						"face_embeddings": [],
						"face_photo_path": None,
						"is_active": True,
						"created_at": now,
						"updated_at": now,
					}
				)
			)

		result = self.collection.bulk_write(operations, ordered=False)
		return int(result.inserted_count)


class CourseDAO:
	def __init__(self, db: Database) -> None:
		self.db = db
		self.collection = db.courses

	def create_course(
		self,
		course_code: str,
		course_name: str,
		department: str,
		instructor: str,
		schedule: list[dict[str, Any]],
	) -> str:
		now = _now()
		doc = {
			"course_code": course_code,
			"course_name": course_name,
			"department": department,
			"instructor": instructor,
			"schedule": schedule,
			"enrolled_students": [],
			"attendance_window_minutes": 15,
			"is_active": True,
			"created_at": now,
			"updated_at": now,
		}
		result = self.collection.insert_one(doc)
		return str(result.inserted_id)

	def get_by_code(self, course_code: str) -> dict[str, Any] | None:
		return self.collection.find_one({"course_code": course_code, "is_active": True})

	def get_by_id(self, course_id: str) -> dict[str, Any] | None:
		oid = _as_object_id(course_id)
		if oid is None:
			return None
		return self.collection.find_one({"_id": oid, "is_active": True})

	def update_course(self, course_id: str, fields: dict[str, Any]) -> bool:
		if not fields:
			return False
		oid = _as_object_id(course_id)
		if oid is None:
			return False
		fields = dict(fields)
		fields["updated_at"] = _now()
		result = self.collection.update_one({"_id": oid}, {"$set": fields})
		return result.modified_count > 0

	def enroll_student(self, course_id: str, student_id: str) -> bool:
		student = self.db.students.find_one({"student_id": student_id, "is_active": True})
		if not student:
			return False

		course_obj_id = _as_object_id(course_id)
		if course_obj_id is None:
			return False
		student_obj_id = student["_id"]

		self.collection.update_one({"_id": course_obj_id}, {"$addToSet": {"enrolled_students": student_obj_id}})
		self.db.students.update_one({"_id": student_obj_id}, {"$addToSet": {"enrolled_courses": course_obj_id}})
		return True

	def unenroll_student(self, course_id: str, student_id: str) -> bool:
		student = self.db.students.find_one({"student_id": student_id})
		if not student:
			return False

		course_obj_id = _as_object_id(course_id)
		if course_obj_id is None:
			return False
		student_obj_id = student["_id"]

		self.collection.update_one({"_id": course_obj_id}, {"$pull": {"enrolled_students": student_obj_id}})
		self.db.students.update_one({"_id": student_obj_id}, {"$pull": {"enrolled_courses": course_obj_id}})
		return True

	def get_enrolled_students(self, course_id: str, page: int = 1, per_page: int = 20) -> dict[str, Any]:
		page = max(page, 1)
		per_page = max(per_page, 1)

		oid = _as_object_id(course_id)
		if oid is None:
			return {"items": [], "total": 0, "page": page, "per_page": per_page}
		course = self.collection.find_one({"_id": oid})
		if not course:
			return {"items": [], "total": 0, "page": page, "per_page": per_page}

		ids = course.get("enrolled_students", [])
		total = len(ids)
		sliced = ids[(page - 1) * per_page : page * per_page]
		students = list(self.db.students.find({"_id": {"$in": sliced}, "is_active": True}))
		return {"items": students, "total": total, "page": page, "per_page": per_page}

	def list_courses(self, department: str | None = None, page: int = 1, per_page: int = 20) -> dict[str, Any]:
		page = max(page, 1)
		per_page = max(per_page, 1)
		filters: dict[str, Any] = {"is_active": True}
		if department:
			filters["department"] = department

		total = self.collection.count_documents(filters)
		items = list(
			self.collection.find(filters)
			.sort("created_at", -1)
			.skip((page - 1) * per_page)
			.limit(per_page)
		)
		return {"items": items, "total": total, "page": page, "per_page": per_page}

	def get_active_courses_for_student(self, student_id: str) -> list[dict[str, Any]]:
		student = self.db.students.find_one({"student_id": student_id, "is_active": True})
		if not student:
			return []
		return list(self.collection.find({"_id": {"$in": student.get("enrolled_courses", [])}, "is_active": True}))

	def get_current_session(self, course_id: str) -> dict[str, Any] | None:
		return self.db.attendance_sessions.find_one(
			{"course_id": _as_object_id(course_id), "status": "open"},
			sort=[("opened_at", -1)],
		)


class AttendanceDAO:
	def __init__(self, db: Database) -> None:
		self.db = db
		self.collection = db.attendance_records

	def record_attendance(
		self,
		student_id: str,
		course_id: str,
		status: str,
		confidence: float,
		liveness_score: float,
		verification_method: str = "face",
		check_date: str | None = None,
		check_in_time: datetime | None = None,
		ip_address: str | None = None,
		user_agent: str | None = None,
		session_id: str | None = None,
	) -> dict[str, Any]:
		student = self.db.students.find_one({"student_id": student_id, "is_active": True})
		if not student:
			return {"inserted": False, "reason": "student_not_found"}

		course_obj_id = _as_object_id(course_id)
		if course_obj_id is None:
			return {"inserted": False, "reason": "course_not_found"}
		date_value = check_date or _now().date().isoformat()
		check_in = check_in_time or _now()

		duplicate = self.collection.find_one(
			{
				"student_id": student["_id"],
				"course_id": course_obj_id,
				"date": date_value,
			}
		)
		if duplicate:
			return {"inserted": False, "reason": "duplicate"}

		doc = {
			"student_id": student["_id"],
			"course_id": course_obj_id,
			"date": date_value,
			"check_in_time": check_in,
			"status": status,
			"verification_method": verification_method,
			"confidence_score": float(confidence),
			"anti_spoofing_score": float(liveness_score),
			"ip_address": ip_address,
			"user_agent": user_agent,
			"session_id": session_id,
			"created_at": _now(),
		}
		result = self.collection.insert_one(doc)
		return {"inserted": True, "id": str(result.inserted_id)}

	def get_for_course(self, course_id: str, date_value: str) -> list[dict[str, Any]]:
		return list(
			self.collection.find(
				{
					"course_id": ObjectId(course_id),
					"date": date_value,
				}
			).sort("check_in_time", 1)
		)

	def get_for_student(
		self,
		student_id: str,
		start_date: str,
		end_date: str,
		course_id: str | None = None,
	) -> list[dict[str, Any]]:
		student = self.db.students.find_one({"student_id": student_id})
		if not student:
			return []

		filters: dict[str, Any] = {
			"student_id": student["_id"],
			"date": {"$gte": start_date, "$lte": end_date},
		}
		if course_id:
			oid = _as_object_id(course_id)
			if oid is None:
				return []
			filters["course_id"] = oid
		return list(self.collection.find(filters).sort("date", 1))

	def get_stats(self, course_id: str) -> dict[str, int]:
		pipeline = [
			{"$match": {"course_id": _as_object_id(course_id)}},
			{"$group": {"_id": "$status", "count": {"$sum": 1}}},
		]
		stats = {"present": 0, "late": 0, "absent": 0}
		for row in self.collection.aggregate(pipeline):
			key = row["_id"]
			if key in stats:
				stats[key] = int(row["count"])
		return stats

	def override_status(self, record_id: str, status: str) -> bool:
		oid = _as_object_id(record_id)
		if oid is None:
			return False
		result = self.collection.update_one(
			{"_id": oid},
			{"$set": {"status": status, "updated_at": _now()}},
		)
		return result.modified_count > 0


class SessionDAO:
	def __init__(self, db: Database) -> None:
		self.collection = db.attendance_sessions

	def open_session(self, course_id: str, admin_id: str, settings: dict[str, Any]) -> dict[str, Any]:
		course_obj = _as_object_id(course_id)
		opened_by = _as_object_id(admin_id)
		if course_obj is None or opened_by is None:
			return {"opened": False, "reason": "invalid_id", "session_id": None}
		existing = self.collection.find_one({"course_id": course_obj, "status": "open"})
		if existing:
			return {"opened": False, "reason": "already_open", "session_id": str(existing["_id"])}

		doc = {
			"course_id": course_obj,
			"date": _now().date().isoformat(),
			"opened_by": opened_by,
			"opened_at": _now(),
			"closed_at": None,
			"status": "open",
			"settings": settings,
		}
		result = self.collection.insert_one(doc)
		return {"opened": True, "session_id": str(result.inserted_id)}

	def close_session(self, session_id: str) -> bool:
		oid = _as_object_id(session_id)
		if oid is None:
			return False
		result = self.collection.update_one(
			{"_id": oid, "status": "open"},
			{"$set": {"status": "closed", "closed_at": _now()}},
		)
		return result.modified_count > 0

	def get_open_session(self, course_id: str) -> dict[str, Any] | None:
		oid = _as_object_id(course_id)
		if oid is None:
			return None
		return self.collection.find_one({"course_id": oid, "status": "open"})


class LogDAO:
	def __init__(self, db: Database) -> None:
		self.collection = db.system_logs

	def log_event(
		self,
		event_type: str,
		actor_id: str | None,
		actor_type: str,
		details: dict[str, Any],
		ip_address: str | None = None,
	) -> str:
		doc = {
			"event_type": event_type,
			"actor_id": _as_object_id(actor_id) if actor_id else None,
			"actor_type": actor_type,
			"details": details,
			"ip_address": ip_address,
			"timestamp": _now(),
		}
		result = self.collection.insert_one(doc)
		return str(result.inserted_id)

	def get_recent_logs(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
		filters: dict[str, Any] = {}
		if event_type:
			filters["event_type"] = event_type
		return list(self.collection.find(filters).sort("timestamp", -1).limit(max(limit, 1)))

	def get_logs_by_actor(self, actor_id: str, limit: int = 100) -> list[dict[str, Any]]:
		oid = _as_object_id(actor_id)
		if oid is None:
			return []
		return list(
			self.collection.find({"actor_id": oid})
			.sort("timestamp", -1)
			.limit(max(limit, 1))
		)
