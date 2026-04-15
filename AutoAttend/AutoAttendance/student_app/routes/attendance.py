from __future__ import annotations

import base64
import time
from datetime import date, timedelta

import cv2
import numpy as np
from flask import Blueprint, render_template, request
from flask_login import current_user, login_required
from flask_socketio import emit

from anti_spoofing.model import MiniFASNetAntiSpoof
from anti_spoofing.spoof_detector import SpoofDetector
from core.extensions import socketio
from core.models import AttendanceDAO
from recognition.config import RecognitionConfig
from recognition.pipeline import RecognitionPipeline
from verification.session import SessionManager
from verification.verifier import Verifier


attendance_bp = Blueprint("student_attendance", __name__)

_verification_sessions: dict[str, dict] = {}
_verifier_instance: Verifier | None = None
_SESSION_TIMEOUT_SECONDS = 30


class _InMemoryRedis:
	def __init__(self):
		self._store: dict[str, str] = {}

	def set(self, key, value, ex=None):
		del ex
		self._store[key] = value
		return True

	def get(self, key):
		return self._store.get(key)

	def delete(self, key):
		if key in self._store:
			del self._store[key]
			return 1
		return 0


def _get_verifier() -> Verifier:
	global _verifier_instance
	if _verifier_instance is not None:
		return _verifier_instance

	from student_app import get_student_dao

	student_dao = get_student_dao()
	db = student_dao.db

	config = RecognitionConfig.from_env()
	pipeline = RecognitionPipeline(config=config, student_dao=student_dao)
	spoof_model = MiniFASNetAntiSpoof(
		model_path_v2="models/anti_spoofing/2.7_80x80_MiniFASNetV2.onnx",
		model_path_v1se="models/anti_spoofing/4_0_0_80x80_MiniFASNetV1SE.onnx",
	)
	spoof_detector = SpoofDetector(threshold=0.5, model=spoof_model)
	attendance_dao = AttendanceDAO(db)

	session_manager = SessionManager(redis_client=_InMemoryRedis(), ttl_seconds=300)
	_verifier_instance = Verifier(
		session_manager=session_manager,
		pipeline=pipeline,
		spoof_detector=spoof_detector,
		attendance_dao=attendance_dao,
	)
	return _verifier_instance


def _resolve_course_id(student_id: str, payload: dict | None) -> str:
	req_course = (payload or {}).get("course_id")
	if req_course:
		return str(req_course)

	from student_app import get_student_dao

	student = get_student_dao().get_by_student_id(student_id)
	if student and student.get("enrolled_courses"):
		return str(student["enrolled_courses"][0])
	return "UNKNOWN"


def _map_intermediate_status(processed: dict) -> dict:
	recognition = processed.get("recognition", {})
	liveness = processed.get("liveness", {})
	rec_status = recognition.get("status")

	if rec_status in {"no_face", "lost"}:
		return {
			"status": "no_face",
			"face_detected": False,
			"liveness_check": False,
			"identity_verified": False,
			"attendance_marked": False,
		}

	if recognition.get("matched"):
		if liveness.get("is_real"):
			return {
				"status": "identity_verified",
				"face_detected": True,
				"liveness_check": True,
				"identity_verified": True,
				"attendance_marked": False,
				"student_id": recognition.get("student_id"),
				"vote_count": recognition.get("vote_count", 0),
			}
		return {
			"status": "liveness_check",
			"face_detected": True,
			"liveness_check": False,
			"identity_verified": False,
			"attendance_marked": False,
			"prompt": "Please blink",
		}

	return {
		"status": "face_detected",
		"face_detected": True,
		"liveness_check": bool(liveness.get("is_real")),
		"identity_verified": False,
		"attendance_marked": False,
	}


def _decode_frame(frame_payload: str) -> np.ndarray | None:
	if not frame_payload:
		return None
	if "," in frame_payload:
		_, encoded = frame_payload.split(",", 1)
	else:
		encoded = frame_payload
	try:
		raw = base64.b64decode(encoded)
		nparr = np.frombuffer(raw, dtype=np.uint8)
		frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		return frame
	except Exception:
		return None


@attendance_bp.get("/attendance/mark")
@login_required
def mark_attendance():
	return render_template("attendance/mark.html", student=current_user)


@attendance_bp.get("/attendance/history")
@login_required
def attendance_history():
	from student_app import get_student_dao

	dao = AttendanceDAO(get_student_dao().db)
	today = date.today()
	default_start = (today - timedelta(days=30)).isoformat()
	start_date = request.args.get("date_from", default_start)
	end_date = request.args.get("date_to", today.isoformat())
	course_id = request.args.get("course_id") or None
	page = max(int(request.args.get("page", 1)), 1)
	per_page = max(int(request.args.get("per_page", 20)), 1)

	records = dao.get_for_student(current_user.student_id, start_date, end_date, course_id)
	total = len(records)
	items = records[(page - 1) * per_page : page * per_page]

	course_ids = {item.get("course_id") for item in items}
	course_map = {}
	for oid in course_ids:
		if oid is None:
			continue
		course = get_student_dao().db.courses.find_one({"_id": oid})
		if course:
			course_map[oid] = course

	return render_template(
		"attendance/history.html",
		items=items,
		course_map=course_map,
		page=page,
		per_page=per_page,
		total=total,
		date_from=start_date,
		date_to=end_date,
		course_id=course_id or "",
	)


@attendance_bp.get("/attendance/status")
@login_required
def attendance_status():
	from student_app import get_student_dao

	db = get_student_dao().db
	student = db.students.find_one({"student_id": current_user.student_id, "is_active": True})
	if not student:
		return render_template("attendance/status.html", items=[], today=date.today().isoformat())

	today = date.today().isoformat()
	courses = list(db.courses.find({"_id": {"$in": student.get("enrolled_courses", [])}, "is_active": True}))
	items = []
	for course in courses:
		record = db.attendance_records.find_one(
			{
				"student_id": student["_id"],
				"course_id": course["_id"],
				"date": today,
			}
		)
		items.append(
			{
				"course": course,
				"status": record.get("status") if record else "not_marked",
				"check_in_time": record.get("check_in_time") if record else None,
			}
		)

	return render_template("attendance/status.html", items=items, today=today)


@socketio.on("start_verification")
def start_verification(payload=None):
	if not current_user.is_authenticated:
		emit("verification_status", {"status": "unauthorized"})
		return

	try:
		verifier = _get_verifier()
		course_id = _resolve_course_id(current_user.student_id, payload or {})
		session_id = verifier.start_session(current_user.student_id, course_id)
	except Exception as exc:
		emit("verification_status", {"status": "service_unavailable", "message": str(exc), "attendance_marked": False})
		return

	_verification_sessions[current_user.id] = {
		"started_at": time.time(),
		"session_id": session_id,
	}
	emit(
		"verification_status",
		{
			"status": "session_started",
			"face_detected": False,
			"liveness_check": False,
			"identity_verified": False,
			"attendance_marked": False,
		},
	)


@socketio.on("frame")
def process_frame(payload):
	if not current_user.is_authenticated:
		emit("verification_status", {"status": "unauthorized"})
		return

	session = _verification_sessions.get(current_user.id)
	if not session:
		emit("verification_status", {"status": "session_not_found"})
		return

	if (time.time() - session["started_at"]) > _SESSION_TIMEOUT_SECONDS:
		try:
			result = _get_verifier().finalize(session["session_id"])
		except Exception:
			result = None
		_verification_sessions.pop(current_user.id, None)
		payload = {"status": "timeout", "attendance_marked": False}
		if result is not None:
			payload["message"] = result.message
		emit("verification_status", payload)
		return

	frame = _decode_frame((payload or {}).get("frame", ""))
	if frame is None:
		emit("verification_status", {"status": "invalid_frame"})
		return

	try:
		verifier = _get_verifier()
		processed = verifier.process_frame(session["session_id"], frame)
	except Exception as exc:
		emit("verification_status", {"status": "processing_error", "message": str(exc), "attendance_marked": False})
		return

	if processed.get("status") != "processing":
		emit("verification_status", {"status": "processing_error", "message": processed.get("message", "unknown"), "attendance_marked": False})
		return

	intermediate = _map_intermediate_status(processed)
	emit("verification_status", intermediate)

	recognition = processed.get("recognition", {})
	liveness = processed.get("liveness", {})
	should_finalize = (
		recognition.get("matched")
		and recognition.get("student_id") == current_user.student_id
		and recognition.get("vote_count", 0) >= 3
		and bool(liveness.get("is_real"))
	)

	if should_finalize:
		final = verifier.finalize(session["session_id"])
		_verification_sessions.pop(current_user.id, None)
		if final.status == "success":
			emit(
				"verification_status",
				{
					"status": "attendance_marked",
					"face_detected": True,
					"liveness_check": True,
					"identity_verified": True,
					"attendance_marked": True,
					"confidence": final.confidence,
				},
			)
		else:
			emit(
				"verification_status",
				{
					"status": "verification_failed",
					"attendance_marked": False,
					"message": final.message,
				},
			)


@socketio.on("cancel_verification")
def cancel_verification(_payload=None):
	if current_user.is_authenticated:
		session = _verification_sessions.pop(current_user.id, None)
		if session:
			try:
				_get_verifier().session_manager.delete_session(session.get("session_id"))
			except Exception:
				pass
	emit("verification_status", {"status": "cancelled", "attendance_marked": False})
