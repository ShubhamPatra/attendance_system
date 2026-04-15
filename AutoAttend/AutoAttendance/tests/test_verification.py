from __future__ import annotations

import numpy as np

from anti_spoofing.spoof_detector import LivenessResult
from verification.session import SessionManager
from verification.verifier import Verifier


class _MockRedis:
	def __init__(self):
		self.store = {}

	def set(self, key, value, ex=None):
		del ex
		self.store[key] = value
		return True

	def get(self, key):
		return self.store.get(key)

	def delete(self, key):
		if key in self.store:
			del self.store[key]
			return 1
		return 0


def test_session_manager_create_get_update_delete():
	manager = SessionManager(redis_client=_MockRedis(), ttl_seconds=300)
	session = manager.create_session(student_id="S001", course_id="C001")

	loaded = manager.get_session(session.session_id)
	assert loaded is not None
	assert loaded.student_id == "S001"
	assert loaded.course_id == "C001"
	assert loaded.status == "active"

	updated = manager.update_session(session.session_id, {"recognition": "ok", "liveness": 0.9})
	assert updated is not None
	assert len(updated.frame_results) == 1

	assert manager.delete_session(session.session_id) is True
	assert manager.get_session(session.session_id) is None


def test_session_manager_max_frame_cap():
	manager = SessionManager(redis_client=_MockRedis(), ttl_seconds=300)
	session = manager.create_session(student_id="S002", course_id="C002")

	for index in range(151):
		manager.update_session(session.session_id, {"frame": index})

	loaded = manager.get_session(session.session_id)
	assert loaded is not None
	assert len(loaded.frame_results) == 150
	assert loaded.status == "max_frames_reached"


class _FakePipeline:
	def __init__(self, student_id: str):
		self.student_id = student_id

	def load_gallery(self, _course_id):
		return 1

	def process_frame(self, _frame):
		return {
			"matched": True,
			"student_id": self.student_id,
			"bbox": (0, 0, 100, 100),
			"status": "matched",
		}


class _FakeSpoofDetector:
	def __init__(self, is_real: bool = True, score: float = 0.9):
		self.is_real = is_real
		self.score = score

	def check_liveness(self, _frame, _bbox, _landmarks):
		return LivenessResult(
			is_real=self.is_real,
			score=self.score,
			confidence_level="HIGH" if self.is_real else "REJECTED",
			details={},
		)


class _FakeAttendanceDAO:
	def record_attendance(self, **_kwargs):
		return {"inserted": True}


def test_verifier_success_flow_marks_attendance():
	manager = SessionManager(redis_client=_MockRedis(), ttl_seconds=300)
	emitted = []

	def _emit(event, payload):
		emitted.append((event, payload))

	verifier = Verifier(
		session_manager=manager,
		pipeline=_FakePipeline(student_id="S100"),
		spoof_detector=_FakeSpoofDetector(is_real=True, score=0.92),
		attendance_dao=_FakeAttendanceDAO(),
		event_emitter=_emit,
	)

	session_id = verifier.start_session(student_id="S100", course_id="C100")
	for _ in range(5):
		verifier.process_frame(session_id, np.zeros((120, 120, 3), dtype=np.uint8))

	result = verifier.finalize(session_id)
	assert result.status == "success"
	assert result.attendance_recorded is True
	assert result.student_id == "S100"
	assert emitted
	assert emitted[0][0] == "verification_success"


def test_verifier_identity_mismatch_fails():
	manager = SessionManager(redis_client=_MockRedis(), ttl_seconds=300)
	verifier = Verifier(
		session_manager=manager,
		pipeline=_FakePipeline(student_id="S999"),
		spoof_detector=_FakeSpoofDetector(is_real=True, score=0.95),
		attendance_dao=_FakeAttendanceDAO(),
	)

	session_id = verifier.start_session(student_id="S100", course_id="C100")
	for _ in range(5):
		verifier.process_frame(session_id, np.zeros((120, 120, 3), dtype=np.uint8))

	result = verifier.finalize(session_id)
	assert result.status == "failed"
	assert result.attendance_recorded is False
	assert result.message == "Identity mismatch"


def test_verifier_spoof_rejection_fails():
	manager = SessionManager(redis_client=_MockRedis(), ttl_seconds=300)
	verifier = Verifier(
		session_manager=manager,
		pipeline=_FakePipeline(student_id="S100"),
		spoof_detector=_FakeSpoofDetector(is_real=False, score=0.1),
		attendance_dao=_FakeAttendanceDAO(),
	)

	session_id = verifier.start_session(student_id="S100", course_id="C100")
	for _ in range(5):
		verifier.process_frame(session_id, np.zeros((120, 120, 3), dtype=np.uint8))

	result = verifier.finalize(session_id)
	assert result.status == "failed"
	assert result.attendance_recorded is False
	assert result.message == "Liveness criteria not satisfied"


def test_verifier_timeout_no_frames_fails():
	manager = SessionManager(redis_client=_MockRedis(), ttl_seconds=300)
	verifier = Verifier(
		session_manager=manager,
		pipeline=_FakePipeline(student_id="S100"),
		spoof_detector=_FakeSpoofDetector(is_real=True, score=0.9),
		attendance_dao=_FakeAttendanceDAO(),
	)

	session_id = verifier.start_session(student_id="S100", course_id="C100")
	result = verifier.finalize(session_id)

	assert result.status == "failed"
	assert result.attendance_recorded is False
	assert "Timeout" in result.message
