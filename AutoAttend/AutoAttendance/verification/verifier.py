from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from verification.session import SessionManager


@dataclass(slots=True)
class VerificationResult:
	status: str
	confidence: float
	liveness_score: float
	message: str
	attendance_recorded: bool
	student_id: str | None = None


class Verifier:
	def __init__(
		self,
		session_manager: SessionManager,
		pipeline,
		spoof_detector,
		attendance_dao,
		event_emitter=None,
	) -> None:
		self.session_manager = session_manager
		self.pipeline = pipeline
		self.spoof_detector = spoof_detector
		self.attendance_dao = attendance_dao
		self.event_emitter = event_emitter

	def start_session(self, student_id: str, course_id: str) -> str:
		self.pipeline.load_gallery(course_id)
		session = self.session_manager.create_session(student_id=student_id, course_id=course_id)
		return session.session_id

	def process_frame(self, session_id: str, frame: np.ndarray) -> dict[str, Any]:
		session = self.session_manager.get_session(session_id)
		if session is None:
			return {"status": "invalid_session", "message": "Session not found"}

		recognition = self.pipeline.process_frame(frame)
		bbox = recognition.get("bbox") or (0, 0, frame.shape[1], frame.shape[0])
		landmarks = recognition.get("landmarks")

		liveness = self.spoof_detector.check_liveness(frame, bbox, landmarks)
		payload = {
			"recognition": recognition,
			"liveness": {
				"is_real": liveness.is_real,
				"score": liveness.score,
				"confidence_level": liveness.confidence_level,
				"details": liveness.details,
			},
		}
		self.session_manager.update_session(session_id, payload)
		return {"status": "processing", "session_id": session_id, **payload}

	def finalize(self, session_id: str) -> VerificationResult:
		session = self.session_manager.get_session(session_id)
		if session is None:
			return VerificationResult(
				status="failed",
				confidence=0.0,
				liveness_score=0.0,
				message="Session not found",
				attendance_recorded=False,
			)

		frames = session.frame_results
		if not frames:
			return VerificationResult(
				status="failed",
				confidence=0.0,
				liveness_score=0.0,
				message="Timeout: no frame results available",
				attendance_recorded=False,
			)

		recognition = self._aggregate_recognition(frames)
		if recognition is None:
			return VerificationResult(
				status="failed",
				confidence=0.0,
				liveness_score=0.0,
				message="No verified identity from recognition",
				attendance_recorded=False,
			)
		winner, recognition_confidence = recognition
		liveness_pass_rate, avg_liveness, liveness_ok = self._aggregate_liveness(frames)

		identity_matches = winner == session.student_id
		overall_confidence = float(min(recognition_confidence, avg_liveness))

		attendance_recorded = False
		status = "failed"
		message = "Verification failed"

		if identity_matches and liveness_ok:
			attendance_recorded = self._record_attendance(session, overall_confidence, avg_liveness)
			status = "success" if attendance_recorded else "failed"
			message = "Attendance marked" if attendance_recorded else "Attendance record not inserted"
			if attendance_recorded and callable(self.event_emitter):
				self.event_emitter(
					"verification_success",
					{
						"session_id": session.session_id,
						"student_id": session.student_id,
						"course_id": session.course_id,
						"confidence": overall_confidence,
						"liveness_score": avg_liveness,
					},
				)
		elif not identity_matches:
			message = "Identity mismatch"
		elif not liveness_ok:
			message = "Liveness criteria not satisfied"

		self.session_manager.delete_session(session_id)
		return VerificationResult(
			status=status,
			confidence=overall_confidence,
			liveness_score=avg_liveness,
			message=message,
			attendance_recorded=attendance_recorded,
			student_id=winner,
		)

	@staticmethod
	def _aggregate_recognition(frames: list[dict[str, Any]]) -> tuple[str, float] | None:
		recognized = [
			entry["recognition"].get("student_id")
			for entry in frames
			if entry.get("recognition", {}).get("matched") and entry.get("recognition", {}).get("student_id")
		]
		if not recognized:
			return None
		winner, winner_count = Counter(recognized).most_common(1)[0]
		confidence = float(winner_count / max(1, len(recognized)))
		return winner, confidence

	@staticmethod
	def _aggregate_liveness(frames: list[dict[str, Any]]) -> tuple[float, float, bool]:
		liveness_flags = [bool(entry.get("liveness", {}).get("is_real")) for entry in frames]
		liveness_scores = [float(entry.get("liveness", {}).get("score", 0.0)) for entry in frames]
		pass_rate = float(sum(liveness_flags) / max(1, len(liveness_flags)))
		avg_score = float(sum(liveness_scores) / max(1, len(liveness_scores)))
		return pass_rate, avg_score, pass_rate >= 0.6

	def _record_attendance(self, session, confidence: float, liveness_score: float) -> bool:
		insert_result = self.attendance_dao.record_attendance(
			student_id=session.student_id,
			course_id=session.course_id,
			status="present",
			confidence=confidence,
			liveness_score=liveness_score,
			verification_method="face",
			session_id=session.session_id,
		)
		return bool(insert_result.get("inserted"))
