from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import redis


def _utcnow() -> datetime:
	return datetime.now(tz=timezone.utc)


@dataclass
class VerificationSession:
	session_id: str
	student_id: str
	course_id: str
	status: str
	created_at: str
	updated_at: str
	expires_at: str
	frame_results: list[dict] = field(default_factory=list)
	max_frames: int = 150

	def to_json(self) -> str:
		return json.dumps(asdict(self), separators=(",", ":"))

	@classmethod
	def from_json(cls, payload: str) -> "VerificationSession":
		return cls(**json.loads(payload))


class SessionManager:
	def __init__(self, redis_client=None, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 300) -> None:
		self.ttl_seconds = int(ttl_seconds)
		self.redis = redis_client or redis.Redis.from_url(redis_url, decode_responses=True)

	@staticmethod
	def _key(session_id: str) -> str:
		return f"verification:session:{session_id}"

	def create_session(self, student_id: str, course_id: str) -> VerificationSession:
		now = _utcnow()
		expires = now + timedelta(seconds=self.ttl_seconds)
		session = VerificationSession(
			session_id=uuid4().hex,
			student_id=student_id,
			course_id=course_id,
			status="active",
			created_at=now.isoformat(),
			updated_at=now.isoformat(),
			expires_at=expires.isoformat(),
		)
		self.redis.set(self._key(session.session_id), session.to_json(), ex=self.ttl_seconds)
		return session

	def get_session(self, session_id: str) -> VerificationSession | None:
		payload = self.redis.get(self._key(session_id))
		if not payload:
			return None
		return VerificationSession.from_json(payload)

	def update_session(self, session_id: str, result: dict) -> VerificationSession | None:
		session = self.get_session(session_id)
		if session is None:
			return None
		if len(session.frame_results) >= session.max_frames:
			session.status = "max_frames_reached"
		else:
			session.frame_results.append(result)

		now = _utcnow()
		expires = now + timedelta(seconds=self.ttl_seconds)
		session.updated_at = now.isoformat()
		session.expires_at = expires.isoformat()
		self.redis.set(self._key(session.session_id), session.to_json(), ex=self.ttl_seconds)
		return session

	def delete_session(self, session_id: str) -> bool:
		deleted = self.redis.delete(self._key(session_id))
		return bool(deleted)
