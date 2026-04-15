"""MongoDB connection manager and index bootstrap helpers."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from core.config import Config

T = TypeVar("T")


def retry_operation(max_attempts: int = 3, base_delay: float = 1.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
	"""Retry Mongo operations using exponential backoff (1s, 2s, 4s by default)."""

	def decorator(func: Callable[..., T]) -> Callable[..., T]:
		def wrapper(*args: Any, **kwargs: Any) -> T:
			last_error: Exception | None = None
			for attempt in range(max_attempts):
				try:
					return func(*args, **kwargs)
				except PyMongoError as exc:
					last_error = exc
					if attempt == max_attempts - 1:
						break
					time.sleep(base_delay * (2**attempt))
			raise RuntimeError(f"MongoDB operation failed after {max_attempts} attempts") from last_error

		return wrapper

	return decorator


class MongoDBClient:
	"""Thread-safe singleton wrapper around pymongo.MongoClient."""

	_instance: MongoDBClient | None = None
	_lock = threading.Lock()

	def __new__(cls, config: Config) -> "MongoDBClient":
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = super().__new__(cls)
					cls._instance._initialized = False
		return cls._instance

	def __init__(self, config: Config) -> None:
		if getattr(self, "_initialized", False):
			return

		self._config = config
		self._client = MongoClient(
			self._config.MONGODB_URI,
			maxPoolSize=50,
			minPoolSize=5,
			serverSelectionTimeoutMS=5000,
			connectTimeoutMS=5000,
			socketTimeoutMS=10000,
			retryWrites=True,
		)
		self._db = self._client[self._config.MONGODB_DB_NAME]
		self._initialized = True

	@retry_operation()
	def get_db(self) -> Database:
		self._client.admin.command("ping")
		return self._db

	@retry_operation()
	def get_collection(self, name: str) -> Collection:
		db = self.get_db()
		return db[name]

	@retry_operation()
	def health_check(self) -> dict[str, Any]:
		start = time.perf_counter()
		self._client.admin.command("ping")
		elapsed = (time.perf_counter() - start) * 1000
		return {"status": "ok", "latency_ms": round(elapsed, 2)}

	def close(self) -> None:
		self._client.close()


def ensure_indexes(db: Database) -> None:
	"""Create all required indexes. Safe to call multiple times."""

	db.students.create_index("student_id", unique=True)
	db.students.create_index("email", unique=True)

	db.admins.create_index("email", unique=True)

	db.courses.create_index("course_code", unique=True)

	db.attendance_records.create_index([("course_id", ASCENDING), ("date", ASCENDING)])
	db.attendance_records.create_index([("student_id", ASCENDING), ("date", ASCENDING)])
	db.attendance_records.create_index(
		[("student_id", ASCENDING), ("course_id", ASCENDING), ("date", ASCENDING)], unique=True
	)

	db.attendance_sessions.create_index([("course_id", ASCENDING), ("date", ASCENDING)])

	db.system_logs.create_index("timestamp", expireAfterSeconds=7776000)


_client_instance: MongoDBClient | None = None
_client_lock = threading.Lock()


def get_mongo_client(config: Config) -> MongoDBClient:
	global _client_instance
	if _client_instance is None:
		with _client_lock:
			if _client_instance is None:
				_client_instance = MongoDBClient(config)
	return _client_instance
