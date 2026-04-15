from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import tasks.embedding_tasks as embedding_tasks


class _FakeDetector:
	def __init__(self, detections):
		self._detections = detections

	def detect(self, _frame):
		return self._detections


class _FakeAligner:
	def align(self, _frame, _landmarks):
		return np.zeros((112, 112, 3), dtype=np.uint8)


class _FakeEmbedder:
	def get_embedding(self, _aligned):
		return np.ones((512,), dtype=np.float32)

	def get_quality_score(self, face_confidence, alignment_error):
		del alignment_error
		return float(face_confidence)


class _FakeStudentDAO:
	def __init__(self, stored=True):
		self.stored = stored

	def add_embedding(self, _student_id, _embedding, _quality, source):
		assert source == "upload_photo"
		return self.stored


def test_generate_embedding_returns_no_face(monkeypatch, tmp_path):
	image_path = tmp_path / "face.jpg"
	image_path.write_bytes(b"x")

	monkeypatch.setattr("tasks.embedding_tasks.cv2.imread", lambda _p: np.zeros((120, 120, 3), dtype=np.uint8))
	monkeypatch.setattr(
		"tasks.embedding_tasks._build_components",
		lambda: (_FakeDetector([]), _FakeAligner(), _FakeEmbedder()),
	)

	result = embedding_tasks.generate_embedding.run("S100", str(image_path))
	assert result["status"] == "failed"
	assert result["reason"] == "no_face"


def test_generate_embedding_success(monkeypatch, tmp_path):
	image_path = tmp_path / "face.jpg"
	image_path.write_bytes(b"x")

	detection = SimpleNamespace(bbox=(10, 10, 80, 80), landmarks=[(20, 20)] * 5, confidence=0.91)
	monkeypatch.setattr("tasks.embedding_tasks.cv2.imread", lambda _p: np.zeros((120, 120, 3), dtype=np.uint8))
	monkeypatch.setattr(
		"tasks.embedding_tasks._build_components",
		lambda: (_FakeDetector([detection]), _FakeAligner(), _FakeEmbedder()),
	)
	monkeypatch.setattr("tasks.embedding_tasks._get_student_dao", lambda: _FakeStudentDAO(stored=True))

	result = embedding_tasks.generate_embedding.run("S100", str(image_path))
	assert result["status"] == "success"
	assert result["student_id"] == "S100"
	assert result["quality"] == 0.91