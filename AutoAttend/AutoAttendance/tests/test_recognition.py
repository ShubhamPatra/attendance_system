from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from recognition.aligner import FaceAligner, REFERENCE_LANDMARKS
from recognition.config import PERFORMANCE_PROFILES, RecognitionConfig, RecognitionMode
from recognition.detector import YuNetDetector
from recognition.embedder import ArcFaceEmbedder
from recognition.matcher import BALANCED_THRESHOLD, EmbeddingCache, FaceMatcher
from recognition.pipeline import RecognitionPipeline
from recognition.detector import DetectionResult
from recognition.tracker import CSRTTracker


class _FakeDetector:
	def __init__(self, faces):
		self.faces = faces
		self.input_sizes = []

	def setInputSize(self, size):
		self.input_sizes.append(size)

	def detect(self, _frame):
		return 1, self.faces


def test_yunet_detector_raises_when_model_missing():
	with pytest.raises(FileNotFoundError):
		YuNetDetector(model_path="does-not-exist.onnx")


def test_yunet_detector_remaps_coordinates(monkeypatch, tmp_path):
	model_path = tmp_path / "yunet.onnx"
	model_path.write_bytes(b"model")

	faces = np.array(
		[
			[
				10.0,
				20.0,
				50.0,
				40.0,
				11.0,
				21.0,
				20.0,
				22.0,
				30.0,
				23.0,
				40.0,
				24.0,
				50.0,
				25.0,
				0.95,
			]
		],
		dtype=np.float32,
	)
	fake_detector = _FakeDetector(faces)

	class _Factory:
		@staticmethod
		def create(*_args, **_kwargs):
			return fake_detector

	monkeypatch.setattr("recognition.detector.cv2.FaceDetectorYN", _Factory)

	detector = YuNetDetector(str(model_path), min_face_size=20, processing_size=(320, 240))
	frame = np.zeros((480, 640, 3), dtype=np.uint8)
	results = detector.detect(frame)

	assert len(results) == 1
	assert fake_detector.input_sizes[-1] == (320, 240)
	assert results[0].bbox == (20, 40, 100, 80)
	assert results[0].landmarks[0] == (22, 42)
	assert results[0].landmarks[-1] == (100, 50)
	assert results[0].confidence == pytest.approx(0.95, abs=1e-6)


def test_yunet_detector_filters_small_faces(monkeypatch, tmp_path):
	model_path = tmp_path / "yunet.onnx"
	model_path.write_bytes(b"model")

	faces = np.array(
		[
			[5.0, 5.0, 10.0, 12.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 0.9],
			[30.0, 40.0, 80.0, 90.0, 35.0, 45.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 0.92],
		],
		dtype=np.float32,
	)
	fake_detector = _FakeDetector(faces)

	class _Factory:
		@staticmethod
		def create(*_args, **_kwargs):
			return fake_detector

	monkeypatch.setattr("recognition.detector.cv2.FaceDetectorYN", _Factory)

	detector = YuNetDetector(str(model_path), min_face_size=60, processing_size=(320, 320))
	frame = np.zeros((320, 320, 3), dtype=np.uint8)
	results = detector.detect(frame)

	assert len(results) == 1
	assert results[0].bbox == (30, 40, 80, 90)


def test_yunet_detector_handles_empty_frame(monkeypatch, tmp_path):
	model_path = tmp_path / "yunet.onnx"
	model_path.write_bytes(b"model")

	fake_detector = _FakeDetector(np.empty((0, 15), dtype=np.float32))

	class _Factory:
		@staticmethod
		def create(*_args, **_kwargs):
			return fake_detector

	monkeypatch.setattr("recognition.detector.cv2.FaceDetectorYN", _Factory)

	detector = YuNetDetector(str(model_path))
	assert detector.detect(np.array([])) == []


@pytest.mark.parametrize("fixture_name", ["sample_face_real_path", "sample_face_spoof_path"])
def test_yunet_detector_uses_sample_face_images(monkeypatch, tmp_path, request, fixture_name):
	model_path = tmp_path / "yunet.onnx"
	model_path.write_bytes(b"model")
	frame_path = request.getfixturevalue(fixture_name)
	frame = cv2.imread(str(frame_path))
	assert frame is not None

	faces = np.array(
		[[
			12.0,
			18.0,
			48.0,
			52.0,
			14.0,
			20.0,
			26.0,
			22.0,
			34.0,
			24.0,
			42.0,
			26.0,
			50.0,
			28.0,
			0.93,
		]],
		dtype=np.float32,
	)
	fake_detector = _FakeDetector(faces)

	class _Factory:
		@staticmethod
		def create(*_args, **_kwargs):
			return fake_detector

	monkeypatch.setattr("recognition.detector.cv2.FaceDetectorYN", _Factory)

	detector = YuNetDetector(str(model_path), min_face_size=20, processing_size=(320, 320))
	results = detector.detect(frame)

	assert len(results) == 1
	assert fake_detector.input_sizes[-1] == (320, 320)
	assert results[0].confidence == pytest.approx(0.93, abs=1e-6)
	assert len(results[0].landmarks) == 5
	assert results[0].bbox[2] >= 20
	assert results[0].bbox[3] >= 20


def test_yunet_detector_lazy_loads_on_first_detect(monkeypatch, tmp_path):
	model_path = tmp_path / "yunet.onnx"
	model_path.write_bytes(b"model")
	created = {"count": 0}

	class _LazyFactory:
		@staticmethod
		def create(*_args, **_kwargs):
			created["count"] += 1
			return _FakeDetector(np.empty((0, 15), dtype=np.float32))

	monkeypatch.setattr("recognition.detector.cv2.FaceDetectorYN", _LazyFactory)
	detector = YuNetDetector(str(model_path))
	assert created["count"] == 0
	assert detector.detect(np.zeros((20, 20, 3), dtype=np.uint8)) == []
	assert created["count"] == 1


class _FakeTracker:
	def __init__(self, updates):
		self._updates = list(updates)

	def init(self, _frame, _bbox):
		return True

	def update(self, _frame):
		if self._updates:
			return self._updates.pop(0)
		return False, (0.0, 0.0, 0.0, 0.0)


def test_csrt_tracker_basic_update_success():
	tracker = CSRTTracker(tracker_factory=lambda: _FakeTracker([(True, (10.2, 20.3, 30.4, 40.5))]))
	frame = np.zeros((100, 100, 3), dtype=np.uint8)

	assert tracker.init_track(frame, (10, 20, 30, 40)) is True
	ok, bbox = tracker.update(frame)
	assert ok is True
	assert bbox == (10, 20, 30, 40)


def test_csrt_tracker_resets_after_failures():
	tracker = CSRTTracker(max_failures=2, tracker_factory=lambda: _FakeTracker([(False, (0, 0, 0, 0)), (False, (0, 0, 0, 0))]))
	frame = np.zeros((100, 100, 3), dtype=np.uint8)

	tracker.init_track(frame, (10, 20, 30, 40))
	assert tracker.update(frame) == (False, None)
	assert tracker.update(frame) == (False, None)
	assert tracker.update(frame) == (False, None)


def test_csrt_tracker_is_same_face_iou_threshold():
	tracker = CSRTTracker(iou_threshold=0.3, tracker_factory=lambda: _FakeTracker([(True, (10, 10, 20, 20))]))
	assert tracker.is_same_face((10, 10, 20, 20), (12, 12, 20, 20)) is True
	assert tracker.is_same_face((10, 10, 20, 20), (50, 50, 20, 20)) is False


def test_csrt_tracker_update_all_multi_trackers():
	frame = np.zeros((120, 120, 3), dtype=np.uint8)
	trackers = iter([
		_FakeTracker([(True, (1.0, 2.0, 10.0, 10.0))]),
		_FakeTracker([(False, (0.0, 0.0, 0.0, 0.0))]),
	])
	tracker = CSRTTracker(tracker_factory=lambda: next(trackers))

	tracker.init_track(frame, (1, 2, 10, 10), track_id="a")
	tracker.init_track(frame, (5, 6, 8, 8), track_id="b")

	results = tracker.update_all(frame)
	assert results["a"] == (True, (1, 2, 10, 10))
	assert results["b"] == (False, None)


def test_csrt_tracker_state_reports_lost_and_high_confidence():
	tracker = CSRTTracker(tracker_factory=lambda: _FakeTracker([(True, (1.0, 2.0, 10.0, 10.0))] * 6))
	frame = np.zeros((120, 120, 3), dtype=np.uint8)

	lost_state = tracker.get_state()
	assert lost_state["status"] == "lost"

	tracker.init_track(frame, (1, 2, 10, 10))
	for _ in range(5):
		tracker.update(frame)

	high_state = tracker.get_state()
	assert high_state["status"] == "high"
	assert high_state["confidence"] > 0.0


def test_face_aligner_returns_112x112_output():
	aligner = FaceAligner()
	frame = np.full((180, 180, 3), 120, dtype=np.uint8)
	landmarks = REFERENCE_LANDMARKS + np.array([20.0, 25.0], dtype=np.float32)

	aligned = aligner.align(frame, landmarks)

	assert aligned.shape == (112, 112, 3)
	assert aligned.dtype == np.uint8


def test_face_aligner_rejects_invalid_landmark_shape():
	aligner = FaceAligner()
	frame = np.zeros((180, 180, 3), dtype=np.uint8)

	with pytest.raises(ValueError, match="Landmarks must be shape"):
		aligner.align(frame, np.zeros((4, 2), dtype=np.float32))


def test_face_aligner_rejects_large_alignment_error():
	aligner = FaceAligner(max_alignment_error=5.0)
	frame = np.zeros((180, 180, 3), dtype=np.uint8)

	bad_landmarks = np.array(
		[
			[5.0, 5.0],
			[175.0, 10.0],
			[10.0, 170.0],
			[170.0, 165.0],
			[90.0, 90.0],
		],
		dtype=np.float32,
	)

	with pytest.raises(ValueError, match="Alignment error too high"):
		aligner.align(frame, bad_landmarks)


class _FakeInput:
	def __init__(self, name: str):
		self.name = name


class _FakeSession:
	def __init__(self, *_args, **_kwargs):
		self._inputs = [_FakeInput("input_0")]
		self.last_feed = None

	def get_inputs(self):
		return self._inputs

	def run(self, _outputs, feed):
		self.last_feed = feed
		batch = next(iter(feed.values()))
		rows = batch.shape[0]
		base = np.arange(1, 513, dtype=np.float32)
		return [np.stack([base + idx for idx in range(rows)], axis=0)]


def test_arcface_embedder_single_embedding_l2_normalized(monkeypatch, tmp_path):
	model_path = tmp_path / "arcface.onnx"
	model_path.write_bytes(b"dummy")
	ArcFaceEmbedder._session_cache.clear()

	monkeypatch.setattr("recognition.embedder.create_onnx_session", lambda *_args, **_kwargs: _FakeSession())

	embedder = ArcFaceEmbedder(str(model_path))
	face = np.full((112, 112, 3), 128, dtype=np.uint8)
	vector = embedder.get_embedding(face)

	assert vector.shape == (512,)
	assert vector.dtype == np.float32
	assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_arcface_embedder_lazy_loads_on_first_use(monkeypatch, tmp_path):
	model_path = tmp_path / "arcface.onnx"
	model_path.write_bytes(b"dummy")
	ArcFaceEmbedder._session_cache.clear()
	created = {"count": 0}

	def _factory(*_args, **_kwargs):
		created["count"] += 1
		return _FakeSession()

	monkeypatch.setattr("recognition.embedder.create_onnx_session", _factory)
	embedder = ArcFaceEmbedder(str(model_path), lazy_loading=True)
	assert created["count"] == 0
	embedder.get_embedding(np.full((112, 112, 3), 128, dtype=np.uint8))
	assert created["count"] == 1


def test_arcface_embedder_batch_embeddings(monkeypatch, tmp_path):
	model_path = tmp_path / "arcface.onnx"
	model_path.write_bytes(b"dummy")
	ArcFaceEmbedder._session_cache.clear()

	monkeypatch.setattr("recognition.embedder.create_onnx_session", lambda *_args, **_kwargs: _FakeSession())

	embedder = ArcFaceEmbedder(str(model_path))
	faces = [
		np.full((112, 112, 3), 120, dtype=np.uint8),
		np.full((128, 128, 3), 130, dtype=np.uint8),
	]
	vectors = embedder.get_embeddings(faces)

	assert vectors.shape == (2, 512)
	assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-5)
	assert np.linalg.norm(vectors[1]) == pytest.approx(1.0, abs=1e-5)
	assert float(np.dot(vectors[0], vectors[1])) > 0.5


def test_arcface_embedder_quality_score_bounds():
	score_high = ArcFaceEmbedder.get_quality_score(face_confidence=0.95, alignment_error=2.0)
	score_low = ArcFaceEmbedder.get_quality_score(face_confidence=0.2, alignment_error=30.0)

	assert 0.0 <= score_high <= 1.0
	assert 0.0 <= score_low <= 1.0
	assert score_high > score_low


def test_face_matcher_cosine_similarity_bounds():
	matcher = FaceMatcher()
	v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
	v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
	v3 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

	assert matcher.cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-6)
	assert matcher.cosine_similarity(v1, v3) == pytest.approx(-1.0, abs=1e-6)


def test_face_matcher_gallery_ranking_multi_embedding():
	matcher = FaceMatcher()
	probe = np.array([1.0, 0.0, 0.0], dtype=np.float32)
	gallery = {
		"student_a": [np.array([0.98, 0.02, 0.0], dtype=np.float32), np.array([0.8, 0.2, 0.0], dtype=np.float32)],
		"student_b": [np.array([0.4, 0.6, 0.0], dtype=np.float32)],
		"student_c": [np.array([-0.8, 0.2, 0.0], dtype=np.float32)],
	}

	results = matcher.match_against_gallery(probe, gallery, top_k=3)

	assert len(results) == 3
	assert results[0].student_id == "student_a"
	assert results[0].rank == 1
	assert results[0].similarity > results[1].similarity


def test_face_matcher_thresholds():
	assert FaceMatcher.is_match(BALANCED_THRESHOLD + 0.01, mode="BALANCED") is True
	assert FaceMatcher.is_match(BALANCED_THRESHOLD - 0.01, mode="BALANCED") is False


def test_embedding_cache_prewarm_and_invalidate():
	cache = EmbeddingCache(max_entries=2)

	loaded = cache.prewarm("course:1", lambda: {"student_x": np.array([[1.0, 0.0]], dtype=np.float32)})
	assert "student_x" in loaded
	assert cache.get("course:1") is not None

	cache.set("course:2", {"student_y": np.array([[0.0, 1.0]], dtype=np.float32)})
	cache.set("course:3", {"student_z": np.array([[0.0, -1.0]], dtype=np.float32)})
	assert cache.get("course:1") is None

	cache.invalidate("course:2")
	assert cache.get("course:2") is None


def test_recognition_config_apply_mode_and_from_env(monkeypatch):
	cfg = RecognitionConfig()
	cfg.apply_mode(RecognitionMode.FAST)
	assert cfg.detect_interval == PERFORMANCE_PROFILES[RecognitionMode.FAST]["detect_interval"]
	assert cfg.processing_size == 320
	assert cfg.frame_scale_width == 320
	assert cfg.frame_scale_height == 240

	monkeypatch.setenv("RECOGNITION_MODE", "ACCURATE")
	monkeypatch.setenv("RECOGNITION_EMBEDDING_INTERVAL", "1")
	loaded = RecognitionConfig.from_env()
	assert loaded.mode == RecognitionMode.ACCURATE
	assert loaded.detect_interval == 2
	assert loaded.embedding_interval == 1


def test_recognition_config_frame_scale_env_override(monkeypatch):
	monkeypatch.setenv("RECOGNITION_MODE", "BALANCED")
	monkeypatch.setenv("RECOGNITION_FRAME_SCALE_WIDTH", "352")
	monkeypatch.setenv("RECOGNITION_FRAME_SCALE_HEIGHT", "288")

	loaded = RecognitionConfig.from_env()
	assert loaded.frame_scale_width == 352
	assert loaded.frame_scale_height == 288


class _PipelineFakeDetector:
	def __init__(self):
		self.calls = 0
		self.last_shape = None

	def detect(self, frame):
		self.calls += 1
		self.last_shape = frame.shape[:2]
		return [
			DetectionResult(
				bbox=(15, 20, 40, 40),
				landmarks=[(20, 28), (37, 28), (29, 35), (22, 46), (35, 46)],
				confidence=0.99,
			)
		]


class _PipelineFakeTracker:
	def init_track(self, _frame, _bbox, track_id="default"):
		return True

	def update(self, _frame, track_id="default"):
		return True, (30, 40, 80, 80)


class _PipelineAdaptiveTracker:
	def init_track(self, _frame, _bbox, track_id="default"):
		return True

	def update(self, _frame, track_id="default"):
		return True, (30, 40, 80, 80)

	def get_state(self, track_id="default"):
		return {"status": "high", "confidence": 0.95, "bbox": (30, 40, 80, 80), "failure_count": 0}


class _PipelineFakeAligner:
	def align(self, _frame, _landmarks):
		return np.zeros((112, 112, 3), dtype=np.uint8)


class _PipelineFakeEmbedder:
	def get_embedding(self, _aligned):
		return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class _PipelineFakeStudentDAO:
	def get_roster_embeddings(self, _course_id):
		return [
			{"student_id": "S100", "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
			{"student_id": "S200", "embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
		]


def test_recognition_pipeline_sequence_returns_stable_identity():
	config = RecognitionConfig().apply_mode("BALANCED")
	config.detect_interval = 2
	config.embedding_interval = 1
	config.smoother_window = 5
	config.smoother_majority = 3
	config.frame_scale_width = 320
	config.frame_scale_height = 240

	detector = _PipelineFakeDetector()

	pipeline = RecognitionPipeline(
		config=config,
		detector=detector,
		tracker=_PipelineFakeTracker(),
		aligner=_PipelineFakeAligner(),
		embedder=_PipelineFakeEmbedder(),
		matcher=FaceMatcher(),
		student_dao=_PipelineFakeStudentDAO(),
	)

	loaded_count = pipeline.load_gallery("COURSE-1")
	assert loaded_count == 2

	frame = np.zeros((480, 640, 3), dtype=np.uint8)
	last = None
	for _ in range(6):
		last = pipeline.process_frame(frame)

	assert last is not None
	assert last["matched"] is True
	assert last["student_id"] == "S100"
	assert last["vote_count"] >= 3
	assert last["fps"] >= 0.0
	assert detector.last_shape == (240, 320)
	assert last["bbox"] == (30, 40, 80, 80)


def test_recognition_pipeline_dynamic_interval_skips_small_movement_detection():
	config = RecognitionConfig().apply_mode("BALANCED")
	config.embedding_interval = 1
	config.frame_scale_width = 320
	config.frame_scale_height = 240

	detector = _PipelineFakeDetector()
	pipeline = RecognitionPipeline(
		config=config,
		detector=detector,
		tracker=_PipelineAdaptiveTracker(),
		aligner=_PipelineFakeAligner(),
		embedder=_PipelineFakeEmbedder(),
		matcher=FaceMatcher(),
		student_dao=_PipelineFakeStudentDAO(),
	)
	pipeline.load_gallery("COURSE-1")

	frame = np.zeros((480, 640, 3), dtype=np.uint8)
	for _ in range(10):
		pipeline.process_frame(frame)

	assert detector.calls == 1


def test_pipeline_dynamic_interval_mapping():
	assert RecognitionPipeline._get_dynamic_interval({"status": "high"}) == 10
	assert RecognitionPipeline._get_dynamic_interval({"status": "uncertain"}) == 3
	assert RecognitionPipeline._get_dynamic_interval({"status": "lost"}) == 1
