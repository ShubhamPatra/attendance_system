from __future__ import annotations

import cv2
import numpy as np
import pytest

from anti_spoofing.blink_detector import BlinkDetector
from anti_spoofing.model import AntiSpoofResult
from anti_spoofing.movement_checker import MovementChecker
from anti_spoofing.model import MiniFASNetAntiSpoof
from anti_spoofing.spoof_detector import SpoofDetector


class _FakeInput:
	def __init__(self, name: str):
		self.name = name


class _FakeSession:
	def __init__(self, path: str, *_, **__):
		self.path = path
		self._inputs = [_FakeInput("input")]

	def get_inputs(self):
		return self._inputs

	def run(self, _outputs, _feed):
		if "V2" in self.path or "v2" in self.path:
			return [np.array([[0.1, 0.9]], dtype=np.float32)]
		return [np.array([[0.7, 0.3]], dtype=np.float32)]


def test_multiscale_crop_outputs_80x80(tmp_path, monkeypatch):
	model_v2 = tmp_path / "model_v2.onnx"
	model_v1 = tmp_path / "model_v1.onnx"
	model_v2.write_bytes(b"x")
	model_v1.write_bytes(b"x")
	monkeypatch.setattr("anti_spoofing.model.create_onnx_session", lambda path, **_kwargs: _FakeSession(path))

	spoof = MiniFASNetAntiSpoof(str(model_v2), str(model_v1))
	frame = np.zeros((120, 160, 3), dtype=np.uint8)
	crop = spoof._multi_scale_crop(frame, (20, 30, 40, 40), scale=2.7)

	assert crop.shape == (80, 80, 3)


def test_preprocess_shape_and_dtype(tmp_path, monkeypatch):
	model_v2 = tmp_path / "model_v2.onnx"
	model_v1 = tmp_path / "model_v1.onnx"
	model_v2.write_bytes(b"x")
	model_v1.write_bytes(b"x")
	monkeypatch.setattr("anti_spoofing.model.create_onnx_session", lambda path, **_kwargs: _FakeSession(path))

	spoof = MiniFASNetAntiSpoof(str(model_v2), str(model_v1))
	crop = np.full((80, 80, 3), 128, dtype=np.uint8)
	tensor = spoof._preprocess(crop)

	assert tensor.shape == (1, 3, 80, 80)
	assert tensor.dtype == np.float32


def test_predict_ensembles_model_scores(tmp_path, monkeypatch):
	model_v2 = tmp_path / "MiniFASNetV2.onnx"
	model_v1 = tmp_path / "MiniFASNetV1SE.onnx"
	model_v2.write_bytes(b"x")
	model_v1.write_bytes(b"x")
	monkeypatch.setattr("anti_spoofing.model.create_onnx_session", lambda path, **_kwargs: _FakeSession(path))

	spoof = MiniFASNetAntiSpoof(str(model_v2), str(model_v1), threshold=0.5)
	frame = np.zeros((200, 200, 3), dtype=np.uint8)
	result = spoof.predict(frame, (50, 50, 80, 80))

	assert result.model_scores["2.7"] > result.model_scores["4.0"]
	assert 0.0 <= result.score <= 1.0
	assert result.is_real is True


def test_missing_model_path_raises(tmp_path):
	model_v2 = tmp_path / "exists.onnx"
	model_v2.write_bytes(b"x")

	with pytest.raises(FileNotFoundError):
		MiniFASNetAntiSpoof(str(model_v2), str(tmp_path / "missing.onnx"))


def test_minifasnet_lazy_loads_sessions_on_first_predict(tmp_path, monkeypatch):
	model_v2 = tmp_path / "MiniFASNetV2.onnx"
	model_v1 = tmp_path / "MiniFASNetV1SE.onnx"
	model_v2.write_bytes(b"x")
	model_v1.write_bytes(b"x")
	created = {"count": 0}

	def _factory(path, **_kwargs):
		created["count"] += 1
		return _FakeSession(path)

	monkeypatch.setattr("anti_spoofing.model.create_onnx_session", _factory)
	spoof = MiniFASNetAntiSpoof(str(model_v2), str(model_v1))
	assert created["count"] == 0
	result = spoof.predict(np.zeros((120, 120, 3), dtype=np.uint8), (20, 20, 50, 50))
	assert created["count"] == 2
	assert isinstance(result.is_real, bool)


@pytest.mark.parametrize("fixture_name", ["sample_face_real_path", "sample_face_spoof_path"])
def test_minifasnet_predicts_using_sample_face_images(tmp_path, monkeypatch, request, fixture_name):
	model_v2 = tmp_path / "MiniFASNetV2.onnx"
	model_v1 = tmp_path / "MiniFASNetV1SE.onnx"
	model_v2.write_bytes(b"x")
	model_v1.write_bytes(b"x")
	monkeypatch.setattr("anti_spoofing.model.create_onnx_session", lambda path, **_kwargs: _FakeSession(path))

	image_path = request.getfixturevalue(fixture_name)
	frame = cv2.imread(str(image_path))
	assert frame is not None

	spoof = MiniFASNetAntiSpoof(str(model_v2), str(model_v1), threshold=0.5)
	result = spoof.predict(frame, (20, 20, 60, 60))

	assert isinstance(result.is_real, bool)
	assert 0.0 <= result.score <= 1.0
	assert set(result.model_scores) == {"2.7", "4.0"}
	assert result.model_scores["2.7"] != result.model_scores["4.0"]


def _eye_from_height(height: float) -> np.ndarray:
	return np.array(
		[
			[0.0, 0.0],
			[1.0, height],
			[2.0, height],
			[4.0, 0.0],
			[2.0, -height],
			[1.0, -height],
		],
		dtype=np.float32,
	)


def test_blink_detector_detects_blink_from_synthetic_sequence():
	detector = BlinkDetector(ear_threshold=0.21)

	open_eye = _eye_from_height(0.5)
	closed_eye = _eye_from_height(0.2)

	for _ in range(5):
		detector.update({"left_eye": open_eye, "right_eye": open_eye})
	for _ in range(3):
		detector.update({"left_eye": closed_eye, "right_eye": closed_eye})
	for _ in range(4):
		detector.update({"left_eye": open_eye, "right_eye": open_eye})

	result = detector.check()
	assert result.blink_count >= 1
	assert result.blink_detected is True


def test_blink_detector_no_blink_for_open_eyes_sequence():
	detector = BlinkDetector(ear_threshold=0.21)
	open_eye = _eye_from_height(0.5)

	for _ in range(20):
		detector.update({"left_eye": open_eye, "right_eye": open_eye})

	result = detector.check()
	assert result.blink_count == 0
	assert result.blink_detected is False


def test_movement_checker_classifies_static_sequence():
	checker = MovementChecker()
	for _ in range(20):
		checker.update((100, 100, 80, 80))

	result = checker.check()
	assert result.is_static is True
	assert result.is_natural is False


def test_movement_checker_classifies_natural_sequence():
	checker = MovementChecker()
	for step in range(30):
		checker.update((100 + (step % 10), 100 + (step % 8), 80, 80))

	result = checker.check()
	assert result.is_natural is True
	assert result.is_static is False


def test_movement_checker_challenge_verification():
	checker = MovementChecker(challenge_threshold=20.0)
	checker.update((100, 100, 80, 80))
	checker.start_challenge("right")
	checker.update((130, 100, 80, 80))
	assert checker.verify_challenge() is True

	checker_left = MovementChecker(challenge_threshold=20.0)
	checker_left.update((130, 100, 80, 80))
	checker_left.start_challenge("left")
	checker_left.update((100, 100, 80, 80))
	assert checker_left.verify_challenge() is True


class _FakeModel:
	def __init__(self, score: float, is_real: bool):
		self._score = score
		self._is_real = is_real

	def predict(self, _frame, _bbox):
		return AntiSpoofResult(is_real=self._is_real, score=self._score, model_scores={"2.7": self._score, "4.0": self._score})


class _FakeBlinkDetector:
	def __init__(self, blink_detected: bool, is_natural: bool):
		self._result = type("BlinkResult", (), {
			"blink_detected": blink_detected,
			"is_natural": is_natural,
			"blink_count": 1 if blink_detected else 0,
		})()

	def update(self, _landmarks):
		return 0.2

	def check(self):
		return self._result


class _FakeMovementChecker:
	def __init__(self, is_natural: bool, is_static: bool):
		self._result = type("MovementResult", (), {
			"is_natural": is_natural,
			"is_static": is_static,
			"displacement_std": 3.0 if is_natural else 0.5,
		})()

	def update(self, _bbox):
		return (0.0, 0.0)

	def check(self):
		return self._result


def test_spoof_detector_high_confidence_real_face():
	frame = np.full((120, 120, 3), 130, dtype=np.uint8)
	detector = SpoofDetector(
		threshold=0.5,
		model=_FakeModel(score=0.9, is_real=True),
		blink_detector=_FakeBlinkDetector(blink_detected=True, is_natural=True),
		movement_checker=_FakeMovementChecker(is_natural=True, is_static=False),
	)

	result = detector.check_liveness(frame, (20, 20, 60, 60), landmarks={"left_eye": _eye_from_height(0.5), "right_eye": _eye_from_height(0.5)})

	assert result.is_real is True
	assert result.confidence_level == "HIGH"
	assert result.score >= 0.8


def test_spoof_detector_rejects_spoof_case():
	frame = np.full((120, 120, 3), 40, dtype=np.uint8)
	detector = SpoofDetector(
		threshold=0.5,
		model=_FakeModel(score=0.2, is_real=False),
		blink_detector=_FakeBlinkDetector(blink_detected=False, is_natural=False),
		movement_checker=_FakeMovementChecker(is_natural=False, is_static=True),
	)

	result = detector.check_liveness(frame, (20, 20, 60, 60), landmarks={"left_eye": _eye_from_height(0.5), "right_eye": _eye_from_height(0.5)})

	assert result.is_real is False
	assert result.confidence_level == "REJECTED"
