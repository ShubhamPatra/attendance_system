"""Unit tests for camera pipeline branches (_create_track / process behavior)."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_fr = types.SimpleNamespace(
        face_locations=lambda *args, **kwargs: [],
        face_encodings=lambda *args, **kwargs: [],
        face_landmarks=lambda *args, **kwargs: [],
        load_image_file=lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "face_recognition", fake_fr)
    import importlib, config
    importlib.reload(config)


class _FakeCap:
    def read(self):
        return False, None

    def isOpened(self):
        return False

    def release(self):
        return None


class _FakeTrack:
    def __init__(self, tid, frame, bbox):
        self.track_id = tid
        self.bbox = bbox
        self.identity = None
        self.liveness = (0, 0.0)
        self.is_spoof = False
        self.is_unknown = False
        self.frames_missing = 0


@patch("camera.cv2.VideoCapture", return_value=_FakeCap())
def test_create_track_uncertain_sets_unknown(mock_cap):
    import camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with patch("camera.FaceTrack", _FakeTrack), \
         patch("camera.check_liveness", return_value=(0, 0.2)), \
         patch("camera.tracker") as mock_tracker:
        cam = camera.Camera(0)
        trk = cam._create_track(frame, (10, 10, 80, 80), raw_frame=frame)

    assert trk.is_unknown is True
    mock_tracker.record_recognition.assert_called_with(False, False)


@patch("camera.cv2.VideoCapture", return_value=_FakeCap())
def test_create_track_antispoof_error_uses_degraded_mode(mock_cap):
    import camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    fake_encoding = np.ones((128,), dtype=np.float64)

    with patch("camera.FaceTrack", _FakeTrack), \
         patch("camera.check_liveness", return_value=(-1, 0.0)), \
         patch("camera.encode_face", return_value=fake_encoding), \
         patch("camera.recognize_face", return_value=("sid", "Alice", 0.95)):
        cam = camera.Camera(0)
        cam._handle_recognized = MagicMock()
        trk = cam._create_track(frame, (10, 10, 80, 80), raw_frame=frame)

    assert trk.identity is not None
    cam._handle_recognized.assert_called_once()
