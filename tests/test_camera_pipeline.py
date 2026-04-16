"""Unit tests for camera pipeline branches (_create_track / process behavior)."""

import os
import sys
import types
import time
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
    import importlib
    import core.config as config
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
        self.detector_misses = 0
        self.liveness_history = []
        self.spoof_hold_until = 0.0
        self.liveness_state = "init"
        self.verification_started_at = None
        self.last_seen_at = 0.0
        self.last_liveness_meta = None
        self.confidence_history = []
        self.motion_history = []
        self.face_center_history = []
        self.screen_history = []
        self.brightness_history = []
        self.contrast_history = []
        self.ppe_state = "none"
        self.ppe_confidence = 0.0
        self.ppe_history = []
        self.ppe_updated_at = 0.0
        self.state = "detecting"
        self.quality_reason = ""
        self.created_at = 0.0
        self.recognition_cycle_count = 0
        self.antispoof_cycle_count = 0


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_create_track_low_confidence_spoof_is_uncertain(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with patch("camera.camera.FaceTrack", _FakeTrack), \
         patch("camera.camera.analyze_liveness_frame", return_value={"label": 0, "confidence": 0.6, "screen_suspicious": False, "too_small": False, "mean_brightness": 120.0, "contrast_proxy": 40.0}), \
         patch("camera.camera.tracker") as mock_tracker:
        cam = camera.Camera(0)
        trk = cam._create_track(frame, (10, 10, 80, 80), raw_frame=frame)

    assert trk.is_spoof is False
    assert trk.is_unknown is True
    mock_tracker.record_recognition.assert_called_with(False, False)


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_create_track_label_two_sets_spoof(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with patch("camera.camera.FaceTrack", _FakeTrack), \
         patch("camera.camera.analyze_liveness_frame", return_value={"label": 2, "confidence": 0.9, "screen_suspicious": False, "too_small": False, "mean_brightness": 110.0, "contrast_proxy": 30.0}), \
         patch("camera.camera.tracker") as mock_tracker:
        cam = camera.Camera(0)
        trk = cam._create_track(frame, (10, 10, 80, 80), raw_frame=frame)

    assert trk.is_spoof is True
    assert trk.is_unknown is False
    mock_tracker.record_recognition.assert_called_with(False, False)


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_create_track_antispoof_error_is_fail_safe_unknown(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with patch("camera.camera.FaceTrack", _FakeTrack), \
         patch("camera.camera.analyze_liveness_frame", return_value={"label": -1, "confidence": 0.0, "screen_suspicious": False, "too_small": False, "mean_brightness": 100.0, "contrast_proxy": 20.0}), \
         patch("camera.camera.encode_face") as mock_encode, \
         patch("camera.camera.tracker") as mock_tracker:
        cam = camera.Camera(0)
        cam._handle_recognized = MagicMock()
        trk = cam._create_track(frame, (10, 10, 80, 80), raw_frame=frame)

    assert trk.identity is None
    assert trk.is_spoof is True
    assert trk.is_unknown is False
    cam._handle_recognized.assert_not_called()
    mock_encode.assert_not_called()
    mock_tracker.record_recognition.assert_called_with(False, False)


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_deduplicate_tracks_keeps_single_face_box(mock_cap):
    import camera.camera as camera

    class _TrackObj:
        def __init__(self, bbox, identity=None, frames_missing=0):
            self.bbox = bbox
            self.identity = identity
            self.frames_missing = frames_missing
            self.detector_misses = 0
            self.liveness_history = []
            self.spoof_hold_until = 0.0
            self.liveness_state = "init"
            self.ppe_state = "none"
            self.ppe_confidence = 0.0
            self.ppe_history = []
            self.ppe_updated_at = 0.0

    cam = camera.Camera(0)
    cam._tracks = [
        _TrackObj((10, 10, 80, 80), identity=("sid", "Alice", 0.9), frames_missing=0),
        _TrackObj((14, 12, 78, 82), identity=None, frames_missing=0),
    ]

    cam._deduplicate_tracks()

    assert len(cam._tracks) == 1
    assert cam._tracks[0].identity is not None


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_deduplicate_tracks_keeps_one_for_nested_boxes(mock_cap):
    import camera.camera as camera

    class _TrackObj:
        def __init__(self, bbox, identity=None, frames_missing=0):
            self.bbox = bbox
            self.identity = identity
            self.frames_missing = frames_missing
            self.detector_misses = 0
            self.liveness_history = []
            self.spoof_hold_until = 0.0
            self.liveness_state = "init"
            self.ppe_state = "none"
            self.ppe_confidence = 0.0
            self.ppe_history = []
            self.ppe_updated_at = 0.0

    cam = camera.Camera(0)
    cam._tracks = [
        _TrackObj((100, 100, 180, 220), identity=None, frames_missing=0),
        _TrackObj((125, 135, 70, 90), identity=None, frames_missing=0),
    ]

    cam._deduplicate_tracks()

    assert len(cam._tracks) == 1


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_temporal_non_real_votes_promote_to_spoof(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    trk = _FakeTrack(1, frame, (10, 10, 80, 80))
    trk.verification_started_at = time.monotonic() - 3.0

    with patch("camera.camera.analyze_liveness_frame", return_value={"label": 0, "confidence": 0.62, "screen_suspicious": False, "too_small": False, "mean_brightness": 105.0, "contrast_proxy": 25.0}):
        # Build enough history to trigger temporal spoof decision.
        for _ in range(3):
            state, score = cam._evaluate_track_liveness(trk, frame, (10, 10, 80, 80))

    assert state == "spoof"
    assert score >= 0.5


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_deduplicate_prefers_higher_confidence_identity(mock_cap):
    import camera.camera as camera

    class _TrackObj:
        def __init__(self, bbox, identity=None, frames_missing=0):
            self.bbox = bbox
            self.identity = identity
            self.frames_missing = frames_missing
            self.detector_misses = 0
            self.liveness_history = []
            self.spoof_hold_until = 0.0
            self.liveness_state = "init"
            self.ppe_state = "none"
            self.ppe_confidence = 0.0
            self.ppe_history = []
            self.ppe_updated_at = 0.0

    cam = camera.Camera(0)
    cam._tracks = [
        _TrackObj((100, 100, 120, 140), identity=("sid-1", "Alice", 0.58), frames_missing=0),
        _TrackObj((108, 108, 118, 138), identity=("sid-2", "Bob", 0.91), frames_missing=0),
    ]

    cam._deduplicate_tracks()

    assert len(cam._tracks) == 1
    assert cam._tracks[0].identity[1] == "Bob"


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_recognition_requires_consistent_confirmation(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    trk = _FakeTrack(1, frame, (10, 10, 80, 80))

    fake_encoding = np.random.rand(128).astype(np.float64)

    with patch("camera.camera.config.SMOOTHING_MIN_FRAMES", 1), \
         patch("camera.camera.config.RECOGNITION_CONFIRM_FRAMES", 2), \
         patch("camera.camera.config.RECOGNITION_STABILITY_MIN_HITS", 2), \
         patch("camera.camera._recognition_module") as mock_rec_mod_factory, \
         patch("camera.camera._face_engine_module") as mock_face_engine_factory:
        mock_rec_mod = MagicMock()
        mock_rec_mod.encode_face_with_reason.return_value = (fake_encoding, "")
        mock_rec_mod_factory.return_value = mock_rec_mod

        mock_face_engine = MagicMock()
        mock_face_engine.recognize_face.return_value = ("sid-1", "Alice", 0.78)
        mock_face_engine_factory.return_value = mock_face_engine

        cam._handle_recognized = MagicMock()

        cam._try_recognize_track(
            trk,
            frame=frame,
            raw_frame=frame,
            raw_bbox_xywh=(10, 10, 80, 80),
            liveness_conf=0.90,
        )
        cam._handle_recognized.assert_not_called()
        assert trk.identity is None

        cam._try_recognize_track(
            trk,
            frame=frame,
            raw_frame=frame,
            raw_bbox_xywh=(10, 10, 80, 80),
            liveness_conf=0.90,
        )
        cam._handle_recognized.assert_called_once()
        assert trk.identity is not None


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_track_identity_cache_skips_recompute_when_valid(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    trk = _FakeTrack(7, frame, (10, 10, 80, 80))

    fake_encoding = np.random.rand(128).astype(np.float64)

    with patch("camera.camera.config.SMOOTHING_MIN_FRAMES", 1), \
         patch("camera.camera.config.SMOOTHING_WINDOW", 3), \
         patch("camera.camera.config.RECOGNITION_CONFIRM_FRAMES", 2), \
         patch("camera.camera.config.RECOGNITION_STABILITY_MIN_HITS", 2), \
         patch("camera.camera.config.RECOGNITION_STABILITY_WINDOW", 3), \
         patch("camera.camera.config.RECOGNITION_TRACK_CACHE_TTL_SECONDS", 5.0), \
         patch("camera.camera.config.BLINK_DETECTION_ENABLED", False), \
         patch("camera.camera._recognition_module") as mock_rec_mod_factory, \
         patch("camera.camera._face_engine_module") as mock_face_engine_factory:
        mock_rec_mod = MagicMock()
        mock_rec_mod.encode_face_with_reason.return_value = (fake_encoding, "")
        mock_rec_mod_factory.return_value = mock_rec_mod

        mock_face_engine = MagicMock()
        mock_face_engine.recognize_face.return_value = ("sid-1", "Alice", 0.78)
        mock_face_engine_factory.return_value = mock_face_engine

        cam._handle_recognized = MagicMock()

        cam._try_recognize_track(
            trk,
            frame=frame,
            raw_frame=frame,
            raw_bbox_xywh=(10, 10, 80, 80),
            liveness_conf=0.90,
        )
        cam._try_recognize_track(
            trk,
            frame=frame,
            raw_frame=frame,
            raw_bbox_xywh=(10, 10, 80, 80),
            liveness_conf=0.90,
        )

        assert mock_face_engine.recognize_face.call_count == 1
        cam._handle_recognized.assert_called_once()


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_track_identity_cache_recomputes_after_expiry(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    trk = _FakeTrack(11, frame, (10, 10, 80, 80))

    fake_encoding = np.random.rand(128).astype(np.float64)

    with patch("camera.camera.config.SMOOTHING_MIN_FRAMES", 1), \
         patch("camera.camera.config.SMOOTHING_WINDOW", 3), \
         patch("camera.camera.config.RECOGNITION_CONFIRM_FRAMES", 3), \
         patch("camera.camera.config.RECOGNITION_STABILITY_MIN_HITS", 3), \
         patch("camera.camera.config.RECOGNITION_STABILITY_WINDOW", 3), \
         patch("camera.camera.config.RECOGNITION_TRACK_CACHE_TTL_SECONDS", 5.0), \
         patch("camera.camera.config.BLINK_DETECTION_ENABLED", False), \
         patch("camera.camera._recognition_module") as mock_rec_mod_factory, \
         patch("camera.camera._face_engine_module") as mock_face_engine_factory:
        mock_rec_mod = MagicMock()
        mock_rec_mod.encode_face_with_reason.return_value = (fake_encoding, "")
        mock_rec_mod_factory.return_value = mock_rec_mod

        mock_face_engine = MagicMock()
        mock_face_engine.recognize_face.return_value = ("sid-1", "Alice", 0.78)
        mock_face_engine_factory.return_value = mock_face_engine

        cam._handle_recognized = MagicMock()

        cam._try_recognize_track(
            trk,
            frame=frame,
            raw_frame=frame,
            raw_bbox_xywh=(10, 10, 80, 80),
            liveness_conf=0.90,
        )
        cam._track_identity_cache[trk.track_id]["expires_at"] = 0.0
        cam._try_recognize_track(
            trk,
            frame=frame,
            raw_frame=frame,
            raw_bbox_xywh=(10, 10, 80, 80),
            liveness_conf=0.90,
        )

        assert mock_face_engine.recognize_face.call_count == 2


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_unknown_snapshot_logs_confidence_and_track(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    cam._last_unknown_save = 0.0
    cam._last_unknown_track_id = 42
    cam._last_unknown_confidence = 0.37

    with patch("camera.camera.os.makedirs") as mock_makedirs, \
         patch("camera.camera.cv2.imwrite", return_value=True), \
         patch("camera.camera._snapshot_executor") as mock_executor, \
         patch("camera.camera.logger") as mock_logger:
        future = MagicMock()
        def _run_now(func):
            func()
            return future

        mock_executor.submit.side_effect = _run_now
        cam._save_unknown_snapshot(frame)

    mock_makedirs.assert_called_once()
    mock_executor.submit.assert_called_once()
    assert mock_logger.info.call_args[0][0] == "Unknown face snapshot saved: %s (track=%s confidence=%.4f)"
    assert mock_logger.info.call_args[0][3] == 0.37


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_screen_spoof_heuristics_only_apply_without_blink_and_motion(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    trk = _FakeTrack(12, frame, (10, 10, 80, 80))
    trk.verification_started_at = time.monotonic() - 3.0

    # Blink evidence present -> do not trigger screen-spoof rejection.
    trk.blink_count = 1
    trk.motion_history = [0.0, 0.0, 0.0]
    with patch(
        "camera.camera.analyze_liveness_frame",
        return_value={
            "label": 1,
            "confidence": 0.9,
            "screen_suspicious": True,
            "too_small": False,
            "mean_brightness": 210.0,
            "contrast_proxy": 10.0,
        },
    ):
        state, _ = cam._evaluate_track_liveness(trk, frame, (10, 10, 80, 80))
    assert state != "spoof"

    # No blink + low motion -> screen spoof rule may reject.
    trk.blink_count = 0
    trk.motion_history = [0.0, 0.0, 0.0]
    trk.verification_started_at = time.monotonic() - 3.0
    with patch(
        "camera.camera.analyze_liveness_frame",
        return_value={
            "label": 1,
            "confidence": 0.9,
            "screen_suspicious": True,
            "too_small": False,
            "mean_brightness": 210.0,
            "contrast_proxy": 10.0,
        },
    ):
        state, _ = cam._evaluate_track_liveness(trk, frame, (10, 10, 80, 80))
    assert state == "spoof"


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_liveness_verification_timeout_resets_track_state(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    trk = _FakeTrack(13, frame, (10, 10, 80, 80))
    trk.verification_started_at = time.monotonic() - 10.0

    with patch(
        "camera.camera.analyze_liveness_frame",
        return_value={
            "label": 1,
            "confidence": 0.9,
            "screen_suspicious": False,
            "too_small": False,
            "mean_brightness": 130.0,
            "contrast_proxy": 35.0,
        },
    ):
        state, _ = cam._evaluate_track_liveness(trk, frame, (10, 10, 80, 80))

    assert state == "uncertain"
    assert trk.verification_started_at is not None
    assert trk.liveness_history == []


@patch("camera.camera.cv2.VideoCapture", return_value=_FakeCap())
def test_spoof_cooldown_temporal_override_blocks_immediate_retry(mock_cap):
    import camera.camera as camera

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cam = camera.Camera(0)
    trk = _FakeTrack(14, frame, (10, 10, 80, 80))
    trk.verification_started_at = time.monotonic() - 3.0
    trk.spoof_hold_until = time.monotonic() + 2.0

    with patch(
        "camera.camera.analyze_liveness_frame",
        return_value={
            "label": 1,
            "confidence": 0.95,
            "screen_suspicious": False,
            "too_small": False,
            "mean_brightness": 120.0,
            "contrast_proxy": 40.0,
        },
    ):
        state, _ = cam._evaluate_track_liveness(trk, frame, (10, 10, 80, 80))

    assert state == "spoof"
