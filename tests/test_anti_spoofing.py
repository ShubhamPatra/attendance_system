"""
Tests for deep learning anti-spoofing liveness detection.
All Silent-Face-Anti-Spoofing library components are mocked –
no GPU, model files, or webcam required.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    monkeypatch.setenv("LIVENESS_CONFIDENCE_THRESHOLD", "0.8")
    import importlib, config
    importlib.reload(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h=480, w=640):
    """Return a dummy BGR image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _setup_real_face():
    """Configure anti_spoofing internals to simulate a real face."""
    import anti_spoofing

    mock_predictor = MagicMock()
    mock_predictor.get_bbox.return_value = [10, 20, 100, 120]
    # label 1 (real) gets the highest score
    mock_predictor.predict.return_value = np.array([[0.1, 0.9, 0.0]])

    mock_cropper = MagicMock()
    mock_cropper.crop.return_value = np.zeros((80, 80, 3), dtype=np.uint8)

    anti_spoofing._predictor = mock_predictor
    anti_spoofing._cropper = mock_cropper
    anti_spoofing._parse_model_name = MagicMock(return_value=(80, 80, 1, 1.0))
    anti_spoofing._model_names = ["model_1"]

    return mock_predictor


def _setup_spoof_face():
    """Configure anti_spoofing internals to simulate a spoof attack."""
    import anti_spoofing

    mock_predictor = MagicMock()
    mock_predictor.get_bbox.return_value = [10, 20, 100, 120]
    # label 0 (spoof) gets the highest score
    mock_predictor.predict.return_value = np.array([[0.9, 0.1, 0.0]])

    mock_cropper = MagicMock()
    mock_cropper.crop.return_value = np.zeros((80, 80, 3), dtype=np.uint8)

    anti_spoofing._predictor = mock_predictor
    anti_spoofing._cropper = mock_cropper
    anti_spoofing._parse_model_name = MagicMock(return_value=(80, 80, 1, 1.0))
    anti_spoofing._model_names = ["model_1"]

    return mock_predictor


# ---------------------------------------------------------------------------
# Tests: real face
# ---------------------------------------------------------------------------

def test_real_face_returns_label_1():
    """A real face should return label == 1 with high confidence."""
    _setup_real_face()
    import anti_spoofing

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    assert label == 1
    assert confidence >= 0.8


def test_real_face_passes_threshold():
    """Real face with sufficient confidence should pass the config threshold."""
    _setup_real_face()
    import anti_spoofing
    import config

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    passes = label == 1 and confidence >= config.LIVENESS_CONFIDENCE_THRESHOLD
    assert passes


# ---------------------------------------------------------------------------
# Tests: spoof face
# ---------------------------------------------------------------------------

def test_spoof_face_returns_label_0():
    """A spoof image should return label == 0."""
    _setup_spoof_face()
    import anti_spoofing

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    assert label == 0


def test_spoof_face_does_not_pass_threshold():
    """A spoof result should not pass the liveness gate."""
    _setup_spoof_face()
    import anti_spoofing
    import config

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    passes = label == 1 and confidence >= config.LIVENESS_CONFIDENCE_THRESHOLD
    assert not passes


# ---------------------------------------------------------------------------
# Tests: no face detected
# ---------------------------------------------------------------------------

def test_no_face_returns_zero():
    """When no face is present, return (0, 0.0)."""
    import anti_spoofing

    mock_predictor = MagicMock()
    mock_predictor.get_bbox.return_value = None
    anti_spoofing._predictor = mock_predictor

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    assert label == 0
    assert confidence == 0.0


def test_empty_bbox_returns_zero():
    """An empty bbox list should also return (0, 0.0)."""
    import anti_spoofing

    mock_predictor = MagicMock()
    mock_predictor.get_bbox.return_value = []
    anti_spoofing._predictor = mock_predictor

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    assert label == 0
    assert confidence == 0.0


# ---------------------------------------------------------------------------
# Tests: confidence threshold enforcement
# ---------------------------------------------------------------------------

def test_low_confidence_real_does_not_pass():
    """Even label==1, if confidence < threshold, liveness should not pass."""
    import anti_spoofing
    import config

    mock_predictor = MagicMock()
    mock_predictor.get_bbox.return_value = [10, 20, 100, 120]
    # Low confidence for real (barely over spoof)
    mock_predictor.predict.return_value = np.array([[0.4, 0.6, 0.0]])

    anti_spoofing._predictor = mock_predictor
    anti_spoofing._cropper = MagicMock(
        crop=MagicMock(return_value=np.zeros((80, 80, 3)))
    )
    anti_spoofing._parse_model_name = MagicMock(return_value=(80, 80, 1, 1.0))
    anti_spoofing._model_names = ["model_1"]

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    assert label == 1
    assert confidence < config.LIVENESS_CONFIDENCE_THRESHOLD


def test_custom_threshold_from_env(monkeypatch):
    """LIVENESS_CONFIDENCE_THRESHOLD should be read from env."""
    monkeypatch.setenv("LIVENESS_CONFIDENCE_THRESHOLD", "0.95")
    import importlib, config
    importlib.reload(config)

    assert config.LIVENESS_CONFIDENCE_THRESHOLD == 0.95


# ---------------------------------------------------------------------------
# Tests: attendance not recorded on spoof
# ---------------------------------------------------------------------------

def test_attendance_not_recorded_when_spoof():
    """Verify that spoof detection prevents attendance marking."""
    _setup_spoof_face()
    import anti_spoofing
    import config

    label, confidence = anti_spoofing.check_liveness(_make_frame())

    should_mark = label == 1 and confidence >= config.LIVENESS_CONFIDENCE_THRESHOLD
    assert not should_mark, "Attendance must not be marked for spoofed faces"


def test_attendance_recorded_only_for_real():
    """Verify that real faces are allowed past the attendance gate."""
    _setup_real_face()
    import anti_spoofing
    import config

    label, confidence = anti_spoofing.check_liveness(_make_frame())

    should_mark = label == 1 and confidence >= config.LIVENESS_CONFIDENCE_THRESHOLD
    assert should_mark, "Attendance should be allowed for real faces"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

def test_exception_returns_safe_default():
    """If the predictor throws, return (-1, 0.0) – no crash."""
    import anti_spoofing

    mock_predictor = MagicMock()
    mock_predictor.get_bbox.side_effect = RuntimeError("model error")
    anti_spoofing._predictor = mock_predictor

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    assert label == -1
    assert confidence == 0.0


def test_uninitialised_raises_runtime_error():
    """Calling check_liveness before init_models should raise RuntimeError."""
    import anti_spoofing

    anti_spoofing._predictor = None

    with pytest.raises(RuntimeError, match="not initialised"):
        anti_spoofing.check_liveness(_make_frame())


# ---------------------------------------------------------------------------
# Tests: multi-model aggregation
# ---------------------------------------------------------------------------

def test_aggregation_across_two_models():
    """Predictions from multiple models should be aggregated correctly."""
    import anti_spoofing

    mock_predictor = MagicMock()
    mock_predictor.get_bbox.return_value = [10, 20, 100, 120]
    # Each model predicts slightly in favour of real
    mock_predictor.predict.return_value = np.array([[0.2, 0.8, 0.0]])

    mock_cropper = MagicMock()
    mock_cropper.crop.return_value = np.zeros((80, 80, 3), dtype=np.uint8)

    anti_spoofing._predictor = mock_predictor
    anti_spoofing._cropper = mock_cropper
    anti_spoofing._parse_model_name = MagicMock(return_value=(80, 80, 1, 1.0))
    anti_spoofing._model_names = ["model_1", "model_2"]

    label, confidence = anti_spoofing.check_liveness(_make_frame())
    assert label == 1
    # Confidence = aggregated_score[label] / num_models = (0.8+0.8) / 2 = 0.8
    assert abs(confidence - 0.8) < 0.01
