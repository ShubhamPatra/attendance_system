"""
Tests for face quality gate in recognition module.
"""

import os
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib
    import core.config as config
    importlib.reload(config)


def test_quality_gate_passes_good_image():
    from vision.recognition import check_face_quality_gate
    # Create a bright, textured image
    rng = np.random.RandomState(42)
    img = rng.randint(100, 200, (200, 200, 3), dtype=np.uint8)
    ok, reason = check_face_quality_gate(img, (20, 20, 100, 100))
    assert ok is True
    assert reason == ""


def test_quality_gate_rejects_blur():
    from vision.recognition import check_face_quality_gate
    # Solid gray - zero Laplacian variance
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, reason = check_face_quality_gate(img, (20, 20, 100, 100))
    assert ok is False
    assert "blurry" in reason.lower() or "blur" in reason.lower()


def test_quality_gate_rejects_dark():
    from vision.recognition import check_face_quality_gate
    # Very dark image with texture
    rng = np.random.RandomState(7)
    img = rng.randint(0, 20, (200, 200, 3), dtype=np.uint8)
    ok, reason = check_face_quality_gate(img, (20, 20, 100, 100))
    assert ok is False
    assert "dark" in reason.lower()


def test_quality_gate_rejects_bright():
    from vision.recognition import check_face_quality_gate
    import core.config as config
    # Build a bright image guaranteed to exceed configured max brightness.
    hi = min(255, int(config.BRIGHTNESS_MAX) + 8)
    lo = min(255, int(config.BRIGHTNESS_MAX) + 2)
    img = np.full((200, 200, 3), hi, dtype=np.uint8)
    # Add some texture to pass blur check
    img[::2, ::2] = lo
    ok, reason = check_face_quality_gate(img, (20, 20, 100, 100))
    assert ok is False
    assert "bright" in reason.lower()


def test_quality_gate_rejects_too_small_face():
    from vision.recognition import check_face_quality_gate
    import core.config as config

    img = np.random.randint(90, 180, (200, 200, 3), dtype=np.uint8)
    small = max(1, int(config.MIN_FACE_SIZE_PIXELS) - 8)
    ok, reason = check_face_quality_gate(img, (20, 20, small, small))
    assert ok is False
    assert "small" in reason.lower()


def test_quality_gate_rejects_tiny_face_area_ratio():
    from vision.recognition import check_face_quality_gate
    import core.config as config

    img = np.random.randint(90, 180, (400, 400, 3), dtype=np.uint8)
    tiny = int((config.MIN_FACE_AREA_RATIO * img.shape[0] * img.shape[1]) ** 0.5)
    tiny = max(1, tiny - 1)
    ok, reason = check_face_quality_gate(img, (20, 20, tiny, tiny))
    assert ok is False
    assert "small" in reason.lower()


def test_quality_gate_empty_roi():
    from vision.recognition import check_face_quality_gate
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # bbox completely outside image
    ok, reason = check_face_quality_gate(img, (300, 300, 10, 10))
    assert ok is False


def test_encode_face_rejects_poor_quality():
    """encode_face should return None for a blurry face."""
    from vision.recognition import encode_face
    # Solid gray image
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = encode_face(img, (20, 20, 100, 100))
    assert result is None


def test_dynamic_threshold_penalty_is_bounded(monkeypatch):
    import importlib
    import core.config as config
    import vision.preprocessing as preprocessing

    monkeypatch.setenv("DYNAMIC_THRESHOLD_ENABLED", "1")
    monkeypatch.setenv("DYNAMIC_THRESHOLD_MAX_PENALTY", "0.04")
    importlib.reload(config)
    importlib.reload(preprocessing)

    adjusted = preprocessing.compute_dynamic_threshold(
        0.60,
        {"is_blurry": True, "is_dark": True, "is_low_contrast": True},
    )
    assert adjusted == pytest.approx(0.56, abs=0.001)
