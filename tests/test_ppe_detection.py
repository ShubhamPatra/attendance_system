"""Tests for PPE detection module (mask/cap)."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib
    import app_core.config as config

    importlib.reload(config)


def test_detect_ppe_returns_none_when_disabled(monkeypatch):
    import importlib
    import app_core.config as config

    monkeypatch.setenv("PPE_DETECTION_ENABLED", "0")
    importlib.reload(config)
    import app_vision.ppe_detection as ppe_detection

    importlib.reload(ppe_detection)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = ppe_detection.detect_ppe(frame, (10, 10, 50, 50))
    assert res["state"] == "none"
    assert res["confidence"] == 0.0


def test_detect_ppe_mask_state_with_mocked_model(monkeypatch):
    import importlib
    import app_core.config as config

    monkeypatch.setenv("PPE_DETECTION_ENABLED", "1")
    monkeypatch.setenv("PPE_MASK_THRESHOLD", "0.6")
    monkeypatch.setenv("PPE_CAP_THRESHOLD", "0.6")
    importlib.reload(config)
    import app_vision.ppe_detection as ppe_detection

    importlib.reload(ppe_detection)

    class _FakeNet:
        def setInput(self, _blob):
            return None

        def forward(self):
            # logits -> strong mask class in [none, mask, cap, both]
            return np.array([[0.1, 4.0, 0.2, 0.1]], dtype=np.float32)

    with patch("app_vision.ppe_detection._net", _FakeNet()):
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        res = ppe_detection.detect_ppe(frame, (20, 20, 60, 60))

    assert res["state"] == "mask"
    assert res["confidence"] >= 0.6


def test_detect_ppe_invalid_bbox_is_safe(monkeypatch):
    import importlib
    import app_core.config as config

    monkeypatch.setenv("PPE_DETECTION_ENABLED", "1")
    importlib.reload(config)
    import app_vision.ppe_detection as ppe_detection

    importlib.reload(ppe_detection)

    class _FakeNet:
        def setInput(self, _blob):
            return None

        def forward(self):
            return np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    with patch("app_vision.ppe_detection._net", _FakeNet()):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        res = ppe_detection.detect_ppe(frame, (200, 200, 10, 10))

    assert res["state"] == "none"
    assert res["confidence"] == 0.0
