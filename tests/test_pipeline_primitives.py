"""Unit tests for pipeline geometry and association helpers."""

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
    import core.config as config
    importlib.reload(config)


def test_iou_overlap():
    import vision.pipeline as pipeline

    val = pipeline.iou((0, 0, 10, 10), (5, 5, 10, 10))
    assert val == pytest.approx(25 / 175)


def test_centroid_distance():
    import vision.pipeline as pipeline

    val = pipeline.centroid_distance((0, 0, 10, 10), (10, 0, 10, 10))
    assert val == pytest.approx(10.0)


def test_detect_motion_first_frame_is_true():
    import vision.pipeline as pipeline

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    changed, gray = pipeline.detect_motion(None, frame)
    assert changed is True
    assert gray.shape == (100, 100)


def test_detect_and_associate_uses_or_semantics():
    import vision.pipeline as pipeline

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class _Track:
        def __init__(self, bbox):
            self.bbox = bbox
            self.frames_missing = 3

    track = _Track((0, 0, 10, 10))

    with patch("app_vision.pipeline.detect_faces_yunet", return_value=[(95, 0, 10, 10), (12, 0, 10, 10)]):
        # First detection is far and non-overlapping (new).
        # Second detection is close in centroid distance (matched due to OR logic).
        new_boxes = pipeline.detect_and_associate(
            frame,
            [track],
            centroid_dist_threshold=30,
            iou_threshold=0.3,
        )

    assert new_boxes == [(95, 0, 10, 10)]
    assert track.frames_missing == 0


def test_detect_and_associate_detailed_returns_matched_track_indices():
    import vision.pipeline as pipeline

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class _Track:
        def __init__(self, bbox):
            self.bbox = bbox
            self.frames_missing = 2

    tracks = [_Track((0, 0, 10, 10)), _Track((70, 0, 10, 10))]

    with patch("app_vision.pipeline.detect_faces_yunet", return_value=[(5, 0, 10, 10)]):
        new_boxes, matched = pipeline.detect_and_associate_detailed(
            frame,
            tracks,
            centroid_dist_threshold=20,
            iou_threshold=0.3,
        )

    assert new_boxes == []
    assert matched == {0}
    assert tracks[0].frames_missing == 0
    assert tracks[1].frames_missing == 2
