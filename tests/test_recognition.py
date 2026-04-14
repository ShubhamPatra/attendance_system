"""
Tests for face recognition engine.
Tests use synthetic 128-D vectors — no webcam or images required.
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
    monkeypatch.setenv("RECOGNITION_MIN_CONFIDENCE", "0.60")
    import importlib
    import app_core.config as config
    importlib.reload(config)


@pytest.fixture
def loaded_cache():
    """Set up an EncodingCache with two known students (each with a list of encodings)."""
    import app_vision.face_engine as face_engine
    import bson

    enc1 = np.random.rand(128).astype(np.float64)
    enc2 = np.random.rand(128).astype(np.float64)
    id1 = bson.ObjectId()
    id2 = bson.ObjectId()

    cache = face_engine.encoding_cache
    cache._ids = [id1, id2]
    cache._names = ["Alice", "Bob"]
    cache._encodings = [[enc1], [enc2]]  # list-of-lists
    cache._rebuild_flat()  # populate flattened arrays for vectorised matching

    return cache, id1, id2, enc1, enc2


# ── Match known encoding ─────────────────────────────────────────────────

def test_recognize_known_face(loaded_cache):
    import app_vision.face_engine as face_engine

    cache, id1, id2, enc1, enc2 = loaded_cache

    # A probe very close to enc1 (add tiny noise)
    probe = enc1 + np.random.normal(0, 0.005, 128)

    result = face_engine.recognize_face(probe, threshold=0.55)
    assert result is not None
    student_id, name, confidence = result
    assert student_id == id1
    assert name == "Alice"
    assert confidence > 0.5


# ── Reject distant / unknown encoding ────────────────────────────────────

def test_reject_unknown_face(loaded_cache):
    import app_vision.face_engine as face_engine

    # A completely random encoding far from both stored
    probe = np.ones(128, dtype=np.float64) * 999.0

    result = face_engine.recognize_face(probe, threshold=0.55)
    assert result is None


# ── Threshold boundary ────────────────────────────────────────────────────

def test_threshold_boundary(loaded_cache):
    import app_vision.face_engine as face_engine

    cache, id1, id2, enc1, enc2 = loaded_cache

    # Same encoding should match with distance ≈ 0
    result_strict = face_engine.recognize_face(enc1, threshold=0.01)
    assert result_strict is not None  # distance ≈ 0, so 0 < 0.01

    # With threshold = 0.0, nothing should match (distance can't be < 0)
    result_zero = face_engine.recognize_face(
        enc1 + np.random.normal(0, 0.1, 128), threshold=0.0
    )
    assert result_zero is None


def test_reject_low_confidence_even_within_distance_threshold(loaded_cache):
    import app_vision.face_engine as face_engine

    cache, id1, id2, enc1, enc2 = loaded_cache
    # Distance ~= 0.46 gives confidence ~= 0.54, below min confidence 0.60.
    probe = enc1 + np.ones(128, dtype=np.float64) * (0.46 / np.sqrt(128.0))

    result = face_engine.recognize_face(probe, threshold=0.55)
    assert result is None


def test_reject_ambiguous_match_by_gap():
    import app_vision.face_engine as face_engine
    import bson

    base = np.zeros(128, dtype=np.float64)
    near = np.ones(128, dtype=np.float64) * 0.005

    cache = face_engine.encoding_cache
    cache._ids = [bson.ObjectId(), bson.ObjectId()]
    cache._names = ["Alice", "Bob"]
    cache._encodings = [[base], [near]]
    cache._rebuild_flat()

    probe = np.ones(128, dtype=np.float64) * 0.0025
    result = face_engine.recognize_face(probe, threshold=0.55)
    assert result is None


# ── Empty cache ───────────────────────────────────────────────────────────

def test_recognize_empty_cache():
    import app_vision.face_engine as face_engine
    import bson

    cache = face_engine.encoding_cache
    cache._ids = []
    cache._names = []
    cache._encodings = []
    cache._rebuild_flat()

    probe = np.random.rand(128).astype(np.float64)
    result = face_engine.recognize_face(probe)
    assert result is None


# ── Cache size property ───────────────────────────────────────────────────

def test_cache_size(loaded_cache):
    import app_vision.face_engine as face_engine
    cache = loaded_cache[0]
    assert cache.size == 2


# ── Cache get_all returns copies ──────────────────────────────────────────

def test_cache_get_all(loaded_cache):
    import app_vision.face_engine as face_engine
    cache, id1, id2, enc1, enc2 = loaded_cache

    ids, names, encodings = cache.get_all()
    assert len(ids) == 2
    assert names == ["Alice", "Bob"]


# ── generate_encoding mocked ─────────────────────────────────────────────

def test_generate_encoding_no_face():
    import app_vision.face_engine as face_engine

    with patch("app_vision.face_engine.face_recognition") as mock_fr:
        mock_fr.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_fr.face_locations.return_value = []  # no faces

        with pytest.raises(ValueError, match="No face detected"):
            face_engine.generate_encoding("fake_path.jpg")


def test_generate_encoding_multiple_faces():
    import app_vision.face_engine as face_engine

    with patch("app_vision.face_engine.face_recognition") as mock_fr:
        mock_fr.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_fr.face_locations.return_value = [(10, 50, 60, 10), (70, 120, 120, 70)]

        with pytest.raises(ValueError, match="Multiple faces"):
            face_engine.generate_encoding("fake_path.jpg")


def test_generate_encoding_success():
    import app_vision.face_engine as face_engine

    fake_enc = np.random.rand(128).astype(np.float64)
    with patch("app_vision.face_engine.face_recognition") as mock_fr:
        mock_fr.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_fr.face_locations.return_value = [(10, 50, 60, 10)]
        mock_fr.face_encodings.return_value = [fake_enc]

        result = face_engine.generate_encoding("fake_path.jpg")
        np.testing.assert_array_equal(result, fake_enc)
