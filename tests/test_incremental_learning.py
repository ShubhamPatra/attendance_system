"""
Tests for incremental learning (encoding appending) in face_engine.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import bson
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib
    import core.config as config
    importlib.reload(config)


def test_append_encoding_calls_database():
    import vision.face_engine as face_engine
    sid = bson.ObjectId()
    enc = np.random.rand(128).astype(np.float64)

    with patch("app_vision.face_engine.database") as mock_db, \
            patch.object(face_engine.encoding_cache, "add_encoding_to_student") as mock_add:
        mock_db.append_student_encoding.return_value = True
        result = face_engine.append_encoding(sid, enc)

    assert result is True
    mock_db.append_student_encoding.assert_called_once_with(sid, enc)
    mock_add.assert_called_once_with(sid, enc)


def test_append_encoding_handles_failure():
    import vision.face_engine as face_engine
    sid = bson.ObjectId()
    enc = np.random.rand(128).astype(np.float64)

    with patch("app_vision.face_engine.database") as mock_db, \
            patch.object(face_engine.encoding_cache, "add_encoding_to_student"):
        mock_db.append_student_encoding.side_effect = Exception("DB error")
        result = face_engine.append_encoding(sid, enc)

    assert result is False


def test_cache_add_encoding_to_student():
    import vision.face_engine as face_engine

    sid = bson.ObjectId()
    enc1 = np.random.rand(128).astype(np.float64)
    enc2 = np.random.rand(128).astype(np.float64)

    cache = face_engine.encoding_cache
    cache._ids = [sid]
    cache._names = ["Alice"]
    cache._encodings = [[enc1]]
    cache._rebuild_flat()

    cache.add_encoding_to_student(sid, enc2)

    assert len(cache._encodings[0]) == 2
    np.testing.assert_array_equal(cache._encodings[0][1], enc2)


def test_cache_add_encoding_caps_at_max():
    import vision.face_engine as face_engine
    import core.config as config

    sid = bson.ObjectId()
    # Fill up to max
    existing = [np.random.rand(128).astype(np.float64) for _ in range(config.MAX_ENCODINGS_PER_STUDENT)]

    cache = face_engine.encoding_cache
    cache._ids = [sid]
    cache._names = ["Alice"]
    cache._encodings = [existing]
    cache._rebuild_flat()

    new_enc = np.random.rand(128).astype(np.float64)
    cache.add_encoding_to_student(sid, new_enc)

    assert len(cache._encodings[0]) == config.MAX_ENCODINGS_PER_STUDENT
    # The newest should be last
    np.testing.assert_array_equal(cache._encodings[0][-1], new_enc)


def test_cache_add_encoding_unknown_student():
    import vision.face_engine as face_engine

    sid = bson.ObjectId()
    enc = np.random.rand(128).astype(np.float64)

    cache = face_engine.encoding_cache
    cache._ids = []
    cache._names = []
    cache._encodings = []
    cache._rebuild_flat()

    with pytest.raises(ValueError):
        cache.add_encoding_to_student(sid, enc)


def test_cache_upsert_student_replaces_existing_entry():
    import vision.face_engine as face_engine

    sid = bson.ObjectId()
    enc1 = np.random.rand(128).astype(np.float64)
    enc2 = np.random.rand(128).astype(np.float64)

    cache = face_engine.encoding_cache
    cache._ids = [sid]
    cache._names = ["Alice"]
    cache._encodings = [[enc1]]
    cache._rebuild_flat()

    cache.upsert_student(sid, "Alicia", [enc2])

    ids, names, encodings = cache.get_all()
    assert ids == [sid]
    assert names == ["Alicia"]
    np.testing.assert_array_equal(encodings[0][0], enc2)
