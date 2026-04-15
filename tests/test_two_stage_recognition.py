"""Tests for two-stage recognition shortlist and candidate handoff behavior."""

from __future__ import annotations

import bson
import numpy as np



def _seed_cache(face_engine_module):
    """Populate encoding cache with a tiny deterministic dataset."""
    alice_id = bson.ObjectId()
    bob_id = bson.ObjectId()
    cara_id = bson.ObjectId()

    cache = face_engine_module.encoding_cache
    cache._ids = [alice_id, bob_id, cara_id]
    cache._names = ["Alice", "Bob", "Cara"]
    cache._encodings = [
        [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.95, 0.05, 0.0], dtype=np.float32),
        ],
        [np.array([0.0, 1.0, 0.0], dtype=np.float32)],
        [np.array([0.0, 0.0, 1.0], dtype=np.float32)],
    ]
    cache._rebuild_flat()
    return alice_id, bob_id, cara_id



def test_two_stage_stage1_to_stage2_match(monkeypatch):
    import app_vision.face_engine as face_engine

    monkeypatch.setattr(face_engine.config, "EMBEDDING_DIM", 3)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_TWO_STAGE_ENABLED", True)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE1_TOP_K", 1)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE1_MIN_SIMILARITY", 0.0)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE1_MARGIN", 0.05)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE2_MIN_CANDIDATES", 1)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_MIN_CONFIDENCE", 0.0)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_MIN_DISTANCE_GAP", 0.0)

    alice_id, _bob_id, _cara_id = _seed_cache(face_engine)

    probe = np.array([0.99, 0.02, 0.0], dtype=np.float32)
    result = face_engine.recognize_face(probe, threshold=0.1)

    assert result is not None
    student_id, name, _confidence = result
    assert student_id == alice_id
    assert name == "Alice"



def test_candidate_handoff_can_restrict_stage2(monkeypatch):
    import app_vision.face_engine as face_engine

    monkeypatch.setattr(face_engine.config, "EMBEDDING_DIM", 3)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_TWO_STAGE_ENABLED", True)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE2_MIN_CANDIDATES", 1)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_MIN_CONFIDENCE", 0.0)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_MIN_DISTANCE_GAP", 0.0)

    alice_id, bob_id, _cara_id = _seed_cache(face_engine)
    probe = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    wrong_candidate_result = face_engine.recognize_face(
        probe,
        threshold=0.8,
        candidate_student_ids=[bob_id],
    )
    assert wrong_candidate_result is None

    right_candidate_result = face_engine.recognize_face(
        probe,
        threshold=0.8,
        candidate_student_ids=[alice_id],
    )
    assert right_candidate_result is not None



def test_stage1_empty_shortlist_falls_back_to_full_search(monkeypatch):
    import app_vision.face_engine as face_engine

    monkeypatch.setattr(face_engine.config, "EMBEDDING_DIM", 3)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_TWO_STAGE_ENABLED", True)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE1_TOP_K", 1)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE1_MIN_SIMILARITY", 0.99999)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE1_MARGIN", 0.0)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_STAGE2_MIN_CANDIDATES", 1)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_MIN_CONFIDENCE", 0.0)
    monkeypatch.setattr(face_engine.config, "RECOGNITION_MIN_DISTANCE_GAP", 0.0)

    alice_id, _bob_id, _cara_id = _seed_cache(face_engine)
    probe = np.array([0.99, 0.0, 0.0], dtype=np.float32)

    result = face_engine.recognize_face(probe, threshold=0.1)
    assert result is not None
    assert result[0] == alice_id
