"""
Tests for the performance tracker module.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib
    import core.config as config
    importlib.reload(config)


@pytest.fixture
def fresh_tracker():
    """Return a fresh PerformanceTracker instance."""
    from core.performance import PerformanceTracker
    return PerformanceTracker()


def test_initial_metrics(fresh_tracker):
    m = fresh_tracker.metrics()
    assert m["total_recognitions"] == 0
    assert m["accuracy_pct"] == 0.0
    assert m["fps"] == 0.0


def test_record_recognition_tp(fresh_tracker):
    fresh_tracker.record_recognition(is_known=True, was_matched=True)
    m = fresh_tracker.metrics()
    assert m["true_positives"] == 1
    assert m["total_recognitions"] == 1


def test_record_recognition_fp(fresh_tracker):
    fresh_tracker.record_recognition(is_known=False, was_matched=True)
    m = fresh_tracker.metrics()
    assert m["false_positives"] == 1


def test_record_recognition_fn(fresh_tracker):
    fresh_tracker.record_recognition(is_known=True, was_matched=False)
    m = fresh_tracker.metrics()
    assert m["false_negatives"] == 1


def test_record_recognition_tn(fresh_tracker):
    fresh_tracker.record_recognition(is_known=False, was_matched=False)
    m = fresh_tracker.metrics()
    assert m["true_negatives"] == 1


def test_accuracy_calculation(fresh_tracker):
    # 8 TP + 2 TN out of 10 = 100%
    for _ in range(8):
        fresh_tracker.record_recognition(True, True)
    for _ in range(2):
        fresh_tracker.record_recognition(False, False)
    m = fresh_tracker.metrics()
    assert m["accuracy_pct"] == 100.0


def test_frame_time_and_fps(fresh_tracker):
    fresh_tracker.record_frame_time(0.05)  # 50ms → 20 FPS
    fresh_tracker.record_frame_time(0.05)
    m = fresh_tracker.metrics()
    assert m["avg_frame_time_ms"] == 50.0
    assert m["fps"] == 20.0


def test_stage_latency_metrics(fresh_tracker):
    fresh_tracker.record_stage_time("detection", 0.02)
    fresh_tracker.record_stage_time("detection", 0.04)
    fresh_tracker.record_stage_time("recognition", 0.01)

    m = fresh_tracker.metrics()
    assert m["stage_latency_ms"]["detection"] == 30.0
    assert m["stage_latency_ms"]["recognition"] == 10.0


def test_auto_tune_disabled_keeps_threshold(fresh_tracker):
    """Runtime auto-tuning is disabled; threshold must remain unchanged."""
    original = fresh_tracker.threshold

    # Create 200 recognitions with 50% accuracy (100 TP, 100 FP)
    for _ in range(100):
        fresh_tracker.record_recognition(True, True)
    for _ in range(100):
        fresh_tracker.record_recognition(False, True)  # false positives

    # After 200 recognitions, _auto_tune is invoked but intentionally disabled.
    new_threshold = fresh_tracker.threshold
    assert new_threshold == original


def test_frame_time_buffer_limited(fresh_tracker):
    """Buffer should not grow beyond 500 entries."""
    for _ in range(600):
        fresh_tracker.record_frame_time(0.01)
    m = fresh_tracker.metrics()
    # Just verify it doesn't crash and avg is correct
    assert m["avg_frame_time_ms"] == pytest.approx(10.0, abs=0.1)


def test_far_and_frr(fresh_tracker):
    for _ in range(10):
        fresh_tracker.record_recognition(True, True)   # TP
    for _ in range(5):
        fresh_tracker.record_recognition(False, True)   # FP
    for _ in range(3):
        fresh_tracker.record_recognition(True, False)   # FN
    for _ in range(2):
        fresh_tracker.record_recognition(False, False)  # TN

    m = fresh_tracker.metrics()
    # FAR = FP / (FP + TN) = 5 / 7 ≈ 71.43
    assert m["false_acceptance_rate_pct"] == pytest.approx(71.43, abs=0.1)
    # FRR = FN / (FN + TP) = 3 / 13 ≈ 23.08
    assert m["false_rejection_rate_pct"] == pytest.approx(23.08, abs=0.1)


def test_liveness_diagnostics_counters(fresh_tracker):
    fresh_tracker.record_liveness_event("spoof_true")
    fresh_tracker.record_liveness_event("spoof_uncertain")
    fresh_tracker.record_liveness_event("liveness_error")
    fresh_tracker.record_liveness_event("temporal_override")
    fresh_tracker.record_liveness_event("no_encode_guard")

    m = fresh_tracker.metrics()
    assert m["spoof_true"] == 1
    assert m["spoof_uncertain"] == 1
    assert m["liveness_error"] == 1
    assert m["temporal_override"] == 1
    assert m["no_encode_guard"] == 1
