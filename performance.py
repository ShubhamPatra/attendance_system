"""
Performance metrics tracker for face recognition system.
"""

import threading

import config
from utils import setup_logging

logger = setup_logging()


class PerformanceTracker:
    """Collects recognition metrics (TP, FP, FN, TN, frame times)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.threshold = config.RECOGNITION_THRESHOLD
        self._tp = 0  # true positives (known face, correctly matched)
        self._fp = 0  # false positives (unknown face, incorrectly matched)
        self._fn = 0  # false negatives (known face, not matched)
        self._tn = 0  # true negatives (unknown face, correctly rejected)
        self._liveness_counters = {
            "spoof_true": 0,
            "spoof_uncertain": 0,
            "liveness_error": 0,
            "temporal_override": 0,
            "no_encode_guard": 0,
            "spoof_detected": 0,
            "quality_rejection": 0,
            "tracker_expired": 0,
            "detector_pruned": 0,
            "liveness_uncertain": 0,
            "ppe_mask_detected": 0,
            "ppe_cap_detected": 0,
            "ppe_both_detected": 0,
            "ppe_model_error": 0,
        }
        self._frame_times: list[float] = []

    # ── recording ─────────────────────────────────────────────────────────

    def record_recognition(self, is_known: bool, was_matched: bool):
        """Record a single recognition outcome."""
        with self._lock:
            if is_known and was_matched:
                self._tp += 1
            elif not is_known and was_matched:
                self._fp += 1
            elif is_known and not was_matched:
                self._fn += 1
            else:
                self._tn += 1

    def record_frame_time(self, elapsed: float):
        with self._lock:
            self._frame_times.append(elapsed)
            # Keep last 500 measurements
            if len(self._frame_times) > 500:
                self._frame_times = self._frame_times[-500:]

    def record_liveness_event(self, event_name: str):
        """Record anti-spoof/liveness diagnostics events."""
        with self._lock:
            if event_name in self._liveness_counters:
                self._liveness_counters[event_name] += 1

    # ── metrics ───────────────────────────────────────────────────────────

    def metrics(self) -> dict:
        with self._lock:
            total = self._tp + self._fp + self._fn + self._tn
            accuracy = (
                (self._tp + self._tn) / total * 100 if total else 0.0
            )
            far = (
                self._fp / (self._fp + self._tn) * 100
                if (self._fp + self._tn)
                else 0.0
            )
            frr = (
                self._fn / (self._fn + self._tp) * 100
                if (self._fn + self._tp)
                else 0.0
            )
            avg_frame = (
                sum(self._frame_times) / len(self._frame_times)
                if self._frame_times
                else 0.0
            )
            fps = 1.0 / avg_frame if avg_frame > 0 else 0.0

            return {
                "total_recognitions": total,
                "true_positives": self._tp,
                "false_positives": self._fp,
                "false_negatives": self._fn,
                "true_negatives": self._tn,
                "recognition_threshold": round(float(self.threshold), 4),
                "accuracy_pct": round(accuracy, 2),
                "false_acceptance_rate_pct": round(far, 2),
                "false_rejection_rate_pct": round(frr, 2),
                "avg_frame_time_ms": round(avg_frame * 1000, 2),
                "fps": round(fps, 1),
                **self._liveness_counters,
            }


# Singleton
tracker = PerformanceTracker()
