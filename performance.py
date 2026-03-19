"""
Performance metrics tracker with auto-tuning.
"""

import threading
import time

import config
from utils import setup_logging

logger = setup_logging()


class PerformanceTracker:
    """Collects recognition metrics and auto-tunes threshold."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tp = 0  # true positives (known face, correctly matched)
        self._fp = 0  # false positives (unknown face, incorrectly matched)
        self._fn = 0  # false negatives (known face, not matched)
        self._tn = 0  # true negatives (unknown face, correctly rejected)
        self._frame_times: list[float] = []
        self._total_recognitions = 0
        self._last_tune_at = 0
        self._threshold = config.RECOGNITION_THRESHOLD

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
            self._total_recognitions += 1

            if self._total_recognitions - self._last_tune_at >= 200:
                self._auto_tune()
                self._last_tune_at = self._total_recognitions

    def record_frame_time(self, elapsed: float):
        with self._lock:
            self._frame_times.append(elapsed)
            # Keep last 500 measurements
            if len(self._frame_times) > 500:
                self._frame_times = self._frame_times[-500:]

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
                "accuracy_pct": round(accuracy, 2),
                "false_acceptance_rate_pct": round(far, 2),
                "false_rejection_rate_pct": round(frr, 2),
                "avg_frame_time_ms": round(avg_frame * 1000, 2),
                "fps": round(fps, 1),
                "current_threshold": self._threshold,
            }

    # ── auto-tuning ───────────────────────────────────────────────────────

    def _auto_tune(self):
        """Adjust threshold if accuracy drifts, clamped to configured bounds.

        The threshold is lowered when accuracy drops below 95 % (stricter
        matching) and raised when accuracy exceeds 98 % (allows more
        lenient matching).  It is always clamped to
        ``[RECOGNITION_THRESHOLD_MIN, RECOGNITION_THRESHOLD_MAX]``.
        """
        total = self._tp + self._fp + self._fn + self._tn
        if total == 0:
            return
        accuracy = (self._tp + self._tn) / total * 100

        old = self._threshold
        if accuracy < 95.0:
            self._threshold -= 0.02
        elif accuracy > 98.0:
            self._threshold += 0.01

        # Clamp to configured bounds
        self._threshold = max(
            config.RECOGNITION_THRESHOLD_MIN,
            min(self._threshold, config.RECOGNITION_THRESHOLD_MAX),
        )

        if self._threshold != old:
            config.RECOGNITION_THRESHOLD = self._threshold
            logger.warning(
                "Accuracy %.1f%%. Threshold adjusted: %.3f → %.3f "
                "(bounds [%.2f, %.2f])",
                accuracy, old, self._threshold,
                config.RECOGNITION_THRESHOLD_MIN,
                config.RECOGNITION_THRESHOLD_MAX,
            )

    @property
    def threshold(self) -> float:
        with self._lock:
            return self._threshold


# Singleton
tracker = PerformanceTracker()
