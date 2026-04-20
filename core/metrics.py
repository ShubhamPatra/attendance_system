"""
Metrics aggregation and tracking for real-time performance monitoring.

PHASE 2: Real-time metrics tracking across cameras and pipeline stages.
Provides thread-safe aggregation of FPS, latencies, queue depth, frame drops,
and processing statistics for dashboard/API consumption.
"""

import threading
import time
from collections import deque
from typing import Dict, Any, Optional

import core.config as config


class CameraMetricsTracker:
    """Tracks and aggregates metrics for a single camera.
    
    Maintains rolling windows of:
    - FPS (frames per second)
    - Detection latency
    - Recognition latency
    - Anti-spoof/liveness latency
    - Queue depth
    - Frame drops
    - Processing time per frame
    
    Thread-safe: All access protected by locks.
    """

    def __init__(self, camera_id: int):
        self._camera_id = camera_id
        self._lock = threading.Lock()

        # Timing metrics (rolling windows)
        self._frame_times = deque(maxlen=300)  # Keep 10 seconds at 30 FPS
        self._detection_times = deque(maxlen=100)
        self._recognition_times = deque(maxlen=100)
        self._liveness_times = deque(maxlen=100)
        self._ppe_times = deque(maxlen=100)

        # Events
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._frame_drop_count = 0
        self._current_queue_depth = 0

        # Recognition stats
        self._recognition_count = 0
        self._recognition_success_count = 0
        self._spoof_detected_count = 0
        self._unknown_face_count = 0

        # Last update timestamp (for aggregation intervals)
        self._last_aggregation_time = time.monotonic()

    def record_frame_time(self, elapsed_ms: float) -> None:
        """Record time spent processing a single frame (milliseconds)."""
        with self._lock:
            self._frame_times.append(elapsed_ms)
            self._frame_count += 1

    def record_detection_time(self, elapsed_ms: float) -> None:
        """Record face detection latency (milliseconds)."""
        with self._lock:
            self._detection_times.append(elapsed_ms)

    def record_recognition_time(self, elapsed_ms: float) -> None:
        """Record face recognition latency (milliseconds)."""
        with self._lock:
            self._recognition_times.append(elapsed_ms)

    def record_liveness_time(self, elapsed_ms: float) -> None:
        """Record liveness/anti-spoof latency (milliseconds)."""
        with self._lock:
            self._liveness_times.append(elapsed_ms)

    def record_ppe_time(self, elapsed_ms: float) -> None:
        """Record PPE detection latency (milliseconds)."""
        with self._lock:
            self._ppe_times.append(elapsed_ms)

    def record_frame_drop(self) -> None:
        """Record a dropped frame."""
        with self._lock:
            self._frame_drop_count += 1

    def set_queue_depth(self, depth: int) -> None:
        """Set current event queue depth."""
        with self._lock:
            self._current_queue_depth = depth

    def record_recognition(self, success: bool) -> None:
        """Record a recognition attempt (success/failure)."""
        with self._lock:
            self._recognition_count += 1
            if success:
                self._recognition_success_count += 1

    def record_spoof_detected(self) -> None:
        """Record a spoof detection event."""
        with self._lock:
            self._spoof_detected_count += 1

    def record_unknown_face(self) -> None:
        """Record an unknown (unmatched) face."""
        with self._lock:
            self._unknown_face_count += 1

    def _safe_avg(self, deque_obj: deque) -> float:
        """Compute average of deque, or 0.0 if empty."""
        if not deque_obj:
            return 0.0
        return sum(deque_obj) / len(deque_obj)

    def _safe_max(self, deque_obj: deque) -> float:
        """Compute max of deque, or 0.0 if empty."""
        if not deque_obj:
            return 0.0
        return max(deque_obj)

    def _safe_min(self, deque_obj: deque) -> float:
        """Compute min of deque, or 0.0 if empty."""
        if not deque_obj:
            return 0.0
        return min(deque_obj)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of current metrics.
        
        Returns dict with:
        - Camera metadata (ID, uptime, frame count)
        - FPS and frame timing
        - Per-stage latencies (detection, recognition, liveness, PPE)
        - Queue and dropping stats
        - Recognition statistics
        """
        with self._lock:
            now = time.monotonic()
            uptime_seconds = now - self._start_time

            # FPS calculation
            fps = 0.0
            if len(self._frame_times) > 1:
                total_time = sum(self._frame_times)
                if total_time > 0:
                    fps = 1000.0 * len(self._frame_times) / total_time

            # Recognition success rate
            recognition_success_rate = 0.0
            if self._recognition_count > 0:
                recognition_success_rate = (
                    100.0 * self._recognition_success_count / self._recognition_count
                )

            return {
                "camera_id": self._camera_id,
                "uptime_seconds": round(uptime_seconds, 1),
                "total_frames": self._frame_count,
                "total_frame_drops": self._frame_drop_count,
                "current_queue_depth": self._current_queue_depth,
                "fps": round(fps, 1),
                "frame_time_ms": {
                    "avg": round(self._safe_avg(self._frame_times), 1),
                    "min": round(self._safe_min(self._frame_times), 1),
                    "max": round(self._safe_max(self._frame_times), 1),
                },
                "detection_latency_ms": {
                    "avg": round(self._safe_avg(self._detection_times), 1),
                    "min": round(self._safe_min(self._detection_times), 1),
                    "max": round(self._safe_max(self._detection_times), 1),
                },
                "recognition_latency_ms": {
                    "avg": round(self._safe_avg(self._recognition_times), 1),
                    "min": round(self._safe_min(self._recognition_times), 1),
                    "max": round(self._safe_max(self._recognition_times), 1),
                },
                "liveness_latency_ms": {
                    "avg": round(self._safe_avg(self._liveness_times), 1),
                    "min": round(self._safe_min(self._liveness_times), 1),
                    "max": round(self._safe_max(self._liveness_times), 1),
                },
                "ppe_latency_ms": {
                    "avg": round(self._safe_avg(self._ppe_times), 1),
                    "min": round(self._safe_min(self._ppe_times), 1),
                    "max": round(self._safe_max(self._ppe_times), 1),
                },
                "recognition_stats": {
                    "total_attempts": self._recognition_count,
                    "successful": self._recognition_success_count,
                    "success_rate_percent": round(recognition_success_rate, 1),
                    "spoofs_detected": self._spoof_detected_count,
                    "unknown_faces": self._unknown_face_count,
                },
            }


class GlobalMetricsRegistry:
    """Global registry of camera-specific metrics trackers.
    
    Provides thread-safe access to per-camera trackers and
    aggregated metrics across all cameras.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._trackers: Dict[int, CameraMetricsTracker] = {}

    def get_tracker(self, camera_id: int) -> CameraMetricsTracker:
        """Get or create a metrics tracker for a camera."""
        with self._lock:
            if camera_id not in self._trackers:
                self._trackers[camera_id] = CameraMetricsTracker(camera_id)
            return self._trackers[camera_id]

    def get_all_snapshots(self) -> Dict[int, Dict[str, Any]]:
        """Get metrics snapshots for all active cameras."""
        with self._lock:
            return {
                cam_id: tracker.get_snapshot()
                for cam_id, tracker in self._trackers.items()
            }

    def get_aggregated_snapshot(self) -> Dict[str, Any]:
        """Get aggregated metrics across all cameras.
        
        Combines statistics from all active trackers.
        """
        snapshots = self.get_all_snapshots()
        if not snapshots:
            return {
                "total_cameras": 0,
                "total_frames": 0,
                "total_frame_drops": 0,
                "aggregated_fps": 0.0,
                "cameras": {},
            }

        total_frames = sum(s.get("total_frames", 0) for s in snapshots.values())
        total_frame_drops = sum(s.get("total_frame_drops", 0) for s in snapshots.values())

        # Aggregate FPS across cameras
        aggregated_fps = sum(s.get("fps", 0.0) for s in snapshots.values())

        # Compute average latencies
        all_frame_times = []
        all_detection_times = []
        all_recognition_times = []
        all_liveness_times = []

        with self._lock:
            for tracker in self._trackers.values():
                all_frame_times.extend(tracker._frame_times)
                all_detection_times.extend(tracker._detection_times)
                all_recognition_times.extend(tracker._recognition_times)
                all_liveness_times.extend(tracker._liveness_times)

        def _safe_avg_list(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "total_cameras": len(snapshots),
            "total_frames": total_frames,
            "total_frame_drops": total_frame_drops,
            "aggregated_fps": round(aggregated_fps, 1),
            "avg_frame_time_ms": round(_safe_avg_list(all_frame_times), 1),
            "avg_detection_latency_ms": round(_safe_avg_list(all_detection_times), 1),
            "avg_recognition_latency_ms": round(_safe_avg_list(all_recognition_times), 1),
            "avg_liveness_latency_ms": round(_safe_avg_list(all_liveness_times), 1),
            "cameras": snapshots,
        }

    def reset_camera(self, camera_id: int) -> None:
        """Reset metrics for a specific camera (e.g., on disconnect)."""
        with self._lock:
            self._trackers.pop(camera_id, None)


# Global singleton instance
_metrics_registry = GlobalMetricsRegistry()


def get_tracker(camera_id: int) -> CameraMetricsTracker:
    """Get or create a metrics tracker for a camera."""
    return _metrics_registry.get_tracker(camera_id)


def get_all_snapshots() -> Dict[int, Dict[str, Any]]:
    """Get metrics snapshots for all active cameras."""
    return _metrics_registry.get_all_snapshots()


def get_aggregated_metrics() -> Dict[str, Any]:
    """Get aggregated metrics across all cameras."""
    return _metrics_registry.get_aggregated_snapshot()


def record_frame_time(camera_id: int, elapsed_ms: float) -> None:
    """Record frame processing time for a camera."""
    _metrics_registry.get_tracker(camera_id).record_frame_time(elapsed_ms)


def record_detection_time(camera_id: int, elapsed_ms: float) -> None:
    """Record detection latency for a camera."""
    _metrics_registry.get_tracker(camera_id).record_detection_time(elapsed_ms)


def record_recognition_time(camera_id: int, elapsed_ms: float) -> None:
    """Record recognition latency for a camera."""
    _metrics_registry.get_tracker(camera_id).record_recognition_time(elapsed_ms)


def record_liveness_time(camera_id: int, elapsed_ms: float) -> None:
    """Record liveness latency for a camera."""
    _metrics_registry.get_tracker(camera_id).record_liveness_time(elapsed_ms)


def record_frame_drop(camera_id: int) -> None:
    """Record a dropped frame for a camera."""
    _metrics_registry.get_tracker(camera_id).record_frame_drop()


def set_queue_depth(camera_id: int, depth: int) -> None:
    """Set current queue depth for a camera."""
    _metrics_registry.get_tracker(camera_id).set_queue_depth(depth)


def record_recognition(camera_id: int, success: bool) -> None:
    """Record a recognition attempt for a camera."""
    _metrics_registry.get_tracker(camera_id).record_recognition(success)


def record_spoof_detected(camera_id: int) -> None:
    """Record a spoof detection for a camera."""
    _metrics_registry.get_tracker(camera_id).record_spoof_detected()


def record_unknown_face(camera_id: int) -> None:
    """Record an unknown face for a camera."""
    _metrics_registry.get_tracker(camera_id).record_unknown_face()
