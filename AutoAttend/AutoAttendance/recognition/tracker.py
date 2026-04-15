from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


BBox = tuple[int, int, int, int]


@dataclass
class _TrackerState:
	tracker: object
	failure_count: int = 0
	frame_count: int = 0
	last_bbox: BBox | None = None
	success_streak: int = 0


def _default_tracker_factory() -> object:
	if hasattr(cv2, "TrackerCSRT_create"):
		return cv2.TrackerCSRT_create()
	legacy = getattr(cv2, "legacy", None)
	if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
		return legacy.TrackerCSRT_create()
	raise RuntimeError("CSRT tracker is not available in this OpenCV build")


class CSRTTracker:
	def __init__(
		self,
		max_failures: int = 5,
		reset_interval: int = 30,
		iou_threshold: float = 0.3,
		tracker_factory: Callable[[], object] | None = None,
	) -> None:
		self.max_failures = int(max_failures)
		self.reset_interval = int(reset_interval)
		self.iou_threshold = float(iou_threshold)
		self._tracker_factory = tracker_factory or _default_tracker_factory
		self._states: dict[str, _TrackerState] = {}

	def init_track(self, frame: np.ndarray, bbox: BBox, track_id: str = "default") -> bool:
		tracker = self._tracker_factory()
		ok = tracker.init(frame, tuple(map(float, bbox)))
		if ok is None:
			ok = True
		if not ok:
			return False
		self._states[track_id] = _TrackerState(tracker=tracker, failure_count=0, frame_count=0, last_bbox=bbox)
		return True

	def update(self, frame: np.ndarray, track_id: str = "default", det_bbox: BBox | None = None) -> tuple[bool, BBox | None]:
		state = self._states.get(track_id)
		if state is None:
			return False, None

		ok, bbox = state.tracker.update(frame)
		state.frame_count += 1
		if not ok:
			state.failure_count += 1
			state.success_streak = 0
			if self._should_reset(state, det_bbox=det_bbox):
				self._states.pop(track_id, None)
			return False, None

		normalized = self._normalize_bbox(bbox)
		state.last_bbox = normalized
		state.failure_count = 0
		state.success_streak += 1

		if self._should_reset(state, det_bbox=det_bbox):
			self._states.pop(track_id, None)
			return False, None

		return True, normalized

	def get_state(self, track_id: str = "default") -> dict[str, object]:
		state = self._states.get(track_id)
		if state is None:
			return {
				"status": "lost",
				"confidence": 0.0,
				"bbox": None,
				"failure_count": 0,
			}

		confidence = min(1.0, (state.success_streak / 5.0))
		if state.failure_count > 0:
			confidence = max(0.0, confidence - (state.failure_count * 0.25))

		status = "high" if (state.success_streak >= 5 and state.failure_count == 0) else "uncertain"
		return {
			"status": status,
			"confidence": float(confidence),
			"bbox": state.last_bbox,
			"failure_count": state.failure_count,
		}

	def update_all(self, frame: np.ndarray, det_bboxes: dict[str, BBox] | None = None) -> dict[str, tuple[bool, BBox | None]]:
		results: dict[str, tuple[bool, BBox | None]] = {}
		for track_id in list(self._states.keys()):
			det_bbox = (det_bboxes or {}).get(track_id)
			results[track_id] = self.update(frame, track_id=track_id, det_bbox=det_bbox)
		return results

	def is_same_face(self, bbox1: BBox, bbox2: BBox) -> bool:
		return self._iou(bbox1, bbox2) >= self.iou_threshold

	def _should_reset(self, state: _TrackerState, det_bbox: BBox | None = None) -> bool:
		if state.failure_count >= self.max_failures:
			return True
		if state.frame_count >= self.reset_interval:
			return True
		if det_bbox is not None and state.last_bbox is not None:
			if self._iou(state.last_bbox, det_bbox) < self.iou_threshold:
				return True
		return False

	@staticmethod
	def _normalize_bbox(bbox: tuple[float, float, float, float]) -> BBox:
		x, y, w, h = bbox
		return int(round(x)), int(round(y)), int(round(w)), int(round(h))

	@staticmethod
	def _iou(bbox1: BBox, bbox2: BBox) -> float:
		x1, y1, w1, h1 = bbox1
		x2, y2, w2, h2 = bbox2

		x1_max = x1 + w1
		y1_max = y1 + h1
		x2_max = x2 + w2
		y2_max = y2 + h2

		inter_x1 = max(x1, x2)
		inter_y1 = max(y1, y2)
		inter_x2 = min(x1_max, x2_max)
		inter_y2 = min(y1_max, y2_max)

		inter_w = max(0, inter_x2 - inter_x1)
		inter_h = max(0, inter_y2 - inter_y1)
		intersection = inter_w * inter_h

		area1 = max(0, w1) * max(0, h1)
		area2 = max(0, w2) * max(0, h2)
		union = area1 + area2 - intersection
		if union <= 0:
			return 0.0
		return intersection / union
