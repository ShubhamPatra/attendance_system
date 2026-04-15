from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BlinkResult:
	blink_detected: bool
	blink_count: int
	ear_current: float
	is_natural: bool


class BlinkDetector:
	def __init__(
		self,
		buffer_size: int = 30,
		ear_threshold: float = 0.21,
		min_consecutive: int = 2,
		max_consecutive: int = 4,
		window_frames: int = 25,
	) -> None:
		self.buffer_size = max(1, int(buffer_size))
		self.ear_threshold = float(ear_threshold)
		self.min_consecutive = int(min_consecutive)
		self.max_consecutive = int(max_consecutive)
		self.window_frames = max(1, int(window_frames))

		self.ear_history: deque[float] = deque(maxlen=self.buffer_size)
		self._frame_index = 0
		self._below_counter = 0
		self._blink_frames: list[int] = []
		self._blink_count = 0

	@staticmethod
	def _compute_ear(eye_landmarks: np.ndarray) -> float:
		pts = np.asarray(eye_landmarks, dtype=np.float32)
		if pts.shape != (6, 2):
			raise ValueError("Eye landmarks must be shape (6, 2)")

		vertical_1 = np.linalg.norm(pts[1] - pts[5])
		vertical_2 = np.linalg.norm(pts[2] - pts[4])
		horizontal = np.linalg.norm(pts[0] - pts[3])
		if horizontal <= 1e-12:
			return 0.0
		return float((vertical_1 + vertical_2) / (2.0 * horizontal))

	def update(self, landmarks) -> float:
		left_eye, right_eye = self._extract_eyes(landmarks)
		ear_left = self._compute_ear(left_eye)
		ear_right = self._compute_ear(right_eye)
		ear = float((ear_left + ear_right) / 2.0)

		self._frame_index += 1
		self.ear_history.append(ear)
		self._detect_blink(ear)
		return ear

	def _detect_blink(self, ear_value: float) -> None:
		if ear_value < self.ear_threshold:
			self._below_counter += 1
			return

		if self.min_consecutive <= self._below_counter <= self.max_consecutive:
			self._blink_count += 1
			self._blink_frames.append(self._frame_index)
		self._below_counter = 0

	def check(self) -> BlinkResult:
		ear_current = float(self.ear_history[-1]) if self.ear_history else 0.0
		recent_blinks = [frame for frame in self._blink_frames if (self._frame_index - frame) <= self.window_frames]
		blink_detected = len(recent_blinks) >= 1
		return BlinkResult(
			blink_detected=blink_detected,
			blink_count=self._blink_count,
			ear_current=ear_current,
			is_natural=self._is_natural_pattern(),
		)

	def _is_natural_pattern(self) -> bool:
		if len(self._blink_frames) < 3:
			return True

		intervals = np.diff(np.asarray(self._blink_frames, dtype=np.float32))
		if intervals.size == 0:
			return True

		std = float(np.std(intervals))
		return std >= 1.0

	@staticmethod
	def _extract_eyes(landmarks) -> tuple[np.ndarray, np.ndarray]:
		if isinstance(landmarks, dict):
			left = np.asarray(landmarks.get("left_eye"), dtype=np.float32)
			right = np.asarray(landmarks.get("right_eye"), dtype=np.float32)
			return left, right

		arr = np.asarray(landmarks, dtype=np.float32)
		if arr.shape == (12, 2):
			return arr[:6], arr[6:]
		raise ValueError("Landmarks must provide left/right eye points")
