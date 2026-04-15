from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MovementResult:
	is_natural: bool
	is_static: bool
	displacement_std: float


class MovementChecker:
	def __init__(
		self,
		buffer_size: int = 30,
		static_threshold: float = 1.0,
		natural_min: float = 2.0,
		natural_max: float = 8.0,
		challenge_threshold: float = 20.0,
	) -> None:
		self.buffer_size = max(1, int(buffer_size))
		self.static_threshold = float(static_threshold)
		self.natural_min = float(natural_min)
		self.natural_max = float(natural_max)
		self.challenge_threshold = float(challenge_threshold)

		self.positions: deque[tuple[float, float]] = deque(maxlen=self.buffer_size)
		self._challenge_direction: str | None = None
		self._challenge_start: tuple[float, float] | None = None

	def update(self, face_bbox: tuple[int, int, int, int]) -> tuple[float, float]:
		x, y, w, h = face_bbox
		center = (float(x + (w / 2.0)), float(y + (h / 2.0)))
		self.positions.append(center)
		return center

	def check(self) -> MovementResult:
		if len(self.positions) < 2:
			return MovementResult(is_natural=False, is_static=True, displacement_std=0.0)

		coords = np.asarray(self.positions, dtype=np.float32)
		std_x = float(np.std(coords[:, 0]))
		std_y = float(np.std(coords[:, 1]))
		displacement_std = float((std_x + std_y) / 2.0)

		is_static = displacement_std < self.static_threshold
		is_natural = (self.natural_min <= displacement_std <= self.natural_max) and not is_static
		return MovementResult(is_natural=is_natural, is_static=is_static, displacement_std=displacement_std)

	def start_challenge(self, direction: str) -> None:
		self._challenge_direction = direction.strip().lower()
		self._challenge_start = self.positions[-1] if self.positions else None

	def verify_challenge(self) -> bool:
		if not self.positions or not self._challenge_direction or self._challenge_start is None:
			return False

		current = self.positions[-1]
		dx = current[0] - self._challenge_start[0]
		if self._challenge_direction == "left":
			return dx <= -self.challenge_threshold
		if self._challenge_direction == "right":
			return dx >= self.challenge_threshold
		return False
