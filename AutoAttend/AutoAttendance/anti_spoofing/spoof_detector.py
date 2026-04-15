from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from anti_spoofing.blink_detector import BlinkDetector
from anti_spoofing.model import MiniFASNetAntiSpoof
from anti_spoofing.movement_checker import MovementChecker


@dataclass(slots=True)
class LivenessResult:
	is_real: bool
	score: float
	confidence_level: str
	details: dict[str, Any]


class SpoofDetector:
	def __init__(
		self,
		threshold: float = 0.5,
		model: MiniFASNetAntiSpoof | None = None,
		blink_detector: BlinkDetector | None = None,
		movement_checker: MovementChecker | None = None,
	) -> None:
		self.threshold = float(threshold)
		self.model = model
		self.blink_detector = blink_detector or BlinkDetector()
		self.movement_checker = movement_checker or MovementChecker()

	def check_liveness(self, frame: np.ndarray, bbox: tuple[int, int, int, int], landmarks) -> LivenessResult:
		if self.model is None:
			raise RuntimeError("SpoofDetector requires an anti-spoof model instance")

		model_result = self.model.predict(frame, bbox)

		blink_score = 0.0
		blink_pass = False
		if landmarks is not None:
			self.blink_detector.update(landmarks)
			blink_result = self.blink_detector.check()
			blink_pass = blink_result.blink_detected and blink_result.is_natural
			blink_score = 1.0 if blink_pass else 0.0
		else:
			blink_result = self.blink_detector.check()

		self.movement_checker.update(bbox)
		movement_result = self.movement_checker.check()
		movement_pass = movement_result.is_natural and not movement_result.is_static
		movement_score = 1.0 if movement_pass else 0.0

		model_score = float(model_result.score)
		combined_score = (0.5 * model_score) + (0.25 * blink_score) + (0.25 * movement_score)

		adaptive_threshold = self._adaptive_threshold(frame)
		pass_count = int(model_result.is_real) + int(blink_pass) + int(movement_pass)
		confidence_level = self._classify_confidence(pass_count, bool(model_result.is_real))

		is_real = bool(model_result.is_real) and combined_score >= adaptive_threshold and confidence_level != "REJECTED"
		details = {
			"model": {
				"score": model_score,
				"is_real": model_result.is_real,
				"model_scores": dict(model_result.model_scores),
			},
			"blink": {
				"score": blink_score,
				"blink_detected": blink_result.blink_detected,
				"is_natural": blink_result.is_natural,
				"blink_count": blink_result.blink_count,
			},
			"movement": {
				"score": movement_score,
				"is_natural": movement_result.is_natural,
				"is_static": movement_result.is_static,
				"displacement_std": movement_result.displacement_std,
			},
			"weights": {"model": 0.5, "blink": 0.25, "movement": 0.25},
			"threshold": adaptive_threshold,
			"pass_count": pass_count,
		}
		return LivenessResult(
			is_real=is_real,
			score=float(combined_score),
			confidence_level=confidence_level,
			details=details,
		)

	def _adaptive_threshold(self, frame: np.ndarray) -> float:
		if frame is None or frame.size == 0:
			return self.threshold
		brightness = float(np.mean(frame))
		if brightness < 60.0:
			return max(0.0, self.threshold - 0.05)
		return self.threshold

	@staticmethod
	def _classify_confidence(pass_count: int, model_pass: bool) -> str:
		if pass_count >= 3:
			return "HIGH"
		if pass_count == 2:
			return "MEDIUM"
		if pass_count == 1 and model_pass:
			return "LOW"
		return "REJECTED"
