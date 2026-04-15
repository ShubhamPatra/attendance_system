from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


@dataclass(slots=True)
class DetectionResult:
	bbox: tuple[int, int, int, int]
	landmarks: list[tuple[int, int]]
	confidence: float


class YuNetDetector:
	def __init__(
		self,
		model_path: str,
		confidence: float = 0.7,
		nms_threshold: float = 0.3,
		top_k: int = 5000,
		min_face_size: int = 60,
		processing_size: tuple[int, int] = (320, 320),
	) -> None:
		model = Path(model_path)
		if not model.exists():
			raise FileNotFoundError(f"Face detector model not found: {model}")

		if not hasattr(cv2, "FaceDetectorYN"):
			raise RuntimeError("OpenCV FaceDetectorYN is not available in this environment")

		self.model_path = str(model)
		self.confidence = float(confidence)
		self.nms_threshold = float(nms_threshold)
		self.top_k = int(top_k)
		self.min_face_size = int(min_face_size)
		self.processing_size = processing_size
		self._detector = None
		self._lock = Lock()

	def _get_detector(self):
		if self._detector is not None:
			return self._detector
		with self._lock:
			if self._detector is None:
				self._detector = cv2.FaceDetectorYN.create(
					self.model_path,
					"",
					self.processing_size,
					self.confidence,
					self.nms_threshold,
					self.top_k,
				)
		return self._detector

	def detect(self, frame: np.ndarray) -> list[DetectionResult]:
		if frame is None or frame.size == 0:
			return []

		orig_h, orig_w = frame.shape[:2]
		proc_w, proc_h = self.processing_size

		if (orig_w, orig_h) != (proc_w, proc_h):
			input_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
		else:
			input_frame = frame

		detector = self._get_detector()
		detector.setInputSize((proc_w, proc_h))
		_, faces = detector.detect(input_frame)
		if faces is None or len(faces) == 0:
			return []

		x_scale = orig_w / float(proc_w)
		y_scale = orig_h / float(proc_h)
		results: list[DetectionResult] = []
		for row in faces:
			x, y, w, h = row[:4]
			x_orig = int(round(x * x_scale))
			y_orig = int(round(y * y_scale))
			w_orig = int(round(w * x_scale))
			h_orig = int(round(h * y_scale))
			if min(w_orig, h_orig) < self.min_face_size:
				continue

			landmarks: list[tuple[int, int]] = []
			for idx in range(5):
				lx = int(round(row[4 + idx * 2] * x_scale))
				ly = int(round(row[5 + idx * 2] * y_scale))
				landmarks.append((lx, ly))

			confidence = float(row[14]) if len(row) > 14 else self.confidence
			results.append(
				DetectionResult(
					bbox=(x_orig, y_orig, w_orig, h_orig),
					landmarks=landmarks,
					confidence=confidence,
				)
			)

		return results
