from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from threading import Lock

import cv2
import numpy as np

from core.onnx_utils import create_onnx_session


@dataclass(slots=True)
class AntiSpoofResult:
	is_real: bool
	score: float
	model_scores: dict[str, float]


class MiniFASNetAntiSpoof:
	def __init__(
		self,
		model_path_v2: str,
		model_path_v1se: str,
		threshold: float = 0.5,
		providers: list[str] | None = None,
	) -> None:
		self.model_path_v2 = self._validate_model_path(model_path_v2)
		self.model_path_v1se = self._validate_model_path(model_path_v1se)
		self.threshold = float(threshold)
		self.providers = providers or ["CPUExecutionProvider"]
		self.session_v2: Any | None = None
		self.session_v1se: Any | None = None
		self.input_name_v2: str | None = None
		self.input_name_v1se: str | None = None
		self._lock = Lock()

	def _ensure_sessions(self) -> None:
		if self.session_v2 is not None and self.session_v1se is not None:
			return
		with self._lock:
			if self.session_v2 is None:
				self.session_v2 = create_onnx_session(self.model_path_v2, providers=self.providers)
				self.input_name_v2 = self.session_v2.get_inputs()[0].name
			if self.session_v1se is None:
				self.session_v1se = create_onnx_session(self.model_path_v1se, providers=self.providers)
				self.input_name_v1se = self.session_v1se.get_inputs()[0].name

	@staticmethod
	def _validate_model_path(path: str) -> str:
		model = Path(path)
		if not model.exists():
			raise FileNotFoundError(f"Anti-spoofing model not found: {model}")
		return str(model)

	@staticmethod
	def _multi_scale_crop(frame: np.ndarray, bbox: tuple[int, int, int, int], scale: float) -> np.ndarray:
		x, y, w, h = bbox
		h_img, w_img = frame.shape[:2]

		cx = x + (w / 2.0)
		cy = y + (h / 2.0)
		scaled_size = max(w, h) * float(scale)
		half = scaled_size / 2.0

		x1 = max(0, int(round(cx - half)))
		y1 = max(0, int(round(cy - half)))
		x2 = min(w_img, int(round(cx + half)))
		y2 = min(h_img, int(round(cy + half)))

		if x2 <= x1 or y2 <= y1:
			crop = np.zeros((80, 80, 3), dtype=np.uint8)
			return crop

		crop = frame[y1:y2, x1:x2]
		return cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)

	@staticmethod
	def _preprocess(crop: np.ndarray) -> np.ndarray:
		rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
		normalized = (rgb.astype(np.float32) - 127.5) / 128.0
		tensor = np.transpose(normalized, (2, 0, 1))
		return np.expand_dims(tensor, axis=0).astype(np.float32)

	@staticmethod
	def _predict_single(session: Any, input_name: str, preprocessed: np.ndarray) -> float:
		output = session.run(None, {input_name: preprocessed})[0]
		arr = np.asarray(output, dtype=np.float32).reshape(-1)
		if arr.size == 0:
			return 0.0
		if arr.size == 1:
			return float(np.clip(arr[0], 0.0, 1.0))

		shifted = arr - np.max(arr)
		exp_vals = np.exp(shifted)
		probs = exp_vals / np.sum(exp_vals)
		return float(np.clip(probs[-1], 0.0, 1.0))

	def predict(self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]) -> AntiSpoofResult:
		self._ensure_sessions()
		crop_v2 = self._multi_scale_crop(frame, face_bbox, scale=2.7)
		crop_v1se = self._multi_scale_crop(frame, face_bbox, scale=4.0)

		pre_v2 = self._preprocess(crop_v2)
		pre_v1se = self._preprocess(crop_v1se)

		score_v2 = self._predict_single(self.session_v2, self.input_name_v2, pre_v2)
		score_v1se = self._predict_single(self.session_v1se, self.input_name_v1se, pre_v1se)

		ensemble_score = float((score_v2 + score_v1se) / 2.0)
		return AntiSpoofResult(
			is_real=ensemble_score >= self.threshold,
			score=ensemble_score,
			model_scores={"2.7": score_v2, "4.0": score_v1se},
		)
