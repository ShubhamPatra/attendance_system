from __future__ import annotations

from pathlib import Path
from typing import Any
from threading import Lock

import cv2
import numpy as np

from core.onnx_utils import create_onnx_session


class ArcFaceEmbedder:
	_session_cache: dict[str, Any] = {}

	def __init__(
		self,
		model_path: str,
		providers: list[str] | None = None,
		lazy_loading: bool = True,
	) -> None:
		model = Path(model_path)
		if not model.exists():
			raise FileNotFoundError(f"ArcFace model not found: {model}")

		self.model_path = str(model)
		self.providers = providers or ["CPUExecutionProvider"]
		self.lazy_loading = bool(lazy_loading)
		self._input_name: str | None = None
		self._lock = Lock()

		if not self.lazy_loading:
			_ = self._get_session()

	def _get_session(self) -> Any:
		session = self._session_cache.get(self.model_path)
		if session is None:
			with self._lock:
				session = self._session_cache.get(self.model_path)
				if session is None:
					session = create_onnx_session(self.model_path, providers=self.providers)
					self._session_cache[self.model_path] = session
		if self._input_name is None:
			self._input_name = session.get_inputs()[0].name
		return session

	@staticmethod
	def _l2_normalize(vector: np.ndarray) -> np.ndarray:
		norm = float(np.linalg.norm(vector))
		if norm <= 1e-12:
			return vector.astype(np.float32)
		return (vector / norm).astype(np.float32)

	def _preprocess(self, face: np.ndarray) -> np.ndarray:
		if face is None or face.size == 0:
			raise ValueError("Input face image is empty")

		if face.shape[:2] != (112, 112):
			face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)

		rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		normalized = (rgb.astype(np.float32) - 127.5) / 128.0
		tensor = np.transpose(normalized, (2, 0, 1))
		return np.expand_dims(tensor, axis=0).astype(np.float32)

	def get_embedding(self, aligned_face_112x112: np.ndarray) -> np.ndarray:
		tensor = self._preprocess(aligned_face_112x112)
		session = self._get_session()
		output = session.run(None, {self._input_name: tensor})[0]
		embedding = np.asarray(output[0], dtype=np.float32)
		return self._l2_normalize(embedding)

	def get_embeddings(self, faces_list: list[np.ndarray]) -> np.ndarray:
		if not faces_list:
			return np.empty((0, 512), dtype=np.float32)

		batch = np.concatenate([self._preprocess(face) for face in faces_list], axis=0)
		session = self._get_session()
		output = session.run(None, {self._input_name: batch})[0]
		embeddings = np.asarray(output, dtype=np.float32)
		norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
		norms = np.where(norms <= 1e-12, 1.0, norms)
		return (embeddings / norms).astype(np.float32)

	@staticmethod
	def get_quality_score(face_confidence: float, alignment_error: float) -> float:
		confidence = float(np.clip(face_confidence, 0.0, 1.0))
		alignment_component = 1.0 - float(np.clip(alignment_error / 20.0, 0.0, 1.0))
		score = (0.7 * confidence) + (0.3 * alignment_component)
		return float(np.clip(score, 0.0, 1.0))
