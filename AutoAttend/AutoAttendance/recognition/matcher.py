from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np


STRICT_THRESHOLD = 0.55
BALANCED_THRESHOLD = 0.45
LENIENT_THRESHOLD = 0.35


@dataclass(slots=True)
class MatchResult:
	student_id: str
	similarity: float
	rank: int


class EmbeddingCache:
	def __init__(self, max_entries: int = 32) -> None:
		self.max_entries = max(1, int(max_entries))
		self._store: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

	def get(self, key: str) -> dict[str, np.ndarray] | None:
		value = self._store.get(key)
		if value is None:
			return None
		self._store.move_to_end(key)
		return value

	def set(self, key: str, value: dict[str, np.ndarray]) -> None:
		self._store[key] = value
		self._store.move_to_end(key)
		while len(self._store) > self.max_entries:
			self._store.popitem(last=False)

	def invalidate(self, key: str) -> None:
		self._store.pop(key, None)

	def clear(self) -> None:
		self._store.clear()

	def prewarm(self, key: str, loader: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
		existing = self.get(key)
		if existing is not None:
			return existing
		loaded = loader()
		self.set(key, loaded)
		return loaded


class FaceMatcher:
	def __init__(self, cache: EmbeddingCache | None = None) -> None:
		self.cache = cache or EmbeddingCache()

	@staticmethod
	def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
		v1 = np.asarray(emb1, dtype=np.float32)
		v2 = np.asarray(emb2, dtype=np.float32)
		n1 = float(np.linalg.norm(v1))
		n2 = float(np.linalg.norm(v2))
		if n1 <= 1e-12 or n2 <= 1e-12:
			return 0.0
		return float(np.dot(v1, v2) / (n1 * n2))

	@staticmethod
	def _normalize_rows(rows: np.ndarray) -> np.ndarray:
		norms = np.linalg.norm(rows, axis=1, keepdims=True)
		norms = np.where(norms <= 1e-12, 1.0, norms)
		return rows / norms

	def match_against_gallery(
		self,
		probe: np.ndarray,
		gallery: dict[str, list[np.ndarray] | np.ndarray],
		top_k: int = 5,
	) -> list[MatchResult]:
		if not gallery:
			return []

		probe_vec = np.asarray(probe, dtype=np.float32).reshape(1, -1)
		probe_norm = self._normalize_rows(probe_vec)[0]

		all_vectors: list[np.ndarray] = []
		owners: list[str] = []
		for student_id, embeddings in gallery.items():
			if isinstance(embeddings, np.ndarray):
				if embeddings.ndim == 1:
					arr = embeddings.reshape(1, -1)
				else:
					arr = embeddings
			else:
				if not embeddings:
					continue
				arr = np.stack([np.asarray(item, dtype=np.float32) for item in embeddings], axis=0)

			for row in arr:
				all_vectors.append(np.asarray(row, dtype=np.float32))
				owners.append(student_id)

		if not all_vectors:
			return []

		matrix = np.stack(all_vectors, axis=0)
		matrix = self._normalize_rows(matrix)
		similarities = matrix @ probe_norm.T

		best_by_student: dict[str, float] = {}
		for idx, owner in enumerate(owners):
			score = float(similarities[idx])
			best_by_student[owner] = max(score, best_by_student.get(owner, -1.0))

		sorted_pairs = sorted(best_by_student.items(), key=lambda item: item[1], reverse=True)
		return [
			MatchResult(student_id=student_id, similarity=similarity, rank=rank)
			for rank, (student_id, similarity) in enumerate(sorted_pairs[: max(1, top_k)], start=1)
		]

	@staticmethod
	def is_match(similarity: float, mode: str = "BALANCED") -> bool:
		mode_key = (mode or "BALANCED").strip().upper()
		thresholds = {
			"STRICT": STRICT_THRESHOLD,
			"BALANCED": BALANCED_THRESHOLD,
			"LENIENT": LENIENT_THRESHOLD,
		}
		threshold = thresholds.get(mode_key, BALANCED_THRESHOLD)
		return float(similarity) >= threshold
