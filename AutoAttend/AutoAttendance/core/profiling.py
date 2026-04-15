from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass(slots=True)
class StageMetrics:
	count: int = 0
	total_seconds: float = 0.0
	_samples: list[float] = field(default_factory=list)

	def add(self, elapsed_seconds: float) -> None:
		self.count += 1
		self.total_seconds += float(elapsed_seconds)
		self._samples.append(float(elapsed_seconds))

	def snapshot(self) -> dict[str, float]:
		if not self._samples:
			return {
				"count": 0,
				"total_ms": 0.0,
				"avg_ms": 0.0,
				"p50_ms": 0.0,
				"p95_ms": 0.0,
				"p99_ms": 0.0,
			}

		ordered = sorted(self._samples)
		count = len(ordered)

		def percentile(p: float) -> float:
			if count == 1:
				return ordered[0] * 1000.0
			index = int(round((count - 1) * p))
			index = max(0, min(count - 1, index))
			return ordered[index] * 1000.0

		return {
			"count": float(self.count),
			"total_ms": self.total_seconds * 1000.0,
			"avg_ms": (self.total_seconds / max(1, self.count)) * 1000.0,
			"p50_ms": percentile(0.50),
			"p95_ms": percentile(0.95),
			"p99_ms": percentile(0.99),
		}


class RollingFPSCounter:
	def __init__(self, window_size: int = 30) -> None:
		self.window_size = max(1, int(window_size))
		self._durations: deque[float] = deque(maxlen=self.window_size)

	def add_frame_time(self, frame_seconds: float) -> None:
		if frame_seconds > 0:
			self._durations.append(float(frame_seconds))

	def value(self) -> float:
		if not self._durations:
			return 0.0
		total = sum(self._durations)
		return float(len(self._durations) / total) if total > 0 else 0.0


class PipelineProfiler:
	def __init__(self) -> None:
		self._metrics: dict[str, StageMetrics] = defaultdict(StageMetrics)

	@contextmanager
	def stage(self, name: str) -> Iterator[None]:
		started = perf_counter()
		try:
			yield
		finally:
			elapsed = perf_counter() - started
			self._metrics[name].add(elapsed)

	def record(self, name: str, elapsed_seconds: float) -> None:
		self._metrics[name].add(elapsed_seconds)

	def snapshot(self) -> dict[str, dict[str, float]]:
		return {name: metrics.snapshot() for name, metrics in self._metrics.items()}

	def reset(self) -> None:
		self._metrics.clear()