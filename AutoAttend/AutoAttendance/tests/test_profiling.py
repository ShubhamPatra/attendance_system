from __future__ import annotations

from core.profiling import PipelineProfiler, RollingFPSCounter
from scripts.benchmark_pipeline import run_benchmark


def test_pipeline_profiler_tracks_stage_stats():
	profiler = PipelineProfiler()
	for _ in range(3):
		with profiler.stage("embed"):
			pass

	snapshot = profiler.snapshot()
	assert "embed" in snapshot
	assert snapshot["embed"]["count"] == 3
	assert snapshot["embed"]["avg_ms"] >= 0.0
	assert snapshot["embed"]["p50_ms"] >= 0.0


def test_rolling_fps_counter_uses_window_average():
	counter = RollingFPSCounter(window_size=3)
	for duration in (0.1, 0.2, 0.1):
		counter.add_frame_time(duration)

	assert counter.value() == 1.0 / (sum((0.1, 0.2, 0.1)) / 3.0)


def test_benchmark_pipeline_returns_summary_table():
	result = run_benchmark(frame_count=4)

	assert result["frame_count"] == 4
	assert "table" in result
	assert "Stage" in result["table"]
	assert "memory_peak_bytes" in result