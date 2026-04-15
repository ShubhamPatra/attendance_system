"""Benchmark recognition pipeline latency and memory usage."""

from __future__ import annotations

import argparse
import json
import time
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from core.profiling import PipelineProfiler
from core.onnx_utils import create_onnx_session
from recognition.config import RecognitionConfig
from recognition.pipeline import RecognitionPipeline


@dataclass(slots=True)
class _BenchmarkStudentDAO:
	def get_roster_embeddings(self, _course_id):
		return [
			{"student_id": "S100", "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
			{"student_id": "S200", "embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
		]


class _BenchmarkDetector:
	def detect(self, _frame):
		from recognition.detector import DetectionResult

		return [
			DetectionResult(
				bbox=(30, 40, 80, 80),
				landmarks=[(40, 55), (75, 55), (58, 70), (45, 92), (70, 92)],
				confidence=0.99,
			)
		]


class _BenchmarkTracker:
	def init_track(self, _frame, _bbox, track_id="main"):
		return True

	def update(self, _frame, track_id="main"):
		return True, (30, 40, 80, 80)


class _BenchmarkAligner:
	def align(self, _frame, _landmarks):
		return np.zeros((112, 112, 3), dtype=np.uint8)


class _BenchmarkEmbedder:
	def get_embedding(self, _aligned_face):
		return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def build_pipeline(mode: str = "BALANCED", scaling_enabled: bool = True, adaptive_detection: bool = True) -> RecognitionPipeline:
	config = RecognitionConfig().apply_mode(mode)
	config.detect_interval = max(1, config.detect_interval)
	config.embedding_interval = 1
	config.smoother_window = 5
	config.smoother_majority = 3
	if not scaling_enabled:
		config.frame_scale_width = 640
		config.frame_scale_height = 480
	return RecognitionPipeline(
		config=config,
		detector=_BenchmarkDetector(),
		tracker=_BenchmarkTracker(),
		aligner=_BenchmarkAligner(),
		embedder=_BenchmarkEmbedder(),
		student_dao=_BenchmarkStudentDAO(),
		profiler=PipelineProfiler(),
		adaptive_detection=adaptive_detection,
	)


def _format_table(rows: list[tuple[str, dict[str, float]]]) -> str:
	headers = ["Stage", "Count", "Total ms", "Avg ms", "P50 ms", "P95 ms", "P99 ms"]
	widths = [max(len(headers[i]), *(len(f"{row[0]}") for row in rows)) if i == 0 else len(headers[i]) for i in range(len(headers))]
	lines = [" | ".join(headers)]
	lines.append("-|-|-|-|-|-|-")
	for stage, metrics in rows:
		lines.append(
			" | ".join(
				[
					stage,
					str(int(metrics["count"])),
					f"{metrics['total_ms']:.2f}",
					f"{metrics['avg_ms']:.2f}",
					f"{metrics['p50_ms']:.2f}",
					f"{metrics['p95_ms']:.2f}",
					f"{metrics['p99_ms']:.2f}",
				]
			)
		)
	return "\n".join(lines)


def run_benchmark(frame_count: int, output_path: str | None = None) -> dict[str, object]:
	pipeline = build_pipeline(mode="BALANCED", scaling_enabled=True)
	pipeline.load_gallery("COURSE-1")
	frame = np.zeros((480, 640, 3), dtype=np.uint8)

	tracemalloc.start()
	for _ in range(max(1, frame_count)):
		pipeline.process_frame(frame)
	current, peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()

	model_path = Path("models/face_detection/face_detection_yunet_2023mar.onnx")
	session_load_times = {
		"optimized_ms": _benchmark_session_load(model_path, optimized=True),
		"baseline_ms": _benchmark_session_load(model_path, optimized=False),
	}
	scaling_benchmark = _benchmark_scaling_by_mode(frame_count=max(10, frame_count))
	adaptive_benchmark = _benchmark_adaptive_interval(frame_count=max(10, frame_count))

	snapshot = pipeline.profiler.snapshot()
	ordered = [(name, metrics) for name, metrics in sorted(snapshot.items())]
	result = {
		"frame_count": frame_count,
		"fps": pipeline.fps,
		"memory_current_bytes": current,
		"memory_peak_bytes": peak,
		"session_load_benchmark": session_load_times,
		"scaling_benchmark": scaling_benchmark,
		"adaptive_benchmark": adaptive_benchmark,
		"stages": snapshot,
		"table": _format_table(ordered),
		"session_table": _format_session_table(session_load_times),
		"scaling_table": _format_scaling_table(scaling_benchmark),
		"adaptive_table": _format_adaptive_table(adaptive_benchmark),
	}

	if output_path:
		Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")

	return result


def _benchmark_session_load(model_path: Path, optimized: bool) -> float:
	started = time.perf_counter()
	_ = create_onnx_session(str(model_path), optimized=optimized)
	return (time.perf_counter() - started) * 1000.0


def _format_session_table(session_load_times: dict[str, float]) -> str:
	return (
		"Session load benchmark (ms)\n"
		"Mode | Time\n"
		"-|-\n"
		f"Optimized | {session_load_times['optimized_ms']:.2f}\n"
		f"Baseline | {session_load_times['baseline_ms']:.2f}"
	)


def _benchmark_scaling_by_mode(frame_count: int) -> dict[str, dict[str, float]]:
	frame = np.zeros((480, 640, 3), dtype=np.uint8)
	modes = ("FAST", "BALANCED", "ACCURATE")
	results: dict[str, dict[str, float]] = {}
	for mode in modes:
		pipeline_scaled = build_pipeline(mode=mode, scaling_enabled=True)
		pipeline_scaled.load_gallery("COURSE-1")
		for _ in range(frame_count):
			pipeline_scaled.process_frame(frame)

		pipeline_unscaled = build_pipeline(mode=mode, scaling_enabled=False)
		pipeline_unscaled.load_gallery("COURSE-1")
		for _ in range(frame_count):
			pipeline_unscaled.process_frame(frame)

		results[mode] = {
			"fps_scaled": pipeline_scaled.fps,
			"fps_unscaled": pipeline_unscaled.fps,
		}
	return results


def _format_scaling_table(scaling_benchmark: dict[str, dict[str, float]]) -> str:
	lines = ["Scaling benchmark (FPS)", "Mode | Scaled | Unscaled", "-|-|-"]
	for mode in ("FAST", "BALANCED", "ACCURATE"):
		row = scaling_benchmark.get(mode, {"fps_scaled": 0.0, "fps_unscaled": 0.0})
		lines.append(f"{mode} | {row['fps_scaled']:.2f} | {row['fps_unscaled']:.2f}")
	return "\n".join(lines)


def _benchmark_adaptive_interval(frame_count: int) -> dict[str, float]:
	frame = np.zeros((480, 640, 3), dtype=np.uint8)

	pipeline_adaptive = build_pipeline(mode="BALANCED", scaling_enabled=True, adaptive_detection=True)
	pipeline_adaptive.load_gallery("COURSE-1")
	for _ in range(frame_count):
		pipeline_adaptive.process_frame(frame)

	pipeline_fixed = build_pipeline(mode="BALANCED", scaling_enabled=True, adaptive_detection=False)
	pipeline_fixed.load_gallery("COURSE-1")
	for _ in range(frame_count):
		pipeline_fixed.process_frame(frame)

	return {
		"fps_adaptive": pipeline_adaptive.fps,
		"fps_fixed": pipeline_fixed.fps,
	}


def _format_adaptive_table(adaptive_benchmark: dict[str, float]) -> str:
	return (
		"Adaptive interval benchmark (FPS)\n"
		"Mode | FPS\n"
		"-|-\n"
		f"Adaptive detection interval | {adaptive_benchmark['fps_adaptive']:.2f}\n"
		f"Fixed detection interval | {adaptive_benchmark['fps_fixed']:.2f}"
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark the AutoAttendance recognition pipeline")
	parser.add_argument("--frames", type=int, default=60, help="Number of frames to process")
	parser.add_argument("--output", type=str, default=None, help="Optional JSON file to write benchmark results")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	result = run_benchmark(frame_count=args.frames, output_path=args.output)
	print(result["table"])
	print(result["session_table"])
	print(result["scaling_table"])
	print(result["adaptive_table"])
	print(f"FPS: {result['fps']:.2f}")
	print(f"Memory current: {result['memory_current_bytes']} bytes")
	print(f"Memory peak: {result['memory_peak_bytes']} bytes")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())