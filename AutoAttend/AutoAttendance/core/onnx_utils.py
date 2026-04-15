from __future__ import annotations

import os
from pathlib import Path

import onnxruntime as ort


def _resolve_providers(gpu: bool, providers: list[str] | None) -> list[str]:
	if providers:
		return providers
	if gpu:
		return ["CUDAExecutionProvider", "CPUExecutionProvider"]
	return ["CPUExecutionProvider"]


def create_onnx_session(
	model_path: str,
	gpu: bool = False,
	providers: list[str] | None = None,
	intra_op_threads: int | None = None,
	optimized: bool = True,
) -> ort.InferenceSession:
	model = Path(model_path)
	if not model.exists():
		raise FileNotFoundError(f"ONNX model file not found: {model}")

	options = ort.SessionOptions()
	options.graph_optimization_level = (
		ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
	)
	options.enable_mem_pattern = bool(optimized)

	threads = intra_op_threads if intra_op_threads is not None else min(4, max(1, os.cpu_count() or 1))
	options.intra_op_num_threads = int(max(1, threads))

	resolved_providers = _resolve_providers(gpu=gpu, providers=providers)
	return ort.InferenceSession(str(model), sess_options=options, providers=resolved_providers)