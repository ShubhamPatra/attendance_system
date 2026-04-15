"""Shared ONNX helpers for the migrated package layout."""

from __future__ import annotations

import os
from pathlib import Path

import onnxruntime as ort


def create_onnx_session(
    model_path: str,
    providers: list[str] | None = None,
    intra_op_threads: int | None = None,
    optimized: bool = True,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with safe defaults."""
    model = Path(model_path)
    if not model.exists():
        raise FileNotFoundError(f"ONNX model file not found: {model}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session_options.enable_mem_pattern = bool(optimized)

    cpu_count = max(1, os.cpu_count() or 1)
    session_options.intra_op_num_threads = int(max(1, intra_op_threads or min(4, cpu_count)))

    resolved_providers = providers or ["CPUExecutionProvider"]
    return ort.InferenceSession(
        str(model),
        sess_options=session_options,
        providers=resolved_providers,
    )
