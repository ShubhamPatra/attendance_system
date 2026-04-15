"""Tests for GPU / ONNX Runtime provider selection."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib
    import app_core.config as config
    importlib.reload(config)


def test_gpu_providers_disabled_fallbacks_to_cpu(monkeypatch):
    import app_vision.face_engine as face_engine
    import app_core.config as config

    monkeypatch.setattr(config, "ENABLE_GPU_PROVIDERS", False)
    monkeypatch.setattr(face_engine, "_DEVICE", "cuda")

    assert face_engine._get_onnxruntime_providers() == ["CPUExecutionProvider"]


def test_gpu_providers_prefer_cuda_when_enabled(monkeypatch):
    import app_vision.face_engine as face_engine
    import app_core.config as config

    monkeypatch.setattr(config, "ENABLE_GPU_PROVIDERS", True)
    monkeypatch.setattr(config, "ONNXRT_PROVIDER_PRIORITY", "CUDAExecutionProvider,CPUExecutionProvider")
    monkeypatch.setattr(face_engine, "_DEVICE", "cuda")

    assert face_engine._get_onnxruntime_providers()[0] == "CUDAExecutionProvider"
