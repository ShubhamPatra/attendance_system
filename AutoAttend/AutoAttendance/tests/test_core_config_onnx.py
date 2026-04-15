from __future__ import annotations

from types import SimpleNamespace

from core import config as config_module
from core import onnx_utils


def test_load_config_variants(monkeypatch):
	monkeypatch.setenv("FLASK_SECRET_KEY", "secret")
	monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017/test")
	monkeypatch.setenv("FLASK_ENV", "production")
	monkeypatch.setenv("FLASK_DEBUG", "0")
	monkeypatch.setenv("SESSION_COOKIE_SECURE", "1")

	prod = config_module.load_config()
	assert prod.ENV == "production"
	assert prod.DEBUG is False
	assert prod.SESSION_COOKIE_SECURE is True

	monkeypatch.setenv("FLASK_ENV", "testing")
	test_cfg = config_module.load_config()
	assert test_cfg.ENV == "testing"
	assert test_cfg.DEBUG is False

	monkeypatch.setenv("FLASK_ENV", "development")
	dev_cfg = config_module.load_config()
	assert dev_cfg.ENV == "development"
	assert dev_cfg.DEBUG is True


def test_create_onnx_session_resolves_providers_and_handles_missing_file(monkeypatch, tmp_path):
	model_path = tmp_path / "model.onnx"
	model_path.write_bytes(b"fake onnx content")

	class DummySessionOptions:
		def __init__(self):
			self.graph_optimization_level = None
			self.enable_mem_pattern = None
			self.intra_op_num_threads = None

	fake_ort = SimpleNamespace(
		SessionOptions=DummySessionOptions,
		GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="enable", ORT_DISABLE_ALL="disable"),
		InferenceSession=lambda path, sess_options=None, providers=None: {
			"path": path,
			"sess_options": sess_options,
			"providers": providers,
		},
	)
	monkeypatch.setattr(onnx_utils, "ort", fake_ort)
	monkeypatch.setattr(onnx_utils.os, "cpu_count", lambda: 8)

	session = onnx_utils.create_onnx_session(str(model_path), gpu=True, optimized=False)
	assert session["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
	assert session["sess_options"].graph_optimization_level == "disable"
	assert session["sess_options"].enable_mem_pattern is False
	assert session["sess_options"].intra_op_num_threads == 4

	session = onnx_utils.create_onnx_session(str(model_path), providers=["CPUExecutionProvider"], optimized=True)
	assert session["providers"] == ["CPUExecutionProvider"]
	assert session["sess_options"].graph_optimization_level == "enable"
	assert session["sess_options"].enable_mem_pattern is True

	missing = tmp_path / "missing.onnx"
	try:
		onnx_utils.create_onnx_session(str(missing))
	except FileNotFoundError as exc:
		assert "ONNX model file not found" in str(exc)