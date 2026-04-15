"""Convert Silent-Face-Anti-Spoofing .pth weights to ONNX models."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANTI_SPOOFING_DIR = ROOT / "models" / "anti_spoofing"

PTH_TO_ONNX = {
	"2.7_80x80_MiniFASNetV2.pth": "2.7_80x80_MiniFASNetV2.onnx",
	"4_0_0_80x80_MiniFASNetV1SE.pth": "4_0_0_80x80_MiniFASNetV1SE.onnx",
}

MINIFASNET_SOURCE_URL = (
	"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/src/model_lib/MiniFASNet.py"
)


def _download_text(url: str, destination: Path) -> None:
	with urllib.request.urlopen(url, timeout=120) as response:
		destination.write_bytes(response.read())


def _load_minifasnet_module():
	with tempfile.TemporaryDirectory(prefix="minifasnet_src_") as tmp_dir:
		tmp_path = Path(tmp_dir)
		source_file = tmp_path / "MiniFASNet.py"
		_download_text(MINIFASNET_SOURCE_URL, source_file)

		spec = importlib.util.spec_from_file_location("minifasnet_source", source_file)
		if spec is None or spec.loader is None:
			raise RuntimeError("Failed to load MiniFASNet source module")
		module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(module)
		return module


def _parse_model_name(filename: str) -> tuple[int, int, str]:
	match = re.search(r"(?P<h>\d+)x(?P<w>\d+)_(?P<model>MiniFASNet[^.]+)\.pth$", filename)
	if not match:
		raise ValueError(f"Unsupported MiniFASNet filename: {filename}")
	return int(match.group("h")), int(match.group("w")), match.group("model")


def _kernel_size(height: int, width: int) -> tuple[int, int]:
	return ((height + 15) // 16, (width + 15) // 16)


def _normalize_state_dict(raw_state):
	import torch

	if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
		raw_state = raw_state["state_dict"]

	if not isinstance(raw_state, dict):
		raise ValueError("Unsupported checkpoint structure")

	clean_state = {}
	for key, value in raw_state.items():
		if not isinstance(value, torch.Tensor):
			continue
		new_key = key[7:] if key.startswith("module.") else key
		clean_state[new_key] = value
	return clean_state


def _build_model(module, model_name: str, height: int, width: int):
	constructor = getattr(module, model_name, None)
	if constructor is None:
		raise ValueError(f"Model constructor not found in source: {model_name}")
	return constructor(conv6_kernel=_kernel_size(height, width))


def convert_model(pth_path: Path, onnx_path: Path) -> None:
	import torch

	if not pth_path.exists():
		raise FileNotFoundError(f"Missing weight file: {pth_path}")

	height, width, model_name = _parse_model_name(pth_path.name)
	module = _load_minifasnet_module()
	model = _build_model(module, model_name, height, width)

	checkpoint = torch.load(str(pth_path), map_location="cpu")
	state_dict = _normalize_state_dict(checkpoint)
	model.load_state_dict(state_dict, strict=True)
	model.eval()

	dummy = torch.randn(1, 3, height, width)
	onnx_path.parent.mkdir(parents=True, exist_ok=True)
	torch.onnx.export(
		model,
		dummy,
		str(onnx_path),
		export_params=True,
		opset_version=11,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["output"],
		dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert MiniFASNet .pth files to ONNX")
	parser.add_argument("--model-dir", type=Path, default=ANTI_SPOOFING_DIR, help="Directory containing .pth weights")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	model_dir = args.model_dir

	for pth_name, onnx_name in PTH_TO_ONNX.items():
		pth_path = model_dir / pth_name
		onnx_path = model_dir / onnx_name
		print(f"[convert] {pth_name} -> {onnx_name}")
		convert_model(pth_path, onnx_path)
		print(f"[ok] {onnx_path} ({onnx_path.stat().st_size} bytes)")

	return 0


if __name__ == "__main__":
	sys.exit(main())
