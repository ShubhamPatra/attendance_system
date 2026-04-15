"""Download and validate required ML model artifacts for AutoAttendance."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
FACE_DETECTION_DIR = MODELS_DIR / "face_detection"
ANTI_SPOOFING_DIR = MODELS_DIR / "anti_spoofing"

YUNET_PATH = FACE_DETECTION_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
	"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
	"https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]

MINIFASNET_V2_PATH = ANTI_SPOOFING_DIR / "2.7_80x80_MiniFASNetV2.onnx"
MINIFASNET_V2_URLS = [
	"https://huggingface.co/onnx-community/anti-spoofing/resolve/main/2.7_80x80_MiniFASNetV2.onnx",
	"https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx",
	"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.onnx",
]

MINIFASNET_V1SE_PATH = ANTI_SPOOFING_DIR / "4_0_0_80x80_MiniFASNetV1SE.onnx"
MINIFASNET_V1SE_URLS = [
	"https://huggingface.co/onnx-community/anti-spoofing/resolve/main/4_0_0_80x80_MiniFASNetV1SE.onnx",
	"https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.onnx",
	"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.onnx",
]

MINIFASNET_V2_PTH_PATH = ANTI_SPOOFING_DIR / "2.7_80x80_MiniFASNetV2.pth"
MINIFASNET_V2_PTH_URLS = [
	"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
]

MINIFASNET_V1SE_PTH_PATH = ANTI_SPOOFING_DIR / "4_0_0_80x80_MiniFASNetV1SE.pth"
MINIFASNET_V1SE_PTH_URLS = [
	"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
]

MIN_SIZE_BYTES = {
	YUNET_PATH: 200_000,
	MINIFASNET_V2_PATH: 200_000,
	MINIFASNET_V1SE_PATH: 200_000,
}


def _sha256(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


def _download(url: str, destination: Path, timeout: int = 120) -> None:
	destination.parent.mkdir(parents=True, exist_ok=True)
	with urllib.request.urlopen(url, timeout=timeout) as response:
		payload = response.read()
	destination.write_bytes(payload)


def _download_if_missing(urls: list[str], destination: Path) -> None:
	if destination.exists() and destination.stat().st_size > 0:
		print(f"[skip] {destination.name} already exists")
		return

	last_error: Exception | None = None
	for url in urls:
		print(f"[download] {destination.name} from {url}")
		try:
			_download(url, destination)
			print(f"[ok] {destination.name} ({destination.stat().st_size} bytes)")
			return
		except Exception as exc:
			last_error = exc

	if destination.exists() and destination.stat().st_size == 0:
		destination.unlink(missing_ok=True)

	if last_error is None:
		raise RuntimeError(f"No download URLs configured for {destination.name}")
	raise last_error


def _try_download_if_missing(urls: list[str], destination: Path) -> bool:
	try:
		_download_if_missing(urls, destination)
		return True
	except Exception as exc:
		print(f"[warn] unable to download {destination.name}: {exc}")
		return False


def _convert_pth_to_onnx() -> None:
	converter = ROOT / "scripts" / "convert_antispoof_to_onnx.py"
	if not converter.exists():
		raise FileNotFoundError(f"Missing converter script: {converter}")

	command = [sys.executable, str(converter), "--model-dir", str(ANTI_SPOOFING_DIR)]
	print(f"[convert] running {' '.join(command)}")
	result = subprocess.run(command, check=False)
	if result.returncode != 0:
		raise RuntimeError("MiniFASNet conversion to ONNX failed")


def _trigger_insightface_download() -> None:
	print("[download] insightface buffalo_l model pack")
	import insightface

	app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
	app.prepare(ctx_id=0, det_size=(640, 640))
	print("[ok] insightface buffalo_l initialized")


def _validate_onnx_load(path: Path) -> None:
	import onnxruntime as ort

	_ = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _validate_file(path: Path) -> None:
	if not path.exists():
		raise FileNotFoundError(f"Missing required file: {path}")

	actual_size = path.stat().st_size
	min_size = MIN_SIZE_BYTES.get(path, 1)
	if actual_size < min_size:
		raise ValueError(f"File too small: {path} ({actual_size} bytes, expected >= {min_size})")

	_validate_onnx_load(path)
	print(f"[validate] {path.name}: size={actual_size} sha256={_sha256(path)[:16]}...")


def _validate_insightface_presence() -> None:
	buffalo_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
	if not buffalo_dir.exists():
		raise FileNotFoundError(f"Missing insightface model directory: {buffalo_dir}")
	files = list(buffalo_dir.rglob("*"))
	if len(files) < 3:
		raise ValueError(f"Unexpected buffalo_l contents in {buffalo_dir}")
	print(f"[validate] insightface buffalo_l present at {buffalo_dir}")


def _manual_instructions() -> str:
	return (
		"Manual fallback:\n"
		f"1) Download YuNet model to {YUNET_PATH}\n"
		f"2) Download MiniFASNetV2 ONNX to {MINIFASNET_V2_PATH} (or PTH to {MINIFASNET_V2_PTH_PATH})\n"
		f"3) Download MiniFASNetV1SE ONNX to {MINIFASNET_V1SE_PATH} (or PTH to {MINIFASNET_V1SE_PTH_PATH})\n"
		"4) Optional conversion from PTH: python scripts/convert_antispoof_to_onnx.py\n"
		"5) Run: python scripts/download_models.py --validate-only\n"
		"6) For InsightFace, run a Python shell and execute:\n"
		"   import insightface\n"
		"   app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])\n"
		"   app.prepare(ctx_id=0, det_size=(640, 640))"
	)


def run(validate_only: bool, skip_insightface: bool) -> int:
	FACE_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
	ANTI_SPOOFING_DIR.mkdir(parents=True, exist_ok=True)

	if not validate_only:
		try:
			_download_if_missing(YUNET_URLS, YUNET_PATH)
			v2_ok = _try_download_if_missing(MINIFASNET_V2_URLS, MINIFASNET_V2_PATH)
			v1se_ok = _try_download_if_missing(MINIFASNET_V1SE_URLS, MINIFASNET_V1SE_PATH)

			if not (v2_ok and v1se_ok):
				print("[info] ONNX direct download unavailable, attempting PTH fallback + conversion")
				_download_if_missing(MINIFASNET_V2_PTH_URLS, MINIFASNET_V2_PTH_PATH)
				_download_if_missing(MINIFASNET_V1SE_PTH_URLS, MINIFASNET_V1SE_PTH_PATH)
				_convert_pth_to_onnx()

			if not skip_insightface:
				_trigger_insightface_download()
		except urllib.error.URLError as exc:
			print(f"[error] network download failed: {exc}")
			print(_manual_instructions())
			return 1
		except Exception as exc:
			print(f"[error] download phase failed: {exc}")
			print(_manual_instructions())
			return 1

	try:
		_validate_file(YUNET_PATH)
		_validate_file(MINIFASNET_V2_PATH)
		_validate_file(MINIFASNET_V1SE_PATH)
		if not skip_insightface:
			_validate_insightface_presence()
	except Exception as exc:
		print(f"[error] validation failed: {exc}")
		print(_manual_instructions())
		return 1

	print("[ok] model bootstrap complete")
	return 0


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download and validate AutoAttendance ML models")
	parser.add_argument("--validate-only", action="store_true", help="Skip download and only validate existing files")
	parser.add_argument("--skip-insightface", action="store_true", help="Skip insightface buffalo_l download/validation")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	sys.exit(run(validate_only=args.validate_only, skip_insightface=args.skip_insightface))
