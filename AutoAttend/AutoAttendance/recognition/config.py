from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class RecognitionMode(str, Enum):
	FAST = "FAST"
	BALANCED = "BALANCED"
	ACCURATE = "ACCURATE"


PERFORMANCE_PROFILES: dict[RecognitionMode, dict[str, int]] = {
	RecognitionMode.FAST: {
		"detect_interval": 10,
		"processing_size": 320,
		"frame_scale_width": 320,
		"frame_scale_height": 240,
	},
	RecognitionMode.BALANCED: {
		"detect_interval": 5,
		"processing_size": 480,
		"frame_scale_width": 480,
		"frame_scale_height": 360,
	},
	RecognitionMode.ACCURATE: {
		"detect_interval": 2,
		"processing_size": 640,
		"frame_scale_width": 640,
		"frame_scale_height": 480,
	},
}


@dataclass
class RecognitionConfig:
	mode: RecognitionMode = RecognitionMode.BALANCED
	detect_interval: int = 5
	processing_size: int = 480
	frame_scale_width: int = 480
	frame_scale_height: int = 360
	embedding_interval: int = 2
	min_face_size: int = 60
	detection_confidence: float = 0.7
	nms_threshold: float = 0.3
	top_k: int = 5000
	smoother_window: int = 5
	smoother_majority: int = 3
	match_mode: str = "BALANCED"
	detector_model_path: str = "models/face_detection/face_detection_yunet_2023mar.onnx"
	embedder_model_path: str = "models/face_recognition/arcface_r100.onnx"

	def apply_mode(self, mode: RecognitionMode | str) -> "RecognitionConfig":
		if isinstance(mode, RecognitionMode):
			selected = mode
		else:
			selected = RecognitionMode(str(mode).strip().upper())
		profile = PERFORMANCE_PROFILES[selected]
		self.mode = selected
		self.detect_interval = profile["detect_interval"]
		self.processing_size = profile["processing_size"]
		self.frame_scale_width = profile["frame_scale_width"]
		self.frame_scale_height = profile["frame_scale_height"]
		return self

	@classmethod
	def from_env(cls) -> "RecognitionConfig":
		cfg = cls()
		mode_raw = os.getenv("RECOGNITION_MODE", cfg.mode.value).strip().upper()
		cfg.apply_mode(mode_raw)

		cfg.embedding_interval = int(os.getenv("RECOGNITION_EMBEDDING_INTERVAL", str(cfg.embedding_interval)))
		cfg.min_face_size = int(os.getenv("RECOGNITION_MIN_FACE_SIZE", str(cfg.min_face_size)))
		cfg.detection_confidence = float(os.getenv("FACE_DETECTION_CONFIDENCE", str(cfg.detection_confidence)))
		cfg.nms_threshold = float(os.getenv("RECOGNITION_NMS_THRESHOLD", str(cfg.nms_threshold)))
		cfg.top_k = int(os.getenv("RECOGNITION_TOP_K", str(cfg.top_k)))
		cfg.frame_scale_width = int(os.getenv("RECOGNITION_FRAME_SCALE_WIDTH", str(cfg.frame_scale_width)))
		cfg.frame_scale_height = int(os.getenv("RECOGNITION_FRAME_SCALE_HEIGHT", str(cfg.frame_scale_height)))
		cfg.smoother_window = int(os.getenv("RECOGNITION_SMOOTHER_WINDOW", str(cfg.smoother_window)))
		cfg.smoother_majority = int(os.getenv("RECOGNITION_SMOOTHER_MAJORITY", str(cfg.smoother_majority)))
		cfg.match_mode = os.getenv("RECOGNITION_MATCH_MODE", cfg.match_mode)
		cfg.detector_model_path = os.getenv("YUNET_MODEL_PATH", cfg.detector_model_path)
		cfg.embedder_model_path = os.getenv("ARCFACE_MODEL_PATH", cfg.embedder_model_path)
		return cfg
