"""Anti-spoofing package compatibility layer."""

from .model import (
	LIVENESS_LABELS,
	check_liveness,
	get_initialization_error,
	init_models,
	is_ready,
)
from . import blink_detector, model, movement_checker, spoof_detector

__all__ = [
	"LIVENESS_LABELS",
	"check_liveness",
	"get_initialization_error",
	"init_models",
	"is_ready",
	"blink_detector",
	"model",
	"movement_checker",
	"spoof_detector",
]
