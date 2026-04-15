"""Anti-spoofing model compatibility module."""

from vision.anti_spoofing import (
    LIVENESS_LABELS,
    check_liveness,
    get_initialization_error,
    init_models,
    is_ready,
)

__all__ = [
    "LIVENESS_LABELS",
    "check_liveness",
    "get_initialization_error",
    "init_models",
    "is_ready",
]
