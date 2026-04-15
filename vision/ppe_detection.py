"""
PPE detection module for mask/cap inference on face crops.

This module is intentionally fail-safe:
* If PPE detection is disabled, it always returns "none".
* If model loading/inference fails, it logs and returns "none".

Model contract (ONNX): output logits/probabilities for either:
* 4 classes: [none, mask, cap, both]
* 3 classes: [none, mask, cap]
* 2 classes: [mask, cap] ("none" inferred)
"""

from __future__ import annotations

import os

import cv2
import numpy as np

import core.config as config
from core.utils import setup_logging

logger = setup_logging()

_net = None
_model_path = ""


def _softmax(vec: np.ndarray) -> np.ndarray:
    x = vec.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    denom = float(np.sum(ex))
    if denom <= 0:
        return np.zeros_like(x)
    return ex / denom


def init_model(model_path: str | None = None) -> bool:
    """Load PPE ONNX model once at startup.

    Returns True when model is ready; False when disabled or unavailable.
    """
    global _net, _model_path

    if not config.PPE_DETECTION_ENABLED:
        _net = None
        _model_path = ""
        return False

    path = model_path or config.PPE_MODEL_PATH
    if not path or not os.path.isfile(path):
        logger.warning("PPE detection enabled but model file not found: %s", path)
        _net = None
        _model_path = ""
        return False

    try:
        net = cv2.dnn.readNetFromONNX(path)
        _net = net
        _model_path = path
        logger.info("PPE model loaded from %s", path)
        return True
    except Exception as exc:
        logger.error("Failed to load PPE model %s: %s", path, exc)
        _net = None
        _model_path = ""
        return False


def is_ready() -> bool:
    return _net is not None


def _clamp_bbox(
    frame: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    fh, fw = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    return x1, y1, x2, y2


def detect_ppe(
    frame_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
) -> dict:
    """Return PPE state and confidence for one face bbox."""
    base = {
        "state": "none",
        "confidence": 0.0,
        "mask_confidence": 0.0,
        "cap_confidence": 0.0,
        "model_ready": is_ready(),
    }

    if not config.PPE_DETECTION_ENABLED or _net is None:
        return base

    x1, y1, x2, y2 = _clamp_bbox(frame_bgr, bbox_xywh)
    if x2 <= x1 or y2 <= y1:
        return base

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return base

    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 255.0,
            size=(128, 128),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )
        _net.setInput(blob)
        out = _net.forward()
        vec = np.array(out).reshape(-1)
        if vec.size < 2:
            return base

        probs = _softmax(vec)

        if probs.size >= 4:
            p_none = float(probs[0])
            p_mask = float(probs[1])
            p_cap = float(probs[2])
            p_both = float(probs[3])
        elif probs.size == 3:
            p_none = float(probs[0])
            p_mask = float(probs[1])
            p_cap = float(probs[2])
            p_both = float(min(p_mask, p_cap))
        else:
            p_mask = float(probs[0])
            p_cap = float(probs[1])
            p_both = float(min(p_mask, p_cap))
            p_none = float(max(0.0, 1.0 - max(p_mask, p_cap)))

        mask_hit = p_mask >= config.PPE_MASK_THRESHOLD
        cap_hit = p_cap >= config.PPE_CAP_THRESHOLD

        if mask_hit and cap_hit:
            state = "both"
            conf = max(p_both, min(p_mask, p_cap))
        elif mask_hit:
            state = "mask"
            conf = p_mask
        elif cap_hit:
            state = "cap"
            conf = p_cap
        else:
            state = "none"
            conf = p_none

        return {
            "state": state,
            "confidence": float(conf),
            "mask_confidence": float(p_mask),
            "cap_confidence": float(p_cap),
            "model_ready": True,
        }
    except Exception as exc:
        logger.debug("PPE detection inference failed: %s", exc)
        return base
