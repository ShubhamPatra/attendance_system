"""
Utility helpers: input sanitisation, file validation, logging setup.
"""

import logging
import logging.handlers
import os
import re
from datetime import datetime, timezone

import cv2
import numpy as np


def setup_logging() -> logging.Logger:
    """Configure and return the application logger (file + console).

    A new log file is created each day (e.g. ``app_2026-03-06.log``).
    Uses :class:`~logging.handlers.TimedRotatingFileHandler` so the
    rollover happens automatically at midnight.
    """
    logger = logging.getLogger("attendance_system")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File – one log file per day, rotated at midnight
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.handlers.TimedRotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        when="midnight",
        interval=1,
        backupCount=30,          # keep last 30 days
        encoding="utf-8",
    )
    fh.suffix = "%Y-%m-%d"      # rolled files: app.log.2026-03-05
    fh.namer = lambda name: name.replace("app.log.", "app_") + ".log"
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Input sanitisation ─────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")


def sanitize_string(value: str, max_length: int = 200) -> str:
    """Strip HTML tags, collapse whitespace, truncate."""
    value = _TAG_RE.sub("", value)
    value = " ".join(value.split())
    return value[:max_length].strip()


# ── File validation ────────────────────────────────────────────────────────

def allowed_file(filename: str, allowed_extensions: set | None = None) -> bool:
    """Return True if *filename* has an allowed extension."""
    if allowed_extensions is None:
        from config import ALLOWED_EXTENSIONS
        allowed_extensions = ALLOWED_EXTENSIONS
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


# ── Date / time helpers ───────────────────────────────────────────────────

def today_str() -> str:
    """Return today's date as YYYY-MM-DD (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_time_str() -> str:
    """Return current time as HH:MM:SS (UTC)."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def validate_required_fields(payload: dict | None, required: list[str]) -> tuple[list[str], dict]:
    """Validate required JSON fields and return ``(errors, normalized_payload)``."""
    data = payload or {}
    errors: list[str] = []
    for field in required:
        value = data.get(field, None)
        if value is None:
            errors.append(f"Missing '{field}' in request body.")
        elif isinstance(value, str) and not value.strip():
            errors.append(f"'{field}' cannot be empty.")
    return errors, data


# ── Image quality checks ──────────────────────────────────────────────────

def check_image_quality(image: np.ndarray) -> tuple[bool, str]:
    """Validate image blur and brightness.

    *image* is a BGR numpy array (as read by OpenCV).
    Returns ``(True, "")`` if acceptable, or ``(False, reason)``.
    Uses same thresholds as check_face_quality_gate() for consistency.
    """
    from config import BLUR_THRESHOLD, BRIGHTNESS_THRESHOLD, BRIGHTNESS_MAX

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur detection via Laplacian variance
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < BLUR_THRESHOLD:
        return False, (
            f"Image is too blurry (sharpness {variance:.0f}, "
            f"minimum required {BLUR_THRESHOLD:.0f})."
        )

    # Brightness check (min and max for consistency with runtime pipeline)
    brightness = float(np.mean(gray))
    if brightness < BRIGHTNESS_THRESHOLD:
        return False, (
            f"Image is too dark (brightness {brightness:.0f}, "
            f"minimum required {BRIGHTNESS_THRESHOLD:.0f})."
        )
    if brightness > BRIGHTNESS_MAX:
        return False, (
            f"Image is too bright (brightness {brightness:.0f}, "
            f"maximum allowed {BRIGHTNESS_MAX:.0f})."
        )

    return True, ""
