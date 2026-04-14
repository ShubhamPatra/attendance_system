"""Shared helper functions for web route modules."""

import collections
import os
import threading
import time
from datetime import datetime

from flask import jsonify, request

import app_core.config as config
import app_core.database as database
from app_core.utils import sanitize_string

# Magic bytes for image MIME validation
_IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG": "png",
}

_rate_limit_lock = threading.Lock()
_rate_limit_buckets: dict[str, collections.deque[float]] = {}
_last_bucket_prune: float = 0.0


def _api_error(message: str, status_code: int = 400):
    """Return a standard JSON error payload."""
    return jsonify({"error": message}), status_code


def _api_errors(messages: list[str], status_code: int = 400):
    """Return a standard JSON payload for multiple validation errors."""
    return jsonify({"errors": messages}), status_code


def _first_error_response(errors: list[str], status_code: int = 400):
    """Return first validation error if present, else None."""
    if not errors:
        return None
    return _api_error(errors[0], status_code)


def _is_within_directory(path: str, root: str) -> bool:
    """Return True when *path* is inside *root* after normalization."""
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _resolve_upload_reference(value: str) -> str | None:
    """Resolve a client-supplied upload reference to a safe absolute path."""
    raw = value.strip()
    if not raw:
        return None

    upload_root = os.path.abspath(config.UPLOAD_DIR)
    if os.path.isabs(raw):
        candidate = os.path.abspath(raw)
    else:
        candidate = os.path.abspath(os.path.join(upload_root, os.path.basename(raw)))

    if not _is_within_directory(candidate, upload_root):
        return None
    return candidate


def _validate_image_mime(file_storage) -> str | None:
    """Read magic bytes and return the detected type, or None if invalid."""
    header = file_storage.read(8)
    file_storage.seek(0)
    for sig, fmt in _IMAGE_SIGNATURES.items():
        if header.startswith(sig):
            return fmt
    return None


def _validate_date_param(value: str, name: str) -> tuple[datetime | None, str | None]:
    """Parse a YYYY-MM-DD string. Returns (date, None) on success or (None, error_message) on failure."""
    try:
        return datetime.strptime(value, "%Y-%m-%d"), None
    except ValueError:
        return None, f"'{name}' must be a valid date in YYYY-MM-DD format."


def _rate_limit_key(endpoint: str) -> str:
    client = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    client = client.split(",", 1)[0].strip()
    return f"{endpoint}:{client}"


def _check_rate_limit(endpoint: str) -> bool:
    """Return True if the current request is within endpoint rate limits."""
    global _last_bucket_prune
    now = time.monotonic()
    window = max(config.API_RATE_LIMIT_WINDOW_SEC, 1)
    limit = max(config.API_RATE_LIMIT_MAX_REQUESTS, 1)
    key = _rate_limit_key(endpoint)

    with _rate_limit_lock:
        if now - _last_bucket_prune >= window:
            global_cutoff = now - window
            stale_keys: list[str] = []
            for bucket_key, bucket_values in _rate_limit_buckets.items():
                while bucket_values and bucket_values[0] < global_cutoff:
                    bucket_values.popleft()
                if not bucket_values:
                    stale_keys.append(bucket_key)
            for bucket_key in stale_keys:
                _rate_limit_buckets.pop(bucket_key, None)
            _last_bucket_prune = now

        bucket = _rate_limit_buckets.setdefault(key, collections.deque())
        cutoff = now - window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def _is_truthy_flag(value) -> bool:
    """Return True when *value* represents an enabled flag."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_report_filters(params: dict, *, default_date: str) -> tuple[dict | None, tuple[dict, int] | None]:
    """Normalize and validate report filters for CSV/XLSX/async endpoints."""
    reg_no = sanitize_string(str(params.get("reg_no", "")).strip())
    start_date = str(params.get("start_date", "")).strip()
    end_date = str(params.get("end_date", "")).strip()
    date = str(params.get("date", "")).strip()
    full = _is_truthy_flag(params.get("full", ""))

    if full:
        return {"mode": "full"}, None

    if reg_no:
        if database.get_student_by_reg_no(reg_no) is None:
            return None, ({"error": "Student not found."}, 404)
        return {"mode": "student", "reg_no": reg_no}, None

    if bool(start_date) ^ bool(end_date):
        return None, ({"error": "Both start_date and end_date are required for range queries."}, 400)

    if start_date and end_date:
        sd, err1 = _validate_date_param(start_date, "start_date")
        ed, err2 = _validate_date_param(end_date, "end_date")
        if err1 or err2:
            return None, ({"error": err1 or err2}, 400)
        if sd > ed:
            return None, ({"error": "start_date must be <= end_date."}, 400)
        return {
            "mode": "range",
            "start_date": start_date,
            "end_date": end_date,
        }, None

    date_str = date or default_date
    _, err = _validate_date_param(date_str, "date")
    if err:
        return None, ({"error": err}, 400)
    return {"mode": "date", "date": date_str}, None