"""
Tests for utility functions.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib, config
    importlib.reload(config)


def test_sanitize_string_strips_html():
    from utils import sanitize_string
    assert sanitize_string("<script>alert('xss')</script>Hello") == "alert('xss')Hello"


def test_sanitize_string_truncates():
    from utils import sanitize_string
    long = "a" * 300
    assert len(sanitize_string(long, max_length=100)) == 100


def test_sanitize_string_collapses_whitespace():
    from utils import sanitize_string
    assert sanitize_string("  hello   world  ") == "hello world"


def test_allowed_file_valid():
    from utils import allowed_file
    assert allowed_file("photo.jpg", {"jpg", "png"})
    assert allowed_file("photo.PNG", {"jpg", "png"})


def test_allowed_file_invalid():
    from utils import allowed_file
    assert not allowed_file("virus.exe", {"jpg", "png"})
    assert not allowed_file("noext", {"jpg", "png"})


def test_today_str_format():
    from utils import today_str
    result = today_str()
    assert len(result) == 10
    assert result[4] == "-" and result[7] == "-"


def test_now_time_str_format():
    from utils import now_time_str
    result = now_time_str()
    assert len(result) == 8
    assert result[2] == ":" and result[5] == ":"


def test_setup_logging_returns_logger():
    from utils import setup_logging
    logger = setup_logging()
    assert logger.name == "attendance_system"
    # Calling again returns the same logger
    logger2 = setup_logging()
    assert logger is logger2


# ── Image quality checks ──────────────────────────────────────────────────

def test_check_image_quality_passes():
    """A well-lit, sharp image should pass."""
    from utils import check_image_quality
    # Create a bright, textured image (enough variance)
    rng = np.random.RandomState(42)
    img = rng.randint(100, 255, (200, 200, 3), dtype=np.uint8)
    ok, reason = check_image_quality(img)
    assert ok is True
    assert reason == ""


def test_check_image_quality_rejects_blur():
    """A perfectly flat (zero variance) image is too blurry."""
    from utils import check_image_quality
    # Solid gray — Laplacian variance ≈ 0
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, reason = check_image_quality(img)
    assert ok is False
    assert "blurry" in reason.lower()


def test_check_image_quality_rejects_dark():
    """A very dark image with texture should fail brightness check."""
    from utils import check_image_quality
    # Dark textured image (brightness < 40)
    rng = np.random.RandomState(7)
    img = rng.randint(0, 20, (200, 200, 3), dtype=np.uint8)
    ok, reason = check_image_quality(img)
    assert ok is False
    assert "dark" in reason.lower()
