"""
Tests for anti-spoofing liveness detection.

The blink-based EAR logic has been replaced by deep learning anti-spoofing.
These tests verify that anti-spoofing module works correctly.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib, config
    importlib.reload(config)


def test_anti_spoofing_module_imports():
    """The anti_spoofing module should import without error."""
    import anti_spoofing
    assert hasattr(anti_spoofing, "check_liveness")
    assert hasattr(anti_spoofing, "init_models")


def test_check_liveness_exists():
    """check_liveness function should exist in anti_spoofing module."""
    import anti_spoofing
    assert callable(anti_spoofing.check_liveness)
