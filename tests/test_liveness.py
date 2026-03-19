"""
Tests for liveness module (deprecated – now a thin wrapper around anti_spoofing).

The blink-based EAR logic has been replaced by deep learning anti-spoofing.
These tests verify that the deprecated module still imports cleanly and
re-exports the expected symbols.
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


def test_liveness_module_imports():
    """The deprecated liveness.py should import without error."""
    import liveness
    assert hasattr(liveness, "check_liveness")
    assert hasattr(liveness, "init_models")


def test_check_liveness_reexported():
    """check_liveness from liveness should be the same as from anti_spoofing."""
    import liveness
    import anti_spoofing
    assert liveness.check_liveness is anti_spoofing.check_liveness
