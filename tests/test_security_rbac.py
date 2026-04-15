"""Auth-disabled behavior checks with ENABLE_RBAC=0."""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _rbac_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    monkeypatch.setenv("ENABLE_RBAC", "0")
    monkeypatch.setenv("ENABLE_RESTX_API", "1")

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_fr = types.SimpleNamespace(
        face_locations=lambda *args, **kwargs: [],
        face_encodings=lambda *args, **kwargs: [],
        face_landmarks=lambda *args, **kwargs: [],
        load_image_file=lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "face_recognition", fake_fr)

    import importlib
    import app_core.config as config

    importlib.reload(config)


@pytest.fixture
def client():
    with patch("app_core.database.get_client") as mock_client, \
         patch("app_core.database.ensure_indexes"), \
         patch("app_vision.face_engine.encoding_cache") as mock_cache, \
         patch("app_vision.anti_spoofing.init_models"):

        mock_client.return_value = MagicMock()
        mock_cache.load = MagicMock()
        mock_cache.size = 0
        mock_cache.get_all.return_value = ([], [], [])

        from app import create_app

        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_v1_and_v2_endpoints_are_accessible_without_auth(client):
    routes = [
        "/api/metrics",
        "/api/events",
        "/api/logs",
        "/api/cameras",
        "/api/v2/ops/metrics",
        "/api/v2/attendance",
        "/api/v2/cameras",
    ]

    for endpoint in routes:
        resp = client.get(endpoint)
        assert resp.status_code == 200, endpoint


def test_analytics_endpoints_are_accessible_without_auth(client):
    browser_routes = [
        "/logs",
        "/metrics",
        "/attendance_activity",
        "/heatmap",
    ]
    api_routes = [
        "/api/attendance_activity",
        "/api/heatmap",
    ]

    for endpoint in browser_routes:
        resp = client.get(endpoint)
        assert resp.status_code == 200, endpoint

    for endpoint in api_routes:
        resp = client.get(endpoint)
        assert resp.status_code == 200, endpoint


def test_v2_public_attendance_activity_remains_accessible(client):
    resp = client.get("/api/v2/public/attendance-activity")
    assert resp.status_code == 200
