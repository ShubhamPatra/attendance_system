"""
Tests for multi-camera support and WebSocket event emission.
Uses mocks - no live webcam or network needed.
"""

import os
import sys
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib, config
    importlib.reload(config)


@pytest.fixture(autouse=True)
def _reset_cameras():
    """Reset the global camera dict between tests."""
    import camera
    camera._cameras = {}
    camera._socketio = None
    yield
    camera._cameras = {}
    camera._socketio = None


# ---------------------------------------------------------------------------
# 1. get_camera creates (and caches) Camera instances
# ---------------------------------------------------------------------------

def test_get_camera_creates_instance():
    """get_camera(0) should return a Camera; calling again returns the same one."""
    import camera

    with patch("camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        # Patch start() so no background thread is spawned
        with patch.object(camera.Camera, "start"):
            cam1 = camera.get_camera(0)
            assert isinstance(cam1, camera.Camera)

            cam2 = camera.get_camera(0)
            assert cam2 is cam1, "Second call should return the cached instance"


# ---------------------------------------------------------------------------
# 2. Different source indices yield different Camera instances
# ---------------------------------------------------------------------------

def test_get_camera_different_sources():
    """get_camera(0) and get_camera(1) should be distinct instances."""
    import camera

    with patch("camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        with patch.object(camera.Camera, "start"):
            cam0 = camera.get_camera(0)
            cam1 = camera.get_camera(1)
            assert cam0 is not cam1, "Different sources must produce different instances"


# ---------------------------------------------------------------------------
# 3. release_camera removes from dict
# ---------------------------------------------------------------------------

def test_release_camera():
    """release_camera(source) removes one; release_camera() removes all."""
    import camera

    with patch("camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        with patch.object(camera.Camera, "start"), \
             patch.object(camera.Camera, "stop"):
            cam0 = camera.get_camera(0)
            cam1 = camera.get_camera(1)
            assert len(camera._cameras) == 2

            # Release only camera 0
            camera.release_camera(0)
            assert 0 not in camera._cameras
            assert 1 in camera._cameras

            # Release all remaining cameras
            camera.release_camera()
            assert len(camera._cameras) == 0


# ---------------------------------------------------------------------------
# 4. Subject management
# ---------------------------------------------------------------------------

def test_set_subject():
    """set_subject / get_subject round-trip through the Camera instance."""
    import camera

    with patch("camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        cam = camera.Camera(source=0)
        assert cam.get_subject() == "General", "Default subject should be 'General'"

        cam.set_subject("Mathematics")
        assert cam.get_subject() == "Mathematics"


# ---------------------------------------------------------------------------
# 5. set_socketio + _emit_event
# ---------------------------------------------------------------------------

def test_socketio_emit():
    """_emit_event should delegate to the stored SocketIO instance."""
    import camera

    mock_sio = MagicMock()
    camera.set_socketio(mock_sio)

    camera._emit_event("test", {"data": 1})
    mock_sio.emit.assert_called_once_with("test", {"data": 1}, namespace="/")


# ---------------------------------------------------------------------------
# 6. _push_event emits via WebSocket AND stores in pop_events buffer
# ---------------------------------------------------------------------------

def test_push_event_emits_websocket():
    """_push_event should store the event and emit via WebSocket."""
    import camera

    mock_sio = MagicMock()
    camera.set_socketio(mock_sio)

    with patch("camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        cam = camera.Camera(source=0)
        event = {"name": "Alice", "status": "marked", "confidence": 0.95}
        cam._push_event(event)

        # Event should appear in the pop_events buffer
        events = cam.pop_events()
        assert len(events) == 1
        assert events[0]["name"] == "Alice"

        # WebSocket emit should have been called (attendance_event + log_event)
        call_args_list = mock_sio.emit.call_args_list
        event_names = [call[0][0] for call in call_args_list]
        assert "attendance_event" in event_names
        assert "log_event" in event_names


# ---------------------------------------------------------------------------
# 7. pop_events clears the buffer
# ---------------------------------------------------------------------------

def test_pop_events_clears_buffer():
    """After pop_events(), a second pop should return an empty list."""
    import camera

    with patch("camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        cam = camera.Camera(source=0)
        cam._push_event({"name": "Bob", "status": "marked"})
        cam._push_event({"name": "Carol", "status": "marked"})

        first_pop = cam.pop_events()
        assert len(first_pop) == 2

        second_pop = cam.pop_events()
        assert len(second_pop) == 0, "Buffer should be empty after first pop"


# ---------------------------------------------------------------------------
# 8. Log buffer is persistent (non-destructive reads)
# ---------------------------------------------------------------------------

def test_log_buffer_persists():
    """get_log_buffer returns all entries and does NOT clear them."""
    import camera

    with patch("camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        cam = camera.Camera(source=0)
        cam._push_event({"name": "Dave", "status": "marked"})
        cam._push_event({"name": "Eve", "status": "marked"})

        first_read = cam.get_log_buffer()
        assert len(first_read) == 2

        # Second read should still return the same entries (non-destructive)
        second_read = cam.get_log_buffer()
        assert len(second_read) == 2
        assert first_read[0]["name"] == second_read[0]["name"]
        assert first_read[1]["name"] == second_read[1]["name"]
