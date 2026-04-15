"""
Tests for multi-camera support and WebSocket event emission.
Uses mocks - no live webcam or network needed.
"""

import os
import sys
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _mock_config(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://test:test@cluster.mongodb.net/test")
    import importlib
    import core.config as config
    importlib.reload(config)


@pytest.fixture(autouse=True)
def _reset_cameras():
    """Reset the global camera dict between tests."""
    import camera.camera as camera
    camera.release_camera()
    camera._cameras.clear()
    camera._camera_viewers.clear()
    camera._socketio = None
    yield
    camera.release_camera()
    camera._cameras.clear()
    camera._camera_viewers.clear()
    camera._socketio = None


# ---------------------------------------------------------------------------
# 1. get_camera creates (and caches) Camera instances
# ---------------------------------------------------------------------------

def test_get_camera_creates_instance():
    """get_camera(0) should return a Camera; calling again returns the same one."""
    import camera.camera as camera

    with patch("app_camera.camera.cv2.VideoCapture") as mock_vc:
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
    import camera.camera as camera

    with patch("app_camera.camera.cv2.VideoCapture") as mock_vc:
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
    import camera.camera as camera

    with patch("app_camera.camera.cv2.VideoCapture") as mock_vc:
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


def test_stream_acquire_release_stops_on_last_viewer():
    """acquire/release stream should stop camera only after last viewer leaves."""
    import camera.camera as camera

    with patch("app_camera.camera.cv2.VideoCapture") as mock_vc:
        mock_vc.return_value = MagicMock()

        with patch.object(camera.Camera, "start"), \
             patch.object(camera.Camera, "stop") as mock_stop:
            cam = camera.acquire_camera_stream(0)
            camera.acquire_camera_stream(0)

            assert camera._camera_viewers[0] == 2
            camera.release_camera_stream(0)
            assert camera._camera_viewers[0] == 1
            mock_stop.assert_not_called()

            camera.release_camera_stream(0)
            assert 0 not in camera._camera_viewers
            assert 0 not in camera._cameras
            mock_stop.assert_called_once()


def test_camera_manager_diagnostics_reports_fps():
    import camera.camera as camera

    manager = camera.CameraManager()
    fake_cam = MagicMock()
    fake_cam.diagnostics.return_value = {"source": 0, "fps": 11.5}
    manager._cameras[0] = fake_cam
    manager._camera_viewers[0] = 2

    data = manager.diagnostics()

    assert data["active_cameras"] == 1
    assert data["viewers"][0] == 2
    assert data["cameras"][0]["fps"] == 11.5


# ---------------------------------------------------------------------------
# 5. set_socketio + _emit_event
# ---------------------------------------------------------------------------

def test_socketio_emit():
    """_emit_event should delegate to the stored SocketIO instance."""
    import camera.camera as camera

    mock_sio = MagicMock()
    camera.set_socketio(mock_sio)

    camera._emit_event("test", {"data": 1})
    mock_sio.emit.assert_called_once_with("test", {"data": 1}, namespace="/")


# ---------------------------------------------------------------------------
# 6. _push_event emits via WebSocket AND stores in pop_events buffer
# ---------------------------------------------------------------------------

def test_push_event_emits_websocket():
    """_push_event should store the event and emit via WebSocket."""
    import camera.camera as camera

    mock_sio = MagicMock()
    camera.set_socketio(mock_sio)

    with patch("app_camera.camera.cv2.VideoCapture") as mock_vc:
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
    import camera.camera as camera

    with patch("app_camera.camera.cv2.VideoCapture") as mock_vc:
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
    import camera.camera as camera

    with patch("app_camera.camera.cv2.VideoCapture") as mock_vc:
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


@patch("app_camera.camera.cv2.VideoCapture")
def test_incremental_learning_requires_min_liveness(mock_vc):
    """High recognition alone must not append encoding when liveness is low."""
    import camera.camera as camera

    mock_vc.return_value = MagicMock()

    with patch("app_camera.camera.database.get_student_by_id", return_value=None), \
         patch("app_camera.camera.database.mark_attendance", return_value=True), \
         patch("app_camera.camera._encoding_cache") as mock_cache_factory, \
         patch("app_camera.camera._face_engine_module") as mock_face_engine_factory:
        mock_cache = MagicMock()
        mock_cache.get_flat.return_value = (None, None, [], [])
        mock_cache_factory.return_value = mock_cache

        mock_face_engine = MagicMock()
        mock_face_engine_factory.return_value = mock_face_engine

        cam = camera.Camera(source=0)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        encoding = np.random.rand(128).astype(np.float64)

        cam._handle_recognized(
            student_id="sid-1",
            name="Alice",
            confidence=0.99,
            liveness_conf=0.10,
            frame=frame,
            encoding=encoding,
        )

        mock_face_engine.append_encoding.assert_not_called()
