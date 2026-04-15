"""Camera-stream page and MJPEG route registrations."""

import time

import cv2
from flask import Response, render_template

from app_web.decorators import require_roles


def register_camera_routes(bp):
    @bp.route("/attendance")
    @require_roles("admin", "teacher")
    def attendance():
        return render_template("attendance.html")

    @bp.route("/video_feed")
    @require_roles("admin", "teacher")
    def video_feed():
        from app_web import routes as routes_module

        from app_camera.camera import acquire_camera_stream, release_camera_stream

        source = 0
        cam = acquire_camera_stream(source)
        min_interval = 1.0 / routes_module.config.MJPEG_TARGET_FPS

        def generate():
            last_yield = 0.0
            try:
                while True:
                    jpeg = cam.get_latest_jpeg()
                    if jpeg is None:
                        time.sleep(0.03)
                        continue
                    now = time.monotonic()
                    elapsed = now - last_yield
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                    last_yield = time.monotonic()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
            finally:
                release_camera_stream(source)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")