"""Camera-stream page and MJPEG route registrations."""

import time

import cv2
from flask import Response, render_template, url_for, jsonify

from web.decorators import require_roles
import core.config as config
import core.database as database


def register_camera_routes(bp):
    @bp.route("/attendance")
    # @require_roles("admin", "teacher")  # DISABLED: Auth removed for local testing
    def attendance():
        # Ensure an active attendance session exists for camera 0
        try:
            session = database.get_active_attendance_session(camera_id="0")
            if session is None:
                # Create a new session automatically
                course_id = config.DEFAULT_COURSE_ID if hasattr(config, 'DEFAULT_COURSE_ID') else "default"
                try:
                    database.create_attendance_session(course_id, "0")
                except ValueError as e:
                    # Session might have been created by another request
                    pass
        except Exception as e:
            # Log but don't fail the page load
            pass
        
        return render_template("attendance.html", video_feed_url=url_for("main.video_feed"))

    @bp.route("/api/attendance/start-session", methods=["POST"])
    # @require_roles("admin", "teacher")  # DISABLED: Auth removed for local testing
    def start_attendance_session():
        """API endpoint to start a new attendance session."""
        try:
            camera_id = "0"
            course_id = config.DEFAULT_COURSE_ID if hasattr(config, 'DEFAULT_COURSE_ID') else "default"
            
            session = database.get_active_attendance_session(camera_id=camera_id)
            if session:
                return jsonify({
                    "success": True,
                    "session_id": str(session.get("_id")),
                    "message": "Attendance session already active"
                })
            
            session_id = database.create_attendance_session(course_id, camera_id)
            return jsonify({
                "success": True,
                "session_id": str(session_id),
                "message": "Attendance session started successfully"
            })
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 409
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to start session: {str(e)}"
            }), 500

    @bp.route("/video_feed")
    # @require_roles("admin", "teacher")  # DISABLED: Auth removed for local testing
    def video_feed():
        from web import routes as routes_module

        from camera.camera import acquire_camera_stream, release_camera_stream

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