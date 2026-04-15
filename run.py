"""
Unified entry point to run both Admin and Student panels simultaneously.

This script starts both applications on their configured ports:
- Admin Panel: http://localhost:5000 (configurable via APP_PORT)
- Student Portal: http://localhost:5001 (configurable via STUDENT_APP_PORT)
"""

import sys
import threading
import time
from multiprocessing import Process

import core.config as config
from admin_app.app import create_app as create_admin_app, socketio as admin_socketio
from student_app.app import create_app as create_student_app


def run_admin():
    """Run the admin application."""
    try:
        app = create_admin_app()
        print(f"\n✓ Admin Panel starting on http://{config.APP_HOST}:{config.APP_PORT}")
        admin_socketio.run(
            app,
            host=config.APP_HOST,
            port=config.APP_PORT,
            debug=config.APP_DEBUG,
            use_reloader=False,
        )
    except Exception as e:
        print(f"✗ Admin Panel failed to start: {e}", file=sys.stderr)
        sys.exit(1)


def run_student():
    """Run the student application."""
    try:
        app = create_student_app()
        print(f"✓ Student Portal starting on http://{config.STUDENT_APP_HOST}:{config.STUDENT_APP_PORT}")
        app.run(
            host=config.STUDENT_APP_HOST,
            port=config.STUDENT_APP_PORT,
            debug=config.APP_DEBUG,
            use_reloader=False,
        )
    except Exception as e:
        print(f"✗ Student Portal failed to start: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Start both admin and student applications."""
    print("\n" + "=" * 70)
    print("  ATTENDANCE SYSTEM - Unified Entry Point")
    print("=" * 70)
    print(f"\nStarting both Admin Panel and Student Portal...\n")

    # Create processes for each app
    admin_process = Process(target=run_admin, daemon=False)
    student_process = Process(target=run_student, daemon=False)

    try:
        # Start both processes
        admin_process.start()
        student_process.start()

        print("\n" + "=" * 70)
        print("  Both applications are running!")
        print("=" * 70)
        print(f"\n📊 Admin Panel:      http://localhost:{config.APP_PORT}")
        print(f"👤 Student Portal:   http://localhost:{config.STUDENT_APP_PORT}")
        print(f"\nPress Ctrl+C to stop all services.\n")

        # Keep the main process alive
        admin_process.join()
        student_process.join()

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        admin_process.terminate()
        student_process.terminate()

        # Wait for processes to terminate gracefully
        admin_process.join(timeout=5)
        student_process.join(timeout=5)

        # Force kill if still running
        if admin_process.is_alive():
            admin_process.kill()
            admin_process.join()
        if student_process.is_alive():
            student_process.kill()
            student_process.join()

        print("✓ All services stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        admin_process.terminate()
        student_process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
