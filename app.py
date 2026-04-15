"""Backward-compatible entrypoint for the grouped admin app module."""

from admin_app.app import create_app, socketio

__all__ = ["create_app", "socketio"]


if __name__ == "__main__":
    import core.config as config

    app = create_app()
    socketio.run(
        app,
        host=config.APP_HOST,
        port=config.APP_PORT,
        debug=config.APP_DEBUG,
        use_reloader=False,
    )
