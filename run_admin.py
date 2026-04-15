"""Admin application runner using the migrated package layout."""

from admin_app.app import create_app, socketio
import core.config as config


if __name__ == "__main__":
    app = create_app()
    socketio.run(app, host=config.APP_HOST, port=config.APP_PORT, debug=config.APP_DEBUG)