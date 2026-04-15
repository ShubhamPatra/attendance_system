"""Student portal runner using the migrated package layout."""

from student_app.app import create_app
import app_core.config as config


if __name__ == "__main__":
    app = create_app()
    app.run(host=config.STUDENT_APP_HOST, port=config.STUDENT_APP_PORT, debug=config.APP_DEBUG)
