from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = BASE_DIR / "venv" / "Lib" / "site-packages"

if VENV_SITE_PACKAGES.exists():
	sys.path.insert(0, str(VENV_SITE_PACKAGES))

from student_app import create_app
from core import socketio

app = create_app()


if __name__ == "__main__":
	socketio.run(app, host="0.0.0.0", port=5001)
