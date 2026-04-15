from flask import Flask
from flask_login import UserMixin, login_user

from admin_app.routes.auth import role_required
from core import login_manager


class DummyUser(UserMixin):
	def __init__(self, user_id: str, role: str):
		self.id = user_id
		self.role = role


def build_app():
	app = Flask(__name__)
	app.config["SECRET_KEY"] = "test"
	app.config["TESTING"] = True

	login_manager.init_app(app)
	login_manager.login_view = "login_page"

	users = {
		"admin": DummyUser("1", "admin"),
		"viewer": DummyUser("2", "viewer"),
		"super_admin": DummyUser("3", "super_admin"),
	}

	@login_manager.user_loader
	def load_user(user_id):
		for user in users.values():
			if user.id == user_id:
				return user
		return None

	@app.get("/login")
	def login_page():
		return "login", 200

	@app.get("/test-login/<role>")
	def test_login(role):
		user = users[role]
		login_user(user)
		return "ok", 200

	@app.get("/protected")
	@role_required("super_admin", "admin")
	def protected():
		return "allowed", 200

	return app


def test_role_required_blocks_anonymous():
	app = build_app()
	client = app.test_client()

	response = client.get("/protected")
	assert response.status_code in (301, 302)


def test_role_required_blocks_viewer():
	app = build_app()
	client = app.test_client()

	client.get("/test-login/viewer")
	response = client.get("/protected")
	assert response.status_code == 403


def test_role_required_allows_admin_and_super_admin():
	app = build_app()
	client = app.test_client()

	client.get("/test-login/admin")
	assert client.get("/protected").status_code == 200

	client = app.test_client()
	client.get("/test-login/super_admin")
	assert client.get("/protected").status_code == 200


def test_admin_login_success_and_invalid_password(admin_client):
	success = admin_client.post(
		"/admin/login",
		data={"email": "admin@example.com", "password": "Secret123!", "remember_me": "y"},
		follow_redirects=False,
	)
	assert success.status_code == 302
	assert "/admin/dashboard" in success.headers.get("Location", "")

	logout = admin_client.get("/admin/logout", follow_redirects=False)
	assert logout.status_code == 302

	failure = admin_client.post(
		"/admin/login",
		data={"email": "admin@example.com", "password": "wrong-password"},
		follow_redirects=False,
	)
	assert failure.status_code == 200
	assert "Location" not in failure.headers


def test_admin_login_rate_limit_blocks_sixth_attempt(admin_client):
	response = None
	for _ in range(5):
		response = admin_client.post(
			"/admin/login",
			data={"email": "admin@example.com", "password": "wrong-password"},
			follow_redirects=False,
		)
		assert response.status_code in (200, 302)

	response = admin_client.post(
		"/admin/login",
		data={"email": "admin@example.com", "password": "wrong-password"},
		follow_redirects=False,
	)
	assert response.status_code == 429


def test_admin_rbac_viewer_blocked(admin_client):
	login = admin_client.post(
		"/admin/login",
		data={"email": "viewer@example.com", "password": "Secret123!"},
		follow_redirects=False,
	)
	assert login.status_code == 302

	response = admin_client.get("/admin/students")
	assert response.status_code == 403


def test_admin_session_invalidation_redirects_to_login(admin_client):
	admin_client.post(
		"/admin/login",
		data={"email": "admin@example.com", "password": "Secret123!"},
		follow_redirects=False,
	)
	with admin_client.session_transaction() as session:
		session.clear()

	response = admin_client.get("/admin/dashboard", follow_redirects=False)
	assert response.status_code in (301, 302)
	assert "/admin/login" in response.headers.get("Location", "")


def test_student_login_registration_and_invalid_password(student_client, test_db):
	invalid = student_client.post(
		"/student/login",
		data={"student_id": "CS2026001", "password": "wrong-password"},
		follow_redirects=False,
	)
	assert invalid.status_code == 200

	register = student_client.post(
		"/student/register",
		data={
			"name": "New Student",
			"student_id": "CS2026999",
			"email": "newstudent@example.com",
			"password": "Password123",
			"confirm_password": "Password123",
			"department": "CS",
			"year": "2",
			"face_photo_path": "",
		},
		follow_redirects=False,
	)
	assert register.status_code == 302
	assert "/student/profile" in register.headers.get("Location", "")
	assert test_db.students.find_one({"student_id": "CS2026999"}) is not None

	success = student_client.post(
		"/student/login",
		data={"student_id": "CS2026001", "password": "Secret123!"},
		follow_redirects=False,
	)
	assert success.status_code == 302
	assert "/student/profile" in success.headers.get("Location", "")


def test_student_login_rate_limit_blocks_sixth_attempt(student_client):
	for _ in range(5):
		response = student_client.post(
			"/student/login",
			data={"student_id": "CS2026001", "password": "wrong-password"},
			follow_redirects=False,
		)
		assert response.status_code in (200, 302)

	response = student_client.post(
		"/student/login",
		data={"student_id": "CS2026001", "password": "wrong-password"},
		follow_redirects=False,
	)
	assert response.status_code == 429
