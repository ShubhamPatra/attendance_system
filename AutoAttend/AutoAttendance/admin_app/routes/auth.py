from __future__ import annotations

from functools import wraps

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import UserMixin, current_user, login_required, login_user, logout_user

from admin_app.forms import ChangePasswordForm, LoginForm
from core import limiter, login_manager
from core.models import AdminDAO
from core.utils import check_password, hash_password


auth_bp = Blueprint("admin_auth", __name__)


class AdminUser(UserMixin):
	def __init__(self, doc):
		self.id = str(doc["_id"])
		self.email = doc["email"]
		self.role = doc.get("role", "viewer")
		self.name = doc.get("name", "Admin")


def role_required(*roles):
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			if not current_user.is_authenticated:
				return login_manager.unauthorized()
			if current_user.role not in roles:
				abort(403)
			return func(*args, **kwargs)

		return wrapper

	return decorator


@login_manager.user_loader
def load_user(user_id: str):
	from admin_app import get_admin_dao

	admin = get_admin_dao().get_by_id(user_id)
	return AdminUser(admin) if admin else None


@auth_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per 15 minutes")
def login():
	from admin_app import get_admin_dao

	if current_user.is_authenticated:
		return redirect(url_for("admin_dashboard.dashboard"))

	form = LoginForm()
	if form.validate_on_submit():
		dao: AdminDAO = get_admin_dao()
		admin = dao.get_by_email(form.email.data or "")
		if admin and check_password(form.password.data or "", admin["password_hash"]):
			login_user(AdminUser(admin), remember=bool(form.remember_me.data))
			dao.update_last_login(str(admin["_id"]))
			return redirect(url_for("admin_dashboard.dashboard"))
		flash("Invalid credentials", "error")

	return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
	logout_user()
	return redirect(url_for("admin_auth.login"))


@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
	from admin_app import get_admin_dao

	form = ChangePasswordForm()
	if form.validate_on_submit():
		dao: AdminDAO = get_admin_dao()
		admin = dao.get_by_id(current_user.id)
		if not admin or not check_password(form.old_password.data or "", admin["password_hash"]):
			flash("Current password is incorrect", "error")
		else:
			dao.update_admin(current_user.id, {"password_hash": hash_password(form.new_password.data or "")})
			flash("Password updated", "success")
			return redirect(url_for("admin_dashboard.dashboard"))

	return render_template("auth/change_password.html", form=form)
