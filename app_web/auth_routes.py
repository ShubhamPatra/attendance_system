"""Authentication route registrations."""

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user


def register_auth_routes(bp):
    @bp.route("/login", methods=["GET", "POST"])
    def login():
        from app_web import routes as routes_module

        if current_user.is_authenticated:
            return redirect(url_for("main.dashboard"))

        if request.method == "GET":
            return render_template("login.html")

        username = routes_module.sanitize_string(request.form.get("username", "")).strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("login.html"), 400

        user = routes_module.authenticate_user(username, password)
        if user is None:
            flash("Invalid credentials.", "danger")
            return render_template("login.html"), 401

        login_user(user, remember=True)
        return redirect(url_for("main.dashboard"))

    @bp.route("/logout", methods=["GET"])
    def logout():
        logout_user()
        return redirect(url_for("main.login"))

    @bp.route("/api/auth/login", methods=["POST"])
    def api_login():
        from app_web import routes as routes_module

        if not routes_module._check_rate_limit("auth_login"):
            return routes_module._api_error("Rate limit exceeded.", 429)

        errors, data = routes_module.validate_required_fields(
            request.get_json(silent=True),
            ["username", "password"],
        )
        error_response = routes_module._first_error_response(errors)
        if error_response is not None:
            return error_response

        username = routes_module.sanitize_string(data["username"]).strip()
        user = routes_module.authenticate_user(username, data["password"])
        if user is None:
            return routes_module._api_error("Invalid credentials.", 401)

        login_user(user, remember=True)
        return jsonify(
            {
                "ok": True,
                "username": user.username,
                "role": user.role,
            }
        )

    @bp.route("/api/auth/logout", methods=["POST"])
    def api_logout():
        logout_user()
        return jsonify({"ok": True})
