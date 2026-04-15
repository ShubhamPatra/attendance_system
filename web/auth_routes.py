"""Authentication route registrations."""

from flask import jsonify, redirect, render_template, request, url_for


def register_auth_routes(bp):
    @bp.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")
        # No auth required - just redirect to dashboard
        return redirect(url_for("main.dashboard"))

    @bp.route("/logout", methods=["GET"])
    def logout():
        return redirect(url_for("main.login"))

    @bp.route("/api/auth/login", methods=["POST"])
    def api_login():
        # No auth required - anyone can proceed
        return jsonify({"ok": True})

    @bp.route("/api/auth/logout", methods=["POST"])
    def api_logout():
        return jsonify({"ok": True})
