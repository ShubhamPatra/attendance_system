"""One-time bootstrap script to create the first admin account."""

from __future__ import annotations

import argparse
import getpass

import app_core.database as database
from app_core.auth import hash_password


VALID_ROLES = {"admin", "teacher"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap AutoAttendance admin/teacher user")
    parser.add_argument("--username", required=True, help="Unique username")
    parser.add_argument("--email", default=None, help="Optional email address")
    parser.add_argument("--role", default="admin", choices=sorted(VALID_ROLES), help="Role to assign")
    parser.add_argument("--password", default=None, help="Optional password (unsafe in shell history)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    username = args.username.strip()
    if not username:
        print("Username is required.")
        return 1

    existing = database.get_user_by_username(username)
    if existing is not None:
        print(f"User '{username}' already exists.")
        return 0

    password = args.password
    if not password:
        password = getpass.getpass("Password: ")
        password_confirm = getpass.getpass("Confirm Password: ")
        if password != password_confirm:
            print("Passwords do not match.")
            return 1

    if len(password) < 8:
        print("Password must be at least 8 characters.")
        return 1

    user_id = database.insert_user(
        username=username,
        password_hash=hash_password(password),
        role=args.role,
        email=args.email,
    )
    print(f"Created user '{username}' with role '{args.role}' (id={user_id}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
