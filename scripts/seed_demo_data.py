"""Seed a small demo dataset for reviewers and local smoke testing."""

from __future__ import annotations

import random

import numpy as np

import core.database as database
from core.auth import hash_password


def _student_encodings(count: int = 2) -> list[np.ndarray]:
    return [np.random.rand(128).astype(np.float64) for _ in range(count)]


def main() -> int:
    demo_students = [
        {
            "name": "Alice Khan",
            "semester": 3,
            "registration_number": "DEMO-001",
            "section": "A",
            "email": "alice@example.com",
        },
        {
            "name": "Bilal Ahmed",
            "semester": 3,
            "registration_number": "DEMO-002",
            "section": "A",
            "email": "bilal@example.com",
        },
        {
            "name": "Sara Iqbal",
            "semester": 5,
            "registration_number": "DEMO-003",
            "section": "B",
            "email": "sara@example.com",
        },
    ]

    for row in demo_students:
        if database.get_student_by_reg_no(row["registration_number"]) is None:
            database.insert_student(
                row["name"],
                row["semester"],
                row["registration_number"],
                row["section"],
                _student_encodings(),
                email=row["email"],
            )

    if database.get_user_by_username("admin") is None:
        database.insert_user(
            username="admin",
            password_hash=hash_password("admin1234"),
            role="admin",
            email="admin@example.com",
        )

    present_ids = []
    for student in database.get_all_students():
        if random.random() > 0.35:
            present_ids.append(student["_id"])

    entries = [
        {"student_id": student_id, "status": "Present", "confidence_score": 0.92}
        for student_id in present_ids
    ]
    if entries:
        database.bulk_upsert_attendance(entries)

    print("Demo data seeded successfully.")
    print("Default demo admin: admin / admin1234")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
