"""
⚠️ DEVELOPMENT ONLY - DO NOT USE IN PRODUCTION ⚠️

Seed a small demo dataset for reviewers and local smoke testing.

This script creates fake/test students and demo attendance data. It is ONLY intended
for local development, testing, and code review. Never run this in production.

To use:
    python scripts/dev-only/seed_demo_data.py
    
    OR:
    
    python -m scripts.dev_only.seed_demo_data

This will create:
- Test students: DEMO-001, DEMO-002, DEMO-003
- Admin user: username=admin, password=admin1234
- Random attendance records (~65% present)

⚠️ This data is FAKE and for TESTING ONLY ⚠️
"""

from __future__ import annotations

import os
import random

import numpy as np

import core.database as database
from core.auth import hash_password


def _student_encodings(count: int = 2) -> list[np.ndarray]:
    return [np.random.rand(128).astype(np.float64) for _ in range(count)]


def main() -> int:
    # Safety check: prevent accidental production use
    if os.environ.get("APP_ENV") == "production":
        raise RuntimeError(
            "❌ Cannot seed demo data in production! "
            "This script is development-only. Set APP_ENV to something other than 'production'."
        )
    
    print("⚠️  SEEDING DEMO DATA - This is development-only data for testing only")
    print()
    
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

    print("✅ Demo data seeded successfully.")
    print("📝 Test Students Created:")
    print("   - DEMO-001: Alice Khan")
    print("   - DEMO-002: Bilal Ahmed")
    print("   - DEMO-003: Sara Iqbal")
    print()
    print("🔐 Demo Admin Account:")
    print("   - Username: admin")
    print("   - Password: admin1234")
    print()
    print("⚠️  Remember: This is TEST DATA ONLY - DO NOT USE IN PRODUCTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
