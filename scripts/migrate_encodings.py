#!/usr/bin/env python3
"""
Migrate student face encodings from dlib (128-D) to ArcFace (512-D).

This script reads every approved student document, re-encodes their
face samples using the ArcFace backend, and updates the encoding
arrays in MongoDB.  It is safe to re-run (idempotent — already-migrated
students with 512-D encodings are detected and skipped).

Usage
-----
    python scripts/migrate_encodings.py [--dry-run] [--student REG_NO]

Flags
-----
    --dry-run       Report what would happen without writing to the database.
    --student REG   Migrate only the student with registration number REG.
    --force         Re-encode even if the student already has 512-D encodings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import app_core.config as config
import app_core.database as database
from app_core.utils import setup_logging

logger = setup_logging()


def _already_migrated(doc: dict) -> bool:
    """Return True if the student already has ≥1 ArcFace (512-D) encoding."""
    if "encodings" not in doc or not doc["encodings"]:
        return False
    for raw in doc["encodings"]:
        nbytes = len(raw)
        if nbytes == 512 * 4:  # float32 × 512
            return True
    return False


def _collect_face_images(doc: dict) -> list[str]:
    """Return paths to face sample images for a student."""
    paths = []
    samples = doc.get("face_samples", [])
    for sample_path in samples:
        # Resolve relative to project root
        full = os.path.join(ROOT, sample_path) if not os.path.isabs(sample_path) else sample_path
        if os.path.isfile(full):
            paths.append(full)
    return paths


def migrate_student(doc: dict, backend, dry_run: bool = False) -> tuple[bool, str]:
    """Re-encode a single student's face samples using ArcFace.

    Returns (success: bool, message: str).
    """
    student_name = doc.get("name", "?")
    reg_no = doc.get("registration_number", "?")
    student_id = doc["_id"]

    image_paths = _collect_face_images(doc)
    if not image_paths:
        return False, f"No face sample images found for {student_name} ({reg_no})"

    new_encodings: list[np.ndarray] = []
    for path in image_paths:
        try:
            enc = backend.generate(path)
            new_encodings.append(enc)
        except Exception as exc:
            logger.warning(
                "Failed to encode image %s for %s: %s",
                path, student_name, exc,
            )

    if not new_encodings:
        return False, f"All images failed encoding for {student_name} ({reg_no})"

    if dry_run:
        return True, (
            f"[DRY RUN] Would update {student_name} ({reg_no}): "
            f"{len(new_encodings)}/{len(image_paths)} images → 512-D encodings"
        )

    # Write to database
    success = database.replace_student_encodings(reg_no, new_encodings)
    if success:
        return True, (
            f"Migrated {student_name} ({reg_no}): "
            f"{len(new_encodings)} ArcFace encodings stored"
        )
    return False, f"Database update failed for {student_name} ({reg_no})"


def main():
    parser = argparse.ArgumentParser(
        description="Migrate face encodings from dlib (128-D) to ArcFace (512-D)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report without modifying the database.",
    )
    parser.add_argument(
        "--student",
        type=str,
        default=None,
        help="Only migrate the student with this registration number.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if already migrated to 512-D.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ArcFace Encoding Migration")
    logger.info("=" * 60)

    # Load the ArcFace backend
    from app_vision.face_engine import ArcFaceEmbeddingBackend
    backend = ArcFaceEmbeddingBackend()
    backend._ensure_loaded()
    logger.info("ArcFace backend ready.")

    # Query students
    db = database.get_db()
    query: dict = {}
    if args.student:
        query["registration_number"] = args.student
    else:
        query["$or"] = [
            {"verification_status": {"$exists": False}},
            {"verification_status": "approved"},
        ]

    students = list(db.students.find(
        query,
        {"name": 1, "registration_number": 1, "encodings": 1, "face_samples": 1},
    ))

    logger.info("Found %d student(s) to process.", len(students))

    success_count = 0
    skip_count = 0
    fail_count = 0
    start = time.time()

    for doc in students:
        if not args.force and _already_migrated(doc):
            skip_count += 1
            logger.info(
                "SKIP %s (%s) — already has 512-D encodings",
                doc.get("name", "?"),
                doc.get("registration_number", "?"),
            )
            continue

        ok, msg = migrate_student(doc, backend, dry_run=args.dry_run)
        if ok:
            success_count += 1
            logger.info("OK   %s", msg)
        else:
            fail_count += 1
            logger.warning("FAIL %s", msg)

    elapsed = time.time() - start
    logger.info("-" * 60)
    logger.info(
        "Migration complete in %.1fs: %d succeeded, %d skipped, %d failed",
        elapsed, success_count, skip_count, fail_count,
    )
    if args.dry_run:
        logger.info("(DRY RUN — no changes were written to the database)")


if __name__ == "__main__":
    main()
