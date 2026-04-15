"""
Celery application – async task definitions for AutoAttendance.

Provides background tasks for CSV generation, face-encoding computation,
and MongoDB backups.

Start the worker::

    celery -A celery_app worker --loglevel=info

Start the beat scheduler (periodic tasks)::

    celery -A celery_app beat --loglevel=info
"""

import os
import sys
import base64
import tempfile
from pathlib import Path

# Ensure the package directory is importable when Celery is started
# from outside the attendance_system folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

from core.utils import setup_logging

load_dotenv()

logger = setup_logging()

# ---------------------------------------------------------------------------
# Celery application instance
# ---------------------------------------------------------------------------
# Read broker/backend URLs directly from the environment so that importing
# this module never triggers the EnvironmentError that config.py raises when
# MONGO_URI is absent (e.g. in CI or on a worker node that only needs Celery).
# ---------------------------------------------------------------------------

_default_data_dir = os.environ.get(
    "CELERY_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "celery_data"),
)

celery_broker = os.environ.get("CELERY_BROKER_URL", "filesystem://")
celery_backend = os.environ.get(
    "CELERY_RESULT_BACKEND",
    str(Path(os.path.join(_default_data_dir, "results")).resolve().as_uri()),
)

celery_app = Celery(
    "attendance_system",
    broker=celery_broker,
    backend=celery_backend,
)

celery_app.conf.update(
    # Autodiscover tasks inside this module
    include=["celery_app"],
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
)

if celery_broker.startswith("filesystem://"):
    broker_in = os.path.join(_default_data_dir, "broker", "in")
    broker_out = os.path.join(_default_data_dir, "broker", "out")
    broker_processed = os.path.join(_default_data_dir, "broker", "processed")
    result_dir = os.path.join(_default_data_dir, "results")

    for path in (broker_in, broker_out, broker_processed, result_dir):
        os.makedirs(path, exist_ok=True)

    celery_app.conf.broker_transport_options = {
        "data_folder_in": broker_in,
        "data_folder_out": broker_out,
        "data_folder_processed": broker_processed,
    }

# ---------------------------------------------------------------------------
# Celery Beat – periodic task schedule
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    "backup-mongodb": {
        "task": "celery_app.backup_mongodb",
        "schedule": crontab(hour=2, minute=0),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Task 1 – CSV generation
# ═══════════════════════════════════════════════════════════════════════════

@celery_app.task(name="celery_app.generate_csv_task", bind=True)
def generate_csv_task(self, query_type, **kwargs):
    """Generate an attendance CSV in the background.

    Parameters
    ----------
    query_type : str
        One of ``"date"``, ``"student"``, ``"range"``, ``"full"``.
    **kwargs :
        Extra arguments forwarded to the corresponding database helper:

        * ``date``  – ``date_str``
        * ``student`` – ``reg_no``
        * ``range`` – ``start_date``, ``end_date``
        * ``full``  – (none)

    Returns
    -------
    str
        Absolute path to the generated temporary CSV file.
    """
    import app_core.database as database  # lazy import to avoid circular / missing-env issues

    logger.info("generate_csv_task started: query_type=%s, kwargs=%s", query_type, kwargs)

    if query_type == "date":
        date_str = kwargs.get("date_str") or kwargs.get("date")
        if not date_str:
            raise ValueError("'date_str' is required for query_type='date'.")
        df = database.get_attendance_csv(date_str)
    elif query_type == "student":
        df = database.get_attendance_csv_by_student(kwargs["reg_no"])
    elif query_type == "range":
        df = database.get_attendance_csv_by_date_range(
            kwargs["start_date"], kwargs["end_date"]
        )
    elif query_type == "full":
        df = database.get_attendance_csv_full()
    else:
        raise ValueError(f"Unknown query_type: {query_type!r}")

    # Write the DataFrame to a temporary CSV file
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", prefix="attendance_"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()

    logger.info("generate_csv_task completed: %s (%d rows)", tmp.name, len(df))
    return tmp.name


# ═══════════════════════════════════════════════════════════════════════════
# Task 2 – face-encoding computation
# ═══════════════════════════════════════════════════════════════════════════

@celery_app.task(name="celery_app.compute_encodings_task", bind=True)
def compute_encodings_task(self, image_paths):
    """Compute face encodings for a batch of images.

    The embedding dimensionality depends on the active backend
    (512-D for ArcFace, 128-D for dlib).

    Parameters
    ----------
    image_paths : list[str]
        File paths to individual face images.

    Returns
    -------
    dict
        ``{"encodings": [...], "errors": [...]}``

        *encodings* is a list of base64-encoded bytes (one per successful
        image).  *errors* contains ``{"path": ..., "error": ...}`` entries
        for images that could not be processed.
    """
    from app_vision.face_engine import generate_encoding  # lazy import

    logger.info("compute_encodings_task started: %d images", len(image_paths))

    results = []
    errors = []

    for path in image_paths:
        try:
            encoding = generate_encoding(path)
            # Encode the numpy array bytes as base64 so the result is
            # JSON-serializable.
            encoded_b64 = base64.b64encode(encoding.tobytes()).decode("ascii")
            results.append(encoded_b64)
        except ValueError as exc:
            logger.warning("Encoding failed for %s: %s", path, exc)
            errors.append({"path": path, "error": str(exc)})

    logger.info(
        "compute_encodings_task completed: %d ok, %d failed",
        len(results),
        len(errors),
    )
    return {"encodings": results, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════════
# Task 3 – MongoDB backup
# ═══════════════════════════════════════════════════════════════════════════

@celery_app.task(name="celery_app.backup_mongodb", bind=True)
def backup_mongodb(self):
    """Export students and attendance collections to JSON, compress into a
    ``.tar.gz`` archive, and clean up old backups beyond the retention
    window.

    Binary encoding fields are excluded from the students export because
    they are not JSON-serializable.  ``ObjectId`` values are converted to
    plain strings.
    """
    import json
    import tarfile
    import shutil
    import app_core.config as config
    import app_core.database as database
    from datetime import datetime, timedelta, timezone

    logger.info("backup_mongodb started")

    db = database.get_db()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    backup_base = config.BACKUP_DIR
    backup_dir = os.path.join(backup_base, today_str)

    os.makedirs(backup_dir, exist_ok=True)
    logger.info("Created backup directory: %s", backup_dir)

    # ── Export students (exclude binary encoding fields, stream to avoid large memory use) ──────────────
    students_path = os.path.join(backup_dir, "students.json")
    student_count = 0
    with open(students_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for idx, doc in enumerate(db.students.find({})):
            record = {}
            for key, value in doc.items():
                if key in ("encodings", "face_encoding"):
                    continue  # skip binary encoding fields
                if key == "_id":
                    record[key] = str(value)
                else:
                    record[key] = value
            # Convert datetime objects to ISO strings for JSON serialization
            if "created_at" in record and hasattr(record["created_at"], "isoformat"):
                record["created_at"] = record["created_at"].isoformat()

            if idx > 0:
                f.write(",\n")
            json.dump(record, f, default=str)
            student_count += 1
        f.write("\n]")
    logger.info("Exported %d students to %s (stream mode)", student_count, students_path)

    # ── Export attendance (stream to avoid large memory use) ─────────────────────────────────────────
    attendance_path = os.path.join(backup_dir, "attendance.json")
    attendance_count = 0
    with open(attendance_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for idx, doc in enumerate(db.attendance.find({})):
            record = {}
            for key, value in doc.items():
                if key == "_id":
                    record[key] = str(value)
                elif key == "student_id":
                    record[key] = str(value)
                else:
                    record[key] = value

            if idx > 0:
                f.write(",\n")
            json.dump(record, f, default=str)
            attendance_count += 1
        f.write("\n]")
    logger.info("Exported %d attendance records to %s (stream mode)", attendance_count, attendance_path)

    # ── Create .tar.gz archive ────────────────────────────────────────
    archive_path = os.path.join(backup_base, f"{today_str}.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(backup_dir, arcname=today_str)
    logger.info("Archive created: %s", archive_path)

    # ── Remove the uncompressed directory ─────────────────────────────
    shutil.rmtree(backup_dir)
    logger.info("Removed uncompressed backup directory: %s", backup_dir)

    # ── Clean up old archives ─────────────────────────────────────────
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.BACKUP_RETENTION_DAYS)
    removed = 0
    for filename in os.listdir(backup_base):
        if not filename.endswith(".tar.gz"):
            continue
        date_part = filename.replace(".tar.gz", "")
        try:
            archive_date = datetime.strptime(date_part, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if archive_date < cutoff:
            old_path = os.path.join(backup_base, filename)
            os.remove(old_path)
            removed += 1
            logger.info("Removed old backup: %s", old_path)

    logger.info(
        "backup_mongodb completed: archive=%s, old_removed=%d",
        archive_path,
        removed,
    )
    return {"archive": archive_path, "students": student_count, "attendance": attendance_count, "old_removed": removed}
