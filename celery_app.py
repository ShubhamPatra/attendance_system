"""
Celery application – async task definitions for AutoAttendance.

Provides background tasks for CSV generation, face-encoding computation,
absence notifications (via SendGrid), and MongoDB backups.

Start the worker::

    celery -A celery_app worker --loglevel=info

Start the beat scheduler (periodic tasks)::

    celery -A celery_app beat --loglevel=info
"""

import os
import sys
import base64
import tempfile

# Ensure the package directory is importable when Celery is started
# from outside the attendance_system folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

from utils import setup_logging

load_dotenv()

logger = setup_logging()

# ---------------------------------------------------------------------------
# Celery application instance
# ---------------------------------------------------------------------------
# Read broker/backend URLs directly from the environment so that importing
# this module never triggers the EnvironmentError that config.py raises when
# MONGO_URI is absent (e.g. in CI or on a worker node that only needs Redis).
# ---------------------------------------------------------------------------

celery_broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

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

# ---------------------------------------------------------------------------
# Celery Beat – periodic task schedule
# ---------------------------------------------------------------------------

celery_app.conf.beat_schedule = {
    "check-absent-students": {
        "task": "celery_app.send_absence_notifications",
        "schedule": crontab(hour=20, minute=0),
    },
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
    import database  # lazy import to avoid circular / missing-env issues

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
    """Compute 128-D face encodings for a batch of images.

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
    from face_engine import generate_encoding  # lazy import

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
# Task 3 – absence notifications (SendGrid)
# ═══════════════════════════════════════════════════════════════════════════

@celery_app.task(name="celery_app.send_absence_notifications", bind=True)
def send_absence_notifications(self):
    """Check attendance over the last 30 days and email a summary of
    students whose attendance percentage falls below the configured
    threshold.

    Requires ``SENDGRID_API_KEY`` and ``NOTIFY_EMAIL`` to be set in the
    environment (via *config*).  If either is missing the task logs a
    warning and exits gracefully.
    """
    import config
    import database
    from datetime import datetime, timedelta, timezone

    logger.info("send_absence_notifications started")

    db = database.get_db()

    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=30)
    total_possible_days = 30

    # Use aggregation pipeline instead of N+1 queries (count_documents per student)
    flagged = list(db.attendance.aggregate([
        {
            "$match": {
                "date": {
                    "$gte": start_date.isoformat(),
                    "$lte": today.isoformat(),
                }
            }
        },
        {
            "$group": {
                "_id": "$student_id",
                "attended": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "students",
                "localField": "_id",
                "foreignField": "_id",
                "as": "student_info",
            }
        },
        {
            "$unwind": {
                "path": "$student_info",
                "preserveNullAndEmptyArrays": False,
            }
        },
        {
            "$addFields": {
                "percentage": {
                    "$round": [
                        {"$multiply": [
                            {"$divide": ["$attended", total_possible_days]},
                            100,
                        ]},
                        1,
                    ]
                },
                "total": total_possible_days,
            }
        },
        {
            "$match": {
                "percentage": {"$lt": config.ABSENCE_THRESHOLD}
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": "$student_info.name",
                "registration_number": "$student_info.registration_number",
                "attended": 1,
                "total": 1,
                "percentage": 1,
            }
        },
    ]))

    # Ensure we got the total for logging
    total_students = db.students.count_documents({})

    if total_students == 0:
        logger.info("No students registered – skipping absence check.")
        return "no_students"

    logger.info(
        "Absence check complete: %d / %d students below %d%% threshold.",
        len(flagged),
        total_students,
        config.ABSENCE_THRESHOLD,
    )

    if not flagged:
        return "all_clear"

    # ── Build summary email ───────────────────────────────────────────
    if not config.SENDGRID_API_KEY or not config.NOTIFY_EMAIL:
        logger.warning(
            "SENDGRID_API_KEY or NOTIFY_EMAIL not configured – "
            "skipping email notification."
        )
        return {"flagged": len(flagged), "email_sent": False}

    rows_html = ""
    for s in flagged:
        rows_html += (
            f"<tr>"
            f"<td>{s['name']}</td>"
            f"<td>{s['registration_number']}</td>"
            f"<td>{s['attended']}</td>"
            f"<td>{s['total']}</td>"
            f"<td>{s['percentage']}%</td>"
            f"</tr>"
        )

    html_body = f"""\
<html>
<body>
<h2>Attendance Alert – Students Below {config.ABSENCE_THRESHOLD}% Threshold</h2>
<p>The following students have attendance below the required threshold
over the last 30 days (as of {today.isoformat()}):</p>
<table border="1" cellpadding="6" cellspacing="0">
<thead>
  <tr>
    <th>Name</th>
    <th>Reg #</th>
    <th>Days Attended</th>
    <th>Total Days</th>
    <th>Percentage</th>
  </tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
<p>This is an automated message from AutoAttendance.</p>
</body>
</html>"""

    try:
        import sendgrid
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=config.NOTIFY_EMAIL,
            to_emails=config.NOTIFY_EMAIL,
            subject=f"Attendance Alert – {len(flagged)} student(s) below threshold",
            html_content=html_body,
        )
        sg = sendgrid.SendGridAPIClient(config.SENDGRID_API_KEY)
        response = sg.send(message)
        logger.info(
            "Absence notification email sent (status %s).", response.status_code
        )
    except Exception as exc:
        logger.error("Failed to send absence notification email: %s", exc)
        return {"flagged": len(flagged), "email_sent": False, "error": str(exc)}

    return {"flagged": len(flagged), "email_sent": True}


# ═══════════════════════════════════════════════════════════════════════════
# Task 4 – MongoDB backup
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
    import config
    import database
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
