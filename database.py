"""
Database layer -- MongoDB Atlas connection, indexes, and CRUD helpers.
"""

from datetime import datetime, timedelta, timezone

import bson
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import DuplicateKeyError, ConnectionFailure

import config
from utils import setup_logging, today_str, now_time_str

logger = setup_logging()

# ---------------------------------------------------------------------------
# Connection (lazy singleton)
# ---------------------------------------------------------------------------
_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return a cached MongoClient instance."""
    global _client
    if _client is None:
        _client = MongoClient(
            config.MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        # Verify the connection early
        try:
            _client.admin.command("ping")
            logger.info("Connected to MongoDB Atlas.")
        except ConnectionFailure as exc:
            _client = None
            logger.error("MongoDB connection failed: %s", exc)
            raise
    return _client


def get_db():
    """Return the application database handle."""
    return get_client()[config.DATABASE_NAME]


# ---------------------------------------------------------------------------
# Index setup (idempotent -- safe to call on every startup)
# ---------------------------------------------------------------------------

def ensure_indexes():
    """Create required unique indexes if they don't exist."""
    db = get_db()
    db.students.create_index(
        [("registration_number", ASCENDING)],
        unique=True,
        name="uq_registration_number",
    )
    # Ensure legacy attendance index variant is removed if present.
    try:
        db.attendance.drop_index("uq_student_date_subject")
    except Exception:
        pass

    db.attendance.create_index(
        [("student_id", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uq_student_date",
    )
    db.attendance.create_index(
        [("date", ASCENDING)],
        name="idx_date",
    )
    logger.info("Database indexes ensured.")


# ---------------------------------------------------------------------------
# Students CRUD
# ---------------------------------------------------------------------------

def insert_student(
    name: str,
    semester: int,
    registration_number: str,
    section: str,
    encodings: list[np.ndarray],
) -> bson.ObjectId:
    """Insert a new student with multiple face encodings.

    Returns the inserted _id.
    Raises ValueError on duplicate registration_number.
    """
    db = get_db()
    doc = {
        "name": name,
        "semester": semester,
        "registration_number": registration_number,
        "section": section,
        "encodings": [bson.Binary(e.tobytes()) for e in encodings],
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = db.students.insert_one(doc)
        logger.info("Inserted student %s (%s).", name, registration_number)
        return result.inserted_id
    except DuplicateKeyError:
        raise ValueError(
            f"Registration number '{registration_number}' already exists."
        )


def get_student_by_id(student_id: bson.ObjectId) -> dict | None:
    """Return a student's details by _id (without face_encoding)."""
    db = get_db()
    return db.students.find_one(
        {"_id": student_id},
        {"face_encoding": 0, "encodings": 0, "created_at": 0},
    )


def get_student_by_reg_no(reg_no: str) -> dict | None:
    """Return a student document by registration number."""
    db = get_db()
    return db.students.find_one({"registration_number": reg_no})


def get_all_students() -> list[dict]:
    """Return all students (without encodings) sorted by name."""
    db = get_db()
    return list(
        db.students.find(
            {},
            {"face_encoding": 0, "encodings": 0},
        ).sort("name", ASCENDING)
    )


def get_student_encodings() -> list[tuple]:
    """Return list of (student_id, name, [np.ndarray, ...]) for all students.

    Handles both the new ``encodings`` array field and the legacy
    ``face_encoding`` single-value field for backward compatibility.
    Encodings that don't have exactly 128 elements are skipped with a warning.
    """
    db = get_db()
    results = []
    for doc in db.students.find({}, {"name": 1, "encodings": 1, "face_encoding": 1}):
        enc_list: list[np.ndarray] = []
        if "encodings" in doc and doc["encodings"]:
            for i, raw in enumerate(doc["encodings"]):
                arr = np.frombuffer(raw, dtype=np.float64)
                if arr.shape == (128,):
                    enc_list.append(arr)
                else:
                    logger.warning(
                        "Skipping corrupt encoding %d for student %s "
                        "(shape=%s, expected (128,))",
                        i, doc.get("name", "?"), arr.shape,
                    )
        elif "face_encoding" in doc:
            arr = np.frombuffer(doc["face_encoding"], dtype=np.float64)
            if arr.shape == (128,):
                enc_list.append(arr)
            else:
                logger.warning(
                    "Skipping corrupt legacy encoding for student %s "
                    "(shape=%s, expected (128,))",
                    doc.get("name", "?"), arr.shape,
                )
        if enc_list:
            results.append((doc["_id"], doc["name"], enc_list))
    return results


def append_student_encoding(student_id: bson.ObjectId, new_encoding: np.ndarray) -> bool:
    """Push a new face encoding to the student's encodings array.

    Uses $push with $each/$slice to keep only the last
    ``config.MAX_ENCODINGS_PER_STUDENT`` encodings, automatically
    dropping the oldest when the cap is exceeded.
    """
    db = get_db()
    result = db.students.update_one(
        {"_id": student_id},
        {"$push": {
            "encodings": {
                "$each": [bson.Binary(new_encoding.tobytes())],
                "$slice": -config.MAX_ENCODINGS_PER_STUDENT,
            }
        }}
    )
    return result.modified_count > 0


def student_count() -> int:
    """Return total number of registered students."""
    return get_db().students.count_documents({})


def delete_student(reg_no: str) -> bool:
    """Delete a student and all their attendance records by registration number."""
    db = get_db()
    student = db.students.find_one({"registration_number": reg_no}, {"_id": 1})
    if not student:
        return False
    db.attendance.delete_many({"student_id": student["_id"]})
    result = db.students.delete_one({"_id": student["_id"]})
    return result.deleted_count > 0


# ---------------------------------------------------------------------------
# Attendance CRUD
# ---------------------------------------------------------------------------

def mark_attendance(student_id: bson.ObjectId, confidence: float) -> bool:
    """Mark attendance for *student_id* today.

    Returns True if a new record was inserted, False if already present.
    """
    db = get_db()
    date = today_str()
    doc = {
        "student_id": student_id,
        "date": date,
        "time": now_time_str(),
        "status": "Present",
        "confidence_score": round(confidence, 4),
    }
    try:
        db.attendance.insert_one(doc)
        logger.info(
            "Attendance marked for student %s on %s.", student_id, date
        )
        return True
    except DuplicateKeyError:
        return False


def bulk_upsert_attendance(entries: list[dict]) -> int:
    """Bulk upsert attendance records for today.

    *entries* is a list of dicts with keys: ``student_id`` (ObjectId),
    ``status`` ("Present"/"Absent"), ``confidence_score`` (float, optional).

    Returns the total number of upserted + modified documents.
    """
    db = get_db()
    date = today_str()
    ops = []
    for entry in entries:
        ops.append(UpdateOne(
            {
                "student_id": entry["student_id"],
                "date": date,
            },
            {"$set": {
                "status": entry.get("status", "Present"),
                "time": now_time_str(),
                "confidence_score": entry.get("confidence_score", 0.0),
            }},
            upsert=True,
        ))
    if not ops:
        return 0
    result = db.attendance.bulk_write(ops)
    return result.upserted_count + result.modified_count


def get_attendance(date_str: str | None = None) -> list[dict]:
    """Return attendance records, optionally filtered by date.

    Each record includes the student's name via aggregation.
    """
    db = get_db()
    match_filters: dict = {}

    if date_str:
        match_filters["date"] = date_str

    pipeline = [
        {"$match": match_filters},
        {
            "$lookup": {
                "from": "students",
                "localField": "student_id",
                "foreignField": "_id",
                "as": "student",
            }
        },
        {"$unwind": "$student"},
        {
            "$project": {
                "_id": 0,
                "name": "$student.name",
                "registration_number": "$student.registration_number",
                "section": "$student.section",
                "semester": "$student.semester",
                "date": 1,
                "time": 1,
                "status": 1,
                "confidence_score": 1,
            }
        },
        {"$sort": {"date": -1, "time": -1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def today_attendance_count() -> int:
    """Return number of students marked present today."""
    return get_db().attendance.count_documents({"date": today_str()})


def get_attendance_csv(date_str: str) -> pd.DataFrame:
    """Return attendance for *date_str* as a pandas DataFrame."""
    records = get_attendance(date_str)
    return _records_to_df(records)


# ---------------------------------------------------------------------------
# Extended attendance queries
# ---------------------------------------------------------------------------

def get_attendance_by_student(reg_no: str) -> list[dict]:
    """Return all attendance records for a given registration number."""
    db = get_db()
    student = db.students.find_one(
        {"registration_number": reg_no}, {"_id": 1}
    )
    if not student:
        return []
    return get_attendance_filtered(student_id=student["_id"])


def get_attendance_by_date_range(
    start_date: str, end_date: str
) -> list[dict]:
    """Return attendance records between *start_date* and *end_date* inclusive."""
    db = get_db()
    pipeline = [
        {"$match": {"date": {"$gte": start_date, "$lte": end_date}}},
        {
            "$lookup": {
                "from": "students",
                "localField": "student_id",
                "foreignField": "_id",
                "as": "student",
            }
        },
        {"$unwind": "$student"},
        {
            "$project": {
                "_id": 0,
                "name": "$student.name",
                "registration_number": "$student.registration_number",
                "section": "$student.section",
                "semester": "$student.semester",
                "date": 1,
                "time": 1,
                "status": 1,
                "confidence_score": 1,
            }
        },
        {"$sort": {"date": -1, "time": -1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_attendance_filtered(
    student_id: bson.ObjectId | None = None,
    date_str: str | None = None,
) -> list[dict]:
    """Flexible attendance query with optional student_id and/or date."""
    db = get_db()
    match: dict = {}
    if student_id:
        match["student_id"] = student_id
    if date_str:
        match["date"] = date_str

    pipeline = [
        {"$match": match},
        {
            "$lookup": {
                "from": "students",
                "localField": "student_id",
                "foreignField": "_id",
                "as": "student",
            }
        },
        {"$unwind": "$student"},
        {
            "$project": {
                "_id": 0,
                "name": "$student.name",
                "registration_number": "$student.registration_number",
                "section": "$student.section",
                "semester": "$student.semester",
                "date": 1,
                "time": 1,
                "status": 1,
                "confidence_score": 1,
            }
        },
        {"$sort": {"date": -1, "time": -1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_attendance_by_hour(date_str: str) -> list[dict]:
    """Return attendance counts grouped by hour for a given date.

    Returns list of ``{"hour": 8, "count": 5}`` dicts.
    """
    db = get_db()
    pipeline = [
        {"$match": {"date": date_str}},
        {
            "$project": {
                "hour": {"$toInt": {"$substr": ["$time", 0, 2]}},
            }
        },
        {
            "$group": {
                "_id": "$hour",
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
        {"$project": {"_id": 0, "hour": "$_id", "count": 1}},
    ]
    return list(db.attendance.aggregate(pipeline))


def get_all_registration_numbers() -> list[str]:
    """Return sorted list of all student registration numbers."""
    db = get_db()
    return sorted(
        doc["registration_number"]
        for doc in db.students.find({}, {"registration_number": 1})
    )


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def get_at_risk_students(days: int = 30, threshold: int | None = None) -> list[dict]:
    """Return students whose attendance percentage is below *threshold*.

    Looks at the last *days* calendar days. Each entry contains
    ``name``, ``reg_no``, ``percentage``, ``days_present``, and ``days_total``.
    Results are sorted by percentage ascending (worst first).

    Uses MongoDB aggregation pipeline for O(1) query cost instead of O(N).
    """
    if threshold is None:
        threshold = config.ABSENCE_THRESHOLD
    db = get_db()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Get total class days in the range
    total_class_days = len(db.attendance.distinct(
        "date",
        {"date": {"$gte": start_str, "$lte": end_str}},
    ))

    if total_class_days == 0:
        return []

    # Single aggregation query: group by student, count attendance,
    # join with students collection, filter by threshold
    pipeline = [
        {
            "$match": {
                "date": {"$gte": start_str, "$lte": end_str},
                "status": "Present",
            }
        },
        {
            "$group": {
                "_id": "$student_id",
                "days_present": {"$sum": 1},
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
                "days_total": total_class_days,
                "percentage": {
                    "$cond": [
                        {"$gt": [total_class_days, 0]},
                        {
                            "$round": [
                                {"$multiply": [
                                    {"$divide": ["$days_present", total_class_days]},
                                    100,
                                ]},
                                1,
                            ]
                        },
                        0,
                    ]
                },
            }
        },
        {
            "$match": {
                "percentage": {"$lt": threshold},
            }
        },
        {
            "$project": {
                "_id": 0,
                "name": "$student_info.name",
                "reg_no": "$student_info.registration_number",
                "percentage": 1,
                "days_present": 1,
                "days_total": 1,
            }
        },
        {
            "$sort": {"percentage": 1},
        },
    ]

    results = list(db.attendance.aggregate(pipeline))
    return results


def get_student_attendance_summary(reg_no: str, days: int = 30) -> dict | None:
    """Return an attendance summary for a single student over the last *days* days.

    Returns ``None`` if the student is not found. Otherwise returns a dict
    with student info, attendance percentage, and individual records.
    """
    db = get_db()
    student = db.students.find_one(
        {"registration_number": reg_no},
        {"name": 1, "registration_number": 1, "semester": 1, "section": 1},
    )
    if not student:
        return None

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    records = list(db.attendance.find(
        {"student_id": student["_id"], "date": {"$gte": start_str, "$lte": end_str}},
        {"_id": 0, "date": 1, "time": 1, "status": 1, "confidence_score": 1}
    ).sort("date", -1))

    # Count distinct class dates (days where attendance was recorded for any student)
    total_class_days = db.attendance.distinct(
        "date", {"date": {"$gte": start_str, "$lte": end_str}}
    )
    total = len(total_class_days)

    count = sum(1 for r in records if r.get("status") == "Present")
    pct = round(count / total * 100, 1) if total > 0 else 0

    return {
        "name": student["name"],
        "registration_number": student["registration_number"],
        "semester": student.get("semester", ""),
        "section": student.get("section", ""),
        "percentage": pct,
        "days_present": count,
        "days_total": total,
        "records": records,
    }


def get_attendance_heatmap_data(days: int = 90) -> list[dict]:
    """Return [{date, count, total_students}] for the last N days."""
    db = get_db()
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    total = db.students.count_documents({})

    pipeline = [
        {"$match": {"date": {"$gte": start_str, "$lte": end_str}}},
        {"$group": {"_id": "$date", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
        {"$project": {"_id": 0, "date": "$_id", "count": 1}},
    ]
    results = list(db.attendance.aggregate(pipeline))
    for r in results:
        r["total_students"] = total
    return results


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "Name", "Registration Number", "Section",
    "Semester", "Date", "Time", "Status", "Confidence",
]

_CSV_RENAME_MAP = {
    "name": "Name",
    "registration_number": "Registration Number",
    "section": "Section",
    "semester": "Semester",
    "date": "Date",
    "time": "Time",
    "status": "Status",
    "confidence_score": "Confidence",
}


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert attendance records to a renamed DataFrame."""
    if not records:
        return pd.DataFrame(columns=_CSV_COLUMNS)
    df = pd.DataFrame(records)
    df.rename(columns=_CSV_RENAME_MAP, inplace=True)
    return df


def get_attendance_csv_by_student(reg_no: str) -> pd.DataFrame:
    return _records_to_df(get_attendance_by_student(reg_no))


def get_attendance_csv_by_date_range(
    start_date: str, end_date: str
) -> pd.DataFrame:
    return _records_to_df(get_attendance_by_date_range(start_date, end_date))


def get_attendance_csv_full() -> pd.DataFrame:
    """Export entire attendance history."""
    return _records_to_df(get_attendance())
