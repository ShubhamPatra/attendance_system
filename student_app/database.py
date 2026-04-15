"""Student portal database helpers built on the shared MongoDB layer."""

from __future__ import annotations

from bson import ObjectId

import core.database as core_database
from core.auth import hash_password


def create_student_account(
    name: str,
    email: str,
    registration_number: str,
    password: str,
    *,
    semester: int = 1,
    section: str = "",
) -> str:
    """Create a pending student account with a hashed password."""
    student_id = core_database.insert_student(
        name=name,
        semester=semester,
        registration_number=registration_number,
        section=section,
        encodings=[],
        email=email,
        password_hash=hash_password(password),
        verification_status="pending",
        verification_score=0.0,
        face_samples=[],
    )
    return str(student_id)


def get_student_by_id(student_id, include_sensitive: bool = False):
    if isinstance(student_id, str):
        try:
            student_id = ObjectId(student_id)
        except Exception:
            return None
    return core_database.get_student_by_id(student_id, include_sensitive=include_sensitive)


def get_student_by_reg_no(reg_no: str, include_sensitive: bool = False):
    return core_database.get_student_by_reg_no(reg_no, include_sensitive=include_sensitive)


def get_student_by_email(email: str, include_sensitive: bool = False):
    return core_database.get_student_by_email(email, include_sensitive=include_sensitive)


def get_student_status(reg_no: str) -> dict | None:
    student = core_database.get_student_by_reg_no(reg_no, include_sensitive=False)
    if not student:
        return None
    return {
        "registration_number": student.get("registration_number"),
        "name": student.get("name"),
        "verification_status": student.get("verification_status", "pending"),
        "verification_score": student.get("verification_score", 0.0),
        "verification_reason": student.get("verification_reason", ""),
        "face_samples": student.get("face_samples", []),
        "created_at": student.get("created_at"),
    }


def save_verification_result(reg_no: str, result, sample_paths: list[str]) -> bool:
    """Persist the verification outcome to the student record."""
    return core_database.update_student_verification(
        reg_no,
        result.status,
        score=result.score,
        reason="; ".join(result.reasons) if result.reasons else None,
        face_samples=sample_paths,
        encodings=result.encodings if result.status == "approved" else None,
    )


def finalize_manual_approval(reg_no: str, encodings, score: float | None = None, reason: str | None = None) -> bool:
    return core_database.update_student_verification(
        reg_no,
        "approved",
        score=score,
        reason=reason,
        encodings=encodings,
    )


def reject_student(reg_no: str, reason: str, score: float | None = None) -> bool:
    return core_database.update_student_verification(
        reg_no,
        "rejected",
        score=score,
        reason=reason,
    )


def get_attendance_overview(reg_no: str, date: str | None = None, month: str | None = None) -> dict | None:
    try:
        student = core_database.get_student_by_reg_no(reg_no, include_sensitive=False)
        if not student:
            return None

        records = core_database.get_attendance_by_student(reg_no)
        if date:
            records = [record for record in records if record.get("date") == date]
        if month:
            records = [record for record in records if str(record.get("date", "")).startswith(month)]

        distinct_dates = sorted({record.get("date") for record in records if record.get("date")})
        total_days = len(distinct_dates)
        days_present = sum(1 for record in records if record.get("status") == "Present")
        percentage = round((days_present / total_days) * 100, 1) if total_days else 0.0

        return {
            "name": student.get("name", ""),
            "registration_number": student.get("registration_number", reg_no),
            "semester": student.get("semester", ""),
            "section": student.get("section", ""),
            "verification_status": student.get("verification_status", "pending"),
            "verification_score": student.get("verification_score", 0.0),
            "percentage": percentage,
            "days_present": days_present,
            "days_total": total_days,
            "records": records,
            "filters": {"date": date, "month": month},
        }
    except Exception as e:
        # Re-raise the exception so the route handler can log and display a user-friendly message
        raise
