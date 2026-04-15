from datetime import datetime

from core.utils import (
    check_password,
    format_datetime,
    generate_id,
    hash_password,
    validate_email,
    validate_file_upload,
    validate_student_id,
)


def test_password_hash_roundtrip():
    hashed = hash_password("StrongPass123!")
    assert isinstance(hashed, str)
    assert check_password("StrongPass123!", hashed)
    assert not check_password("wrong", hashed)


def test_generate_id_unique_hex():
    first = generate_id()
    second = generate_id()
    assert first != second
    assert len(first) == 32
    int(first, 16)


def test_format_datetime_iso():
    value = datetime(2026, 4, 15, 10, 30, 0)
    assert format_datetime(value) == "2026-04-15T10:30:00"


def test_validate_email():
    assert validate_email("user@example.com")
    assert not validate_email("invalid-email")


def test_validate_student_id():
    assert validate_student_id("CS_2026_001")
    assert not validate_student_id("bad id")


def test_validate_file_upload():
    assert validate_file_upload("photo.jpg")
    assert not validate_file_upload("photo.exe")
